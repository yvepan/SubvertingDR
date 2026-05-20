import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tools.paths import DATASET_ROOT, OUTPUTS_ROOT, REPO_ROOT, require_repo_root

PROJECT_ROOT = REPO_ROOT
TEMP_ROOT = OUTPUTS_ROOT / "temp_web_poison_docs"
ARCHIVE_ROOT = OUTPUTS_ROOT / "batch_web_poison"
EMPTY_LOCAL_DIR = OUTPUTS_ROOT / "empty_local_poison_docs"

# One subdirectory per topic category from paper Table 2.
# Each subdirectory contains one sub-subdirectory per query; the sub-subdirectory
# name IS the query string passed verbatim to the deep-research agent.
# Expected layout under FORGE_DATASET_ROOT:
#   ControversialIssues/
#     Which policy priority should come first .../   ← directory name = query text
#       p1_doc_title.txt                             ← adversarial documents
#       p2_doc_title.txt
#       ...
#   FactualSurveys/
#     ...
CATEGORY_DIRS = [
    DATASET_ROOT / "ControversialIssues",
    DATASET_ROOT / "FactualSurveys",
    DATASET_ROOT / "HistoricalDevelopment",
    DATASET_ROOT / "TrendForecasting",
    DATASET_ROOT / "MethodComparison",
]

POISON_DOC_COUNT = 3
DEPTH_VALUES = range(1, 5)
BREADTH_VALUE = 2
BASE_ARGS = [
    "--report_type", "deep",
    "--report_source", "hybrid",
    "--deep_research_breadth", str(BREADTH_VALUE),
    "--no-pdf",
    "--no-docx",
]
REQUIRED_OUTPUT_SUFFIXES = (".json", ".md", "_graph.md")
SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".doc", ".docx", ".pptx", ".csv", ".xls", ".xlsx", ".md", ".html"
}


def iter_query_dirs(category_dir: Path) -> list[Path]:
    entries = []
    with os_scandir(category_dir) as scanner:
        for entry in scanner:
            if entry.is_dir():
                entries.append(Path(entry.path))
    return sorted(entries, key=lambda path: path.name)


def get_poison_docs(query_dir: Path) -> list[Path]:
    docs = []
    with os_scandir(query_dir) as scanner:
        for entry in scanner:
            if not entry.is_file():
                continue
            path = Path(entry.path)
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            docs.append(path)
    docs.sort(key=lambda item: item.name)
    return docs[:POISON_DOC_COUNT]


def build_temp_dir(category_name: str, query_name: str) -> Path:
    digest = hashlib.md5(query_name.encode("utf-8")).hexdigest()[:8]
    return TEMP_ROOT / f"{category_name}_{digest}"


def with_extended_path_prefix(path: Path | str) -> str:
    raw = str(path)
    if raw.startswith("\\\\?\\"):
        return raw
    if raw.startswith("\\\\"):
        return "\\\\?\\UNC\\" + raw[2:]
    return "\\\\?\\" + raw


def os_scandir(path: Path | str):
    return os.scandir(with_extended_path_prefix(path))


def slugify(text: str) -> str:
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in text)
    collapsed = "_".join(part for part in sanitized.split("_") if part)
    base = collapsed[:36] or "query"
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{base}_{digest}"


def prepare_temp_web_poison_dir(query_dir: Path, category_name: str) -> Path:
    selected_docs = get_poison_docs(query_dir)
    if len(selected_docs) < POISON_DOC_COUNT:
        raise ValueError(
            f"Query directory '{query_dir}' has only {len(selected_docs)} supported documents; "
            f"expected at least {POISON_DOC_COUNT}."
        )

    temp_dir = build_temp_dir(category_name, query_dir.name)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    for doc in selected_docs:
        shutil.copy2(with_extended_path_prefix(doc), temp_dir / doc.name)

    return temp_dir


def build_command(query: str, depth: int, web_poison_dir: Path) -> list[str]:
    return [
        sys.executable,
        "cli.py",
        query,
        *BASE_ARGS,
        "--deep_research_depth", str(depth),
        "--doc_path_web_poison", str(web_poison_dir),
        "--doc_path_local_poison", str(EMPTY_LOCAL_DIR),
    ]


def snapshot_output_files() -> dict[Path, float]:
    tracked_files: dict[Path, float] = {}
    for path in OUTPUTS_ROOT.iterdir():
        if not path.is_file():
            continue
        tracked_files[path] = path.stat().st_mtime
    return tracked_files


def archive_run_outputs(category_name: str, query_name: str, depth: int, before: dict[Path, float]) -> list[str]:
    query_slug = slugify(query_name)
    run_dir = ARCHIVE_ROOT / category_name / query_slug / f"depth_{depth}"
    run_dir.mkdir(parents=True, exist_ok=True)

    archived_files: list[str] = []
    for path in OUTPUTS_ROOT.iterdir():
        if not path.is_file():
            continue
        if path.name == "run_manifest.jsonl":
            continue
        previous_mtime = before.get(path)
        current_mtime = path.stat().st_mtime
        is_new_file = previous_mtime is None
        is_updated_file = previous_mtime is not None and current_mtime > previous_mtime
        if not (is_new_file or is_updated_file):
            continue

        destination = run_dir / path.name
        if destination.exists():
            Path(with_extended_path_prefix(destination)).unlink()
        shutil.move(with_extended_path_prefix(path), with_extended_path_prefix(destination))
        archived_files.append(str(destination.relative_to(PROJECT_ROOT)))

    return sorted(archived_files)


def append_manifest_record(record: dict) -> None:
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = ARCHIVE_ROOT / "run_manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_manifest_records() -> list[dict]:
    manifest_path = ARCHIVE_ROOT / "run_manifest.jsonl"
    if not manifest_path.exists():
        return []
    return [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def get_latest_record_index(records: list[dict]) -> dict[tuple[str, str, int], dict]:
    latest: dict[tuple[str, str, int], dict] = {}
    for record in records:
        key = (record["category"], record["query_slug"], int(record["depth"]))
        latest[key] = record
    return latest


def has_complete_outputs(category_name: str, query_name: str, depth: int) -> bool:
    run_dir = ARCHIVE_ROOT / category_name / slugify(query_name) / f"depth_{depth}"
    if not os.path.isdir(with_extended_path_prefix(run_dir)):
        return False

    files = []
    with os_scandir(run_dir) as scanner:
        for entry in scanner:
            if entry.is_file():
                files.append(entry.name)
    has_json = any(name.endswith(".json") for name in files)
    has_md = any(name.endswith(".md") and not name.endswith("_graph.md") for name in files)
    has_graph = any(name.endswith("_graph.md") for name in files)
    return has_json and has_md and has_graph


def should_skip_run(category_name: str, query_name: str, depth: int) -> bool:
    return has_complete_outputs(category_name, query_name, depth)


def main() -> None:
    require_repo_root()
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    EMPTY_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    runs: list[tuple[Path, Path, int]] = []
    for category_dir in CATEGORY_DIRS:
        if not category_dir.exists():
            raise FileNotFoundError(f"Category directory not found: {category_dir}")

        for query_dir in iter_query_dirs(category_dir):
            for depth in DEPTH_VALUES:
                runs.append((category_dir, query_dir, depth))

    total_runs = len(runs)
    current_run = 0

    for category_dir, query_dir, depth in runs:
        current_run += 1
        temp_dir = prepare_temp_web_poison_dir(query_dir, category_dir.name)
        selected_docs = [doc.name for doc in get_poison_docs(query_dir)]
        command = build_command(query_dir.name, depth, temp_dir)

        if should_skip_run(category_dir.name, query_dir.name, depth):
            print("=" * 80)
            print(f"[{current_run}/{total_runs}] category={category_dir.name} depth={depth}")
            print(f"query={query_dir.name}")
            print("status=already_completed, skip")
            continue

        print("=" * 80)
        print(f"[{current_run}/{total_runs}] category={category_dir.name} depth={depth}")
        print(f"query={query_dir.name}")
        print(f"web_poison_docs={selected_docs}")
        print("command=" + subprocess.list2cmdline(command))

        before_outputs = snapshot_output_files()
        status = "success"
        archived_files: list[str] = []
        try:
            run_env = os.environ.copy()
            run_env["PYTHONIOENCODING"] = "utf-8"
            subprocess.run(command, cwd=PROJECT_ROOT, env=run_env, check=True)
        except subprocess.CalledProcessError as exc:
            status = f"failed_exit_{exc.returncode}"
            print(f"Run failed with exit code {exc.returncode}")
        except KeyboardInterrupt:
            print("Interrupted by user.")
            raise
        finally:
            archived_files = archive_run_outputs(category_dir.name, query_dir.name, depth, before_outputs)
            record = {
                "timestamp": datetime.now().isoformat(),
                "category": category_dir.name,
                "query": query_dir.name,
                "query_slug": slugify(query_dir.name),
                "depth": depth,
                "breadth": BREADTH_VALUE,
                "report_type": "deep",
                "report_source": "hybrid",
                "selected_docs": selected_docs,
                "command": command,
                "status": status,
                "archived_files": archived_files,
            }
            append_manifest_record(record)
            print(f"archived_files={archived_files}")


if __name__ == "__main__":
    main()
