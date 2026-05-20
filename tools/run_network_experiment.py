"""Batch web-poison experiment runner at a single fixed research depth.

This runner sweeps all queries in the dataset at depth=DEPTH_VALUE (default 2),
producing one report per (category, query) pair.  It is the companion to
``run_depth_web_poison_experiment.py``, which sweeps depths 1–4 for the
depth-ablation results (paper Figure 3).

Use this script for the network-condition baseline where research depth is
held constant so that only the number of poisoning documents (j) varies
(paper Figure 2).

Expected dataset layout is identical to ``run_depth_web_poison_experiment.py``;
see ``data/README.md`` for details.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from tools.paths import DATASET_ROOT, OUTPUTS_ROOT, REPO_ROOT, require_repo_root
from tools.run_depth_web_poison_experiment import (
    BREADTH_VALUE,
    CATEGORY_DIRS,
    POISON_DOC_COUNT,
    SUPPORTED_EXTENSIONS,
    append_manifest_record,
    archive_run_outputs,
    build_temp_dir,
    get_poison_docs,
    iter_query_dirs,
    os_scandir,
    prepare_temp_web_poison_dir,
    should_skip_run,
    slugify,
    snapshot_output_files,
    with_extended_path_prefix,
)

PROJECT_ROOT = REPO_ROOT
ARCHIVE_ROOT = OUTPUTS_ROOT / "batch_web_poison_network"
EMPTY_LOCAL_DIR = OUTPUTS_ROOT / "empty_local_poison_docs"

# Fixed depth for the network-condition runs (paper Figure 2).
DEPTH_VALUE = 2

BASE_ARGS = [
    "--report_type", "deep",
    "--report_source", "hybrid",
    "--deep_research_breadth", str(BREADTH_VALUE),
    "--no-pdf",
    "--no-docx",
]


def build_command(query: str, web_poison_dir: Path) -> list[str]:
    return [
        sys.executable,
        "cli.py",
        query,
        *BASE_ARGS,
        "--deep_research_depth", str(DEPTH_VALUE),
        "--doc_path_web_poison", str(web_poison_dir),
        "--doc_path_local_poison", str(EMPTY_LOCAL_DIR),
    ]


def main() -> None:
    require_repo_root()
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    EMPTY_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[Path, Path]] = []
    for category_dir in CATEGORY_DIRS:
        if not category_dir.exists():
            raise FileNotFoundError(f"Category directory not found: {category_dir}")
        for query_dir in iter_query_dirs(category_dir):
            runs.append((category_dir, query_dir))

    total_runs = len(runs)
    current_run = 0

    for category_dir, query_dir in runs:
        current_run += 1
        temp_dir = prepare_temp_web_poison_dir(query_dir, category_dir.name)
        selected_docs = [doc.name for doc in get_poison_docs(query_dir)]
        command = build_command(query_dir.name, temp_dir)

        if should_skip_run(category_dir.name, query_dir.name, DEPTH_VALUE):
            print("=" * 80)
            print(f"[{current_run}/{total_runs}] category={category_dir.name} depth={DEPTH_VALUE}")
            print(f"query={query_dir.name}")
            print("status=already_completed, skip")
            continue

        print("=" * 80)
        print(f"[{current_run}/{total_runs}] category={category_dir.name} depth={DEPTH_VALUE}")
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
            archived_files = archive_run_outputs(category_dir.name, query_dir.name, DEPTH_VALUE, before_outputs)
            record = {
                "timestamp": datetime.now().isoformat(),
                "category": category_dir.name,
                "query": query_dir.name,
                "query_slug": slugify(query_dir.name),
                "depth": DEPTH_VALUE,
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
