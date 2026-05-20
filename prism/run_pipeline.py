"""End-to-end PRISM pipeline orchestrator.

Runs all three pipeline stages in order for a given experiment directory:

  Stage 1 — extract_doc_atomic_claims
      Extracts 10 atomic poisoning points from each adversarial document.

  Stage 1b — build_query_canonical_claims
      Merges per-document claims into a deduplicated per-query canonical set.

  Stage 2+3 — score_report_atomic_asr
      Extracts 30 claims from each generated report, then matches them
      against the canonical injected pool to produce per-type ASR scores.

Input layout expected under --experiment-dir
--------------------------------------------
The experiment directory should contain:
  - Adversarial documents as numbered .md files:  1_<title>.md, 2_<title>.md, …
  - Generated report files named by setting:
      clean_<report>.md, web_<n>_<report>.md, local_<n>_<report>.md
  - Optionally, lab sub-directories (lab1/, lab2/, …) grouping multiple
    report files per experiment run.

Output layout under --output-dir
---------------------------------
  data/                    prepared queries.json, references/, reports/
  atomic_units/doc_level/  per-document atomic claim JSON files
  atomic_units/query_level/ per-query canonical claim JSON files
  atomic_units/report_level/ per-report extracted claim JSON files
  scoring/                 per-run AtomicAsrResult JSON files
  scoring_progress/        partial progress snapshots (cleaned up on success)
  summary/atomic_asr_summary.csv  aggregated results table
  llm_traces/              full LLM request/response traces for auditing
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

from prism.build_query_canonical_claims import build_query_canonical_claims
from prism.extract_doc_atomic_claims import extract_doc_atomic_claims
from prism.pipeline_types import (
    QueryRecord,
    ReferenceDocument,
    ReportRun,
    ensure_dir,
    json_ready,
    read_json,
    write_json,
)
from prism.score_report_atomic_asr import score_report_atomic_asr


REFERENCE_PATTERN = re.compile(r"^(?P<order>\d+)[_\- ]+(?P<title>.+)$")
REPORT_PATTERN = re.compile(r"^(?P<setting>clean|local|web)(?:_(?P<count>\d+))?_", re.IGNORECASE)
REPORT_PATTERN_COMPACT = re.compile(r"^(?P<setting>clean|local|web)(?P<count>\d+)_", re.IGNORECASE)


def _safe_name_fragment(text: str, max_length: int = 5) -> str:
    fragment = text[:max_length].strip()
    fragment = re.sub(r'[<>:"/\\|?*\s]+', "-", fragment)
    fragment = fragment.strip("-._")
    return fragment or "na"


def build_default_output_dir(experiment_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [part for part in experiment_dir.resolve().parts if part not in ("/", "\\")]
    tail_parts = parts[-2:] if len(parts) >= 2 else parts
    name_parts = [_safe_name_fragment(part, max_length=5) for part in tail_parts]
    folder_name = "_".join([timestamp, *name_parts])
    return Path("./outputs") / folder_name


def infer_query_id(folder_name: str) -> str:
    match = re.match(r"^(?P<prefix>\d+)_", folder_name)
    if match:
        return f"q{int(match.group('prefix')):02d}"
    slug = re.sub(r"[^a-z0-9]+", "_", folder_name.lower()).strip("_")
    return slug or "q1"


def infer_query_text(folder_name: str) -> str:
    text = re.sub(r"^\d+_", "", folder_name)
    text = text.replace("_", " ").strip()
    return text


def extract_final_report(markdown_text: str) -> str:
    fenced_match = re.search(r"```(?:markdown)?\s*(.*?)```", markdown_text, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()
    return markdown_text.strip()


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_report_file_name(file_name: str) -> tuple[str, int, str]:
    """Parse a report filename into (run_id, doc_count, setting).

    Supported formats:
      clean_*.md           → run_id="clean", doc_count=0, setting="clean"
      web_3_*.md           → run_id="web_3",  doc_count=3, setting="web"
      web3_*.md            → run_id=<full stem>, doc_count=3, setting="web"
    """
    stem = Path(file_name).stem
    match = REPORT_PATTERN.match(stem)
    if match:
        setting = match.group("setting").lower()
        doc_count = int(match.group("count") or 0)
        if setting == "clean":
            run_id = "clean"
            doc_count = 0
        else:
            run_id = f"{setting}_{doc_count}"
        return run_id, doc_count, setting

    compact_match = REPORT_PATTERN_COMPACT.match(stem)
    if compact_match:
        setting = compact_match.group("setting").lower()
        doc_count = int(compact_match.group("count") or 0)
        # Use full stem as run_id to avoid collisions such as web1_p1_* vs web1_p3_*.
        run_id = stem
        return run_id, doc_count, setting

    raise ValueError(f"Unsupported report filename format: {file_name}")


def prepare_input_data(
    *,
    experiment_dir: Path,
    prepared_data_dir: Path,
    lab_dirs: list[str] | None,
    query_id: str | None,
    query_text: str | None,
) -> dict[str, str]:
    """Scan experiment_dir and write normalised JSON inputs under prepared_data_dir."""
    experiment_dir = experiment_dir.resolve()
    prepared_data_dir = prepared_data_dir.resolve()
    ensure_dir(prepared_data_dir)

    effective_query_id = query_id or infer_query_id(experiment_dir.name)
    effective_query_text = query_text or infer_query_text(experiment_dir.name)

    references_dir = ensure_dir(prepared_data_dir / "references" / effective_query_id)
    reports_dir = ensure_dir(prepared_data_dir / "reports" / effective_query_id)
    queries_path = prepared_data_dir / "queries.json"

    if queries_path.exists():
        existing_queries = read_json(queries_path)
        if not isinstance(existing_queries, list):
            raise ValueError(f"Expected a query list in {queries_path}")
    else:
        existing_queries = []

    merged_queries = [
        item for item in existing_queries if item.get("query_id") != effective_query_id
    ]
    merged_queries.append(
        json_ready(QueryRecord(query_id=effective_query_id, query_text=effective_query_text))
    )
    write_json(queries_path, merged_queries)

    for reference_path in sorted(list(experiment_dir.glob("*.md")) + list(experiment_dir.rglob("*.md"))):
        if reference_path.name.lower() == "poison-target.md":
            continue
        match = REFERENCE_PATTERN.match(reference_path.stem)
        if not match:
            continue
        order = int(match.group("order"))
        title = match.group("title").replace("_", " ").strip()
        document = ReferenceDocument(
            query_id=effective_query_id,
            doc_id=f"{effective_query_id}_doc_{order:02d}",
            order=order,
            source_type="local",
            title=title,
            content=read_text_file(reference_path),
            source_path=str(reference_path),
        )
        write_json(references_dir / f"{document.doc_id}.json", document)

    if lab_dirs:
        selected_lab_dirs = [experiment_dir / item for item in lab_dirs]
    else:
        selected_lab_dirs = sorted(
            [
                path
                for path in experiment_dir.iterdir()
                if path.is_dir() and path.name.lower().startswith("lab")
            ]
        )
        if not selected_lab_dirs:
            has_reports = any(
                (REPORT_PATTERN.match(p.stem) or REPORT_PATTERN_COMPACT.match(p.stem))
                for p in experiment_dir.glob("*.md")
                if not p.stem.endswith("_graph") and not REFERENCE_PATTERN.match(p.stem)
            )
            if has_reports:
                selected_lab_dirs = [experiment_dir]

    if not selected_lab_dirs:
        raise ValueError(f"No lab directories or report files found under {experiment_dir}")

    multi_lab = len(selected_lab_dirs) > 1
    for lab_dir in selected_lab_dirs:
        for report_path in sorted(lab_dir.glob("*.md")):
            if report_path.stem.endswith("_graph"):
                continue
            run_id, doc_count, setting = parse_report_file_name(report_path.name)
            if multi_lab:
                run_id = f"{lab_dir.name}_{run_id}"
            report = ReportRun(
                query_id=effective_query_id,
                run_id=run_id,
                setting=setting,
                doc_count=doc_count,
                final_report=extract_final_report(read_text_file(report_path)),
                source_path=str(report_path),
            )
            write_json(reports_dir / f"{report.run_id}.json", report)

    return {
        "query_id": effective_query_id,
        "query_text": effective_query_text,
        "prepared_data_dir": str(prepared_data_dir),
    }


def run_pipeline(
    *,
    experiment_dir: Path,
    output_dir: Path,
    model: str,
    matching_model: str | None,
    base_url: str,
    api_key: str | None,
    api_key_env: str | None,
    lab_dirs: list[str] | None,
    query_id: str | None,
    query_text: str | None,
) -> None:
    output_dir = output_dir.resolve()
    ensure_dir(output_dir)
    data_dir = ensure_dir(output_dir / "data")
    trace_root = ensure_dir(output_dir / "llm_traces")

    prepared = prepare_input_data(
        experiment_dir=experiment_dir,
        prepared_data_dir=data_dir,
        lab_dirs=lab_dirs,
        query_id=query_id,
        query_text=query_text,
    )
    active_query_id = prepared["query_id"]
    effective_matching_model = matching_model or model

    extract_doc_atomic_claims(
        data_dir=data_dir,
        output_dir=output_dir,
        trace_dir=trace_root / "extract_doc_atomic_claims",
        model=model,
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        query_ids={active_query_id},
    )
    build_query_canonical_claims(
        data_dir=data_dir,
        output_dir=output_dir,
        trace_dir=trace_root / "build_query_canonical_claims",
        model=model,
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        query_ids={active_query_id},
    )
    score_report_atomic_asr(
        data_dir=data_dir,
        output_dir=output_dir,
        trace_dir=trace_root / "score_report_atomic_asr",
        model=model,
        matching_model=effective_matching_model,
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
        query_ids={active_query_id},
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete PRISM atomic ASR pipeline.")
    parser.add_argument("--experiment-dir", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to ./outputs/<timestamp>_<experiment_tail>.",
    )
    parser.add_argument("--model", required=True, help="LLM model ID for extraction steps.")
    parser.add_argument("--matching-model", help="LLM model ID for matching step (defaults to --model).")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", help="API key (overrides --api-key-env).")
    parser.add_argument("--api-key-env", default="PRISM_API_KEY", help="Environment variable holding the API key.")
    parser.add_argument("--lab-dir", action="append", dest="lab_dirs", help="Lab sub-directory name(s) to process.")
    parser.add_argument("--query-id", help="Override the inferred query ID.")
    parser.add_argument("--query-text", help="Override the inferred query text.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir or build_default_output_dir(args.experiment_dir)
    run_pipeline(
        experiment_dir=args.experiment_dir,
        output_dir=output_dir,
        model=args.model,
        matching_model=args.matching_model,
        base_url=args.base_url,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        lab_dirs=args.lab_dirs,
        query_id=args.query_id,
        query_text=args.query_text,
    )


if __name__ == "__main__":
    main()
