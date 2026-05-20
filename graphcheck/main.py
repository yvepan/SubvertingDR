from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json

from .graph_parser import parse_graph, parsed_graph_payload
from .resolver import resolve_experiment_context
from .scoring import load_injected_claims, score_items
from .stats import generate_statistics
from .writer import write_csv, write_json


TASK_FIELDNAMES = [
    "run_id",
    "setting",
    "doc_count",
    "subquestion_id",
    "depth",
    "parent_subquestion_id",
    "subquestion_text",
    "graph_path",
    "planning_web_refs",
    "planning_local_refs",
    "planning_total_refs",
    "research_web_refs",
    "research_local_refs",
    "research_total_refs",
    "total_refs",
    "planning_poison_refs",
    "research_poison_refs",
    "poison_refs",
    "poison_ratio",
    "planning_poison_ratio",
    "research_poison_ratio",
    "asr_poisoned",
    "asr_best_status",
    "asr_matched_types",
    "asr_matched_claim_ids",
    "asr_justifications",
]

LEARNING_FIELDNAMES = [
    "run_id",
    "setting",
    "doc_count",
    "learning_id",
    "learning_text",
    "learning_source_url",
    "learning_link_is_poison",
    "source_subtask_id",
    "source_subtask_text",
    "source_subtask_asr_poisoned",
    "graph_path",
    "asr_poisoned",
    "asr_best_status",
    "asr_matched_types",
    "asr_matched_claim_ids",
    "asr_justifications",
]


def _task_row(task) -> dict:
    return {
        "run_id": task.run_id,
        "setting": task.setting,
        "doc_count": task.doc_count,
        "subquestion_id": task.item_id,
        "depth": task.depth,
        "parent_subquestion_id": task.parent_item_id or "",
        "subquestion_text": task.text,
        "graph_path": task.graph_path,
        "planning_web_refs": task.planning_web_refs,
        "planning_local_refs": task.planning_local_refs,
        "planning_total_refs": task.planning_total_refs,
        "research_web_refs": task.research_web_refs,
        "research_local_refs": task.research_local_refs,
        "research_total_refs": task.research_total_refs,
        "total_refs": task.total_refs,
        "planning_poison_refs": task.planning_poison_refs,
        "research_poison_refs": task.research_poison_refs,
        "poison_refs": task.poison_refs,
        "poison_ratio": f"{task.poison_ratio:.6f}",
        "planning_poison_ratio": f"{task.planning_poison_ratio:.6f}",
        "research_poison_ratio": f"{task.research_poison_ratio:.6f}",
        "asr_poisoned": "" if task.asr_poisoned is None else str(task.asr_poisoned),
        "asr_best_status": task.asr_best_status or "",
        "asr_matched_types": json.dumps(task.asr_matched_types, ensure_ascii=False),
        "asr_matched_claim_ids": json.dumps(task.asr_matched_claim_ids, ensure_ascii=False),
        "asr_justifications": json.dumps(task.asr_justifications, ensure_ascii=False),
    }


def _learning_row(learning) -> dict:
    return {
        "run_id": learning.run_id,
        "setting": learning.setting,
        "doc_count": learning.doc_count,
        "learning_id": learning.item_id,
        "learning_text": learning.text,
        "learning_source_url": learning.source_url or "",
        "learning_link_is_poison": str(learning.link_is_poison),
        "source_subtask_id": learning.source_subtask_item_id or "",
        "source_subtask_text": learning.source_subtask_text or "",
        "source_subtask_asr_poisoned": "" if learning.source_subtask_asr_poisoned is None else str(learning.source_subtask_asr_poisoned),
        "graph_path": learning.graph_path,
        "asr_poisoned": "" if learning.asr_poisoned is None else str(learning.asr_poisoned),
        "asr_best_status": learning.asr_best_status or "",
        "asr_matched_types": json.dumps(learning.asr_matched_types, ensure_ascii=False),
        "asr_matched_claim_ids": json.dumps(learning.asr_matched_claim_ids, ensure_ascii=False),
        "asr_justifications": json.dumps(learning.asr_justifications, ensure_ascii=False),
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tasks_from_scoring(path: Path) -> list:
    payload = _read_json(path)
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"Invalid scoring cache format: {path}")
    from .models import TaskRecord

    return [TaskRecord(**item) for item in items if isinstance(item, dict)]


def _load_learnings_from_scoring(path: Path) -> list:
    payload = _read_json(path)
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"Invalid scoring cache format: {path}")
    from .models import LearningRecord

    return [LearningRecord(**item) for item in items if isinstance(item, dict)]


def _load_parsed_graph(path: Path):
    payload = _read_json(path)
    from .models import LearningRecord, ParsedGraph, TaskRecord

    tasks = [TaskRecord(**item) for item in payload.get("tasks", []) if isinstance(item, dict)]
    learnings = [LearningRecord(**item) for item in payload.get("learnings", []) if isinstance(item, dict)]
    return ParsedGraph(
        run_id=str(payload.get("run_id", "")),
        setting=str(payload.get("setting", "")),
        doc_count=int(payload.get("doc_count", 0)),
        graph_path=Path(str(payload.get("graph_path", ""))),
        tasks=tasks,
        learnings=learnings,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit graph subquestions and learnings using atomic ASR poisoning matches."
    )
    parser.add_argument("--asr-output-dir", required=True, type=Path)
    parser.add_argument(
        "--graph-experiment-dir",
        type=Path,
        help="Optional. Manually specify the dataset experiment directory used to resolve graph files.",
    )
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env", default="ATOMIC_ASR_API_KEY")
    parser.add_argument(
        "--score-max-workers",
        type=int,
        default=1,
        help="Retained for compatibility. Single-pass matching now uses one full-pool request per batch.",
    )
    parser.add_argument(
        "--score-type-attempts",
        type=int,
        default=2,
        help="How many times to retry a full-pool matching batch after request-level retries fail.",
    )
    parser.add_argument(
        "--skip-asr",
        action="store_true",
        help="Only parse graphs and generate source statistics without LLM-based ASR scoring.",
    )
    parser.add_argument(
        "--reuse-intermediate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing files under new_parsed/new_scoring when available (default: true).",
    )
    parser.add_argument(
        "--exclude-local-data",
        action="store_true",
        help="Exclude runs where setting=local. Only clean/web runs will be processed.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    context = resolve_experiment_context(
        args.asr_output_dir,
        graph_experiment_dir=args.graph_experiment_dir,
    )

    if args.exclude_local_data:
        context.runs = [run for run in context.runs if run.setting != "local"]
        if not context.runs:
            raise ValueError("No runs remaining after applying --exclude-local-data filter.")

    injected_claims = load_injected_claims(context.asr_output_dir)

    if not args.skip_asr and (not args.model or not args.base_url):
        raise ValueError("ASR scoring requires both --model and --base-url unless --skip-asr is used.")

    all_task_rows: list[dict] = []
    all_learning_rows: list[dict] = []

    for run in context.runs:
        parsed_cache_path = context.new_parsed_dir / f"new_{run.run_id}_graph.json"
        scoring_task_path = context.new_scoring_dir / f"new_{run.run_id}_subquestions.json"
        scoring_learning_path = context.new_scoring_dir / f"new_{run.run_id}_learnings.json"

        if args.reuse_intermediate and scoring_task_path.exists() and scoring_learning_path.exists():
            cached_tasks = _load_tasks_from_scoring(scoring_task_path)
            cached_learnings = _load_learnings_from_scoring(scoring_learning_path)
            all_task_rows.extend(_task_row(task) for task in cached_tasks)
            all_learning_rows.extend(_learning_row(learning) for learning in cached_learnings)
            continue

        if args.reuse_intermediate and parsed_cache_path.exists():
            parsed_graph = _load_parsed_graph(parsed_cache_path)
        else:
            parsed_graph = parse_graph(run)
            write_json(
                parsed_cache_path,
                parsed_graph_payload(parsed_graph),
            )

        task_scores = {}
        learning_scores = {}
        if not args.skip_asr:
            task_scores = score_items(
                query_text=context.query_text,
                items=[(task.item_id, task.text) for task in parsed_graph.tasks],
                injected_claims=injected_claims,
                run_id=run.run_id,
                kind="subquestions",
                trace_dir=context.new_traces_dir,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                api_key_env=args.api_key_env,
                score_max_workers=args.score_max_workers,
                score_type_attempts=args.score_type_attempts,
            )
            learning_scores = score_items(
                query_text=context.query_text,
                items=[(learning.item_id, learning.text) for learning in parsed_graph.learnings],
                injected_claims=injected_claims,
                run_id=run.run_id,
                kind="learnings",
                trace_dir=context.new_traces_dir,
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                api_key_env=args.api_key_env,
                score_max_workers=args.score_max_workers,
                score_type_attempts=args.score_type_attempts,
            )

        task_poison_lookup: dict[str, bool | None] = {}
        for task in parsed_graph.tasks:
            score = task_scores.get(task.item_id)
            if score is not None:
                task.asr_poisoned = score.poisoned
                task.asr_best_status = score.best_status
                task.asr_matched_types = score.matched_types
                task.asr_matched_claim_ids = score.matched_claim_ids
                task.asr_justifications = score.justifications
            task_poison_lookup[task.item_id] = task.asr_poisoned
            all_task_rows.append(_task_row(task))

        for learning in parsed_graph.learnings:
            if learning.source_subtask_item_id:
                learning.source_subtask_asr_poisoned = task_poison_lookup.get(learning.source_subtask_item_id)
            score = learning_scores.get(learning.item_id)
            if score is not None:
                learning.asr_poisoned = score.poisoned
                learning.asr_best_status = score.best_status
                learning.asr_matched_types = score.matched_types
                learning.asr_matched_claim_ids = score.matched_claim_ids
                learning.asr_justifications = score.justifications
            all_learning_rows.append(_learning_row(learning))

        write_json(
            scoring_task_path,
            {
                "run_id": run.run_id,
                "setting": run.setting,
                "doc_count": run.doc_count,
                "items": [asdict(task) for task in parsed_graph.tasks],
            },
        )
        write_json(
            scoring_learning_path,
            {
                "run_id": run.run_id,
                "setting": run.setting,
                "doc_count": run.doc_count,
                "items": [asdict(learning) for learning in parsed_graph.learnings],
            },
        )

    write_csv(
        context.new_summary_dir / "new_subquestions_summary.csv",
        all_task_rows,
        TASK_FIELDNAMES,
    )
    write_csv(
        context.new_summary_dir / "new_learnings_summary.csv",
        all_learning_rows,
        LEARNING_FIELDNAMES,
    )

    manifest = {
        "asr_output_dir": str(context.asr_output_dir),
        "dataset_experiment_dir": str(context.dataset_experiment_dir),
        "query_id": context.query_id,
        "query_text": context.query_text,
        "query_text_fragment": context.query_text_fragment,
        "runs": [asdict(run) for run in context.runs],
        "skip_asr": args.skip_asr,
        "injected_claim_count": len(injected_claims),
    }

    injected_claim_lookup = {
        claim.claim_id: {
            "claim_type": claim.claim_type,
            "text": claim.text,
        }
        for claim in injected_claims
    }

    write_json(
        context.new_summary_dir / "new_run_manifest.json",
        manifest,
    )

    generate_statistics(
        summary_dir=context.new_summary_dir,
        task_rows=all_task_rows,
        learning_rows=all_learning_rows,
        manifest=manifest,
        injected_claim_lookup=injected_claim_lookup,
    )


if __name__ == "__main__":
    main()
