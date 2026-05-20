from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import median, pstdev
import json

from .writer import write_csv, write_json


BY_RUN_FIELDNAMES = [
    "run_id",
    "setting",
    "doc_count",
    "subquestion_count",
    "learning_count",
    "avg_subquestion_depth",
    "max_subquestion_depth",
    "root_subquestion_count",
    "leaf_subquestion_count",
    "deep_subquestion_count",
    "avg_subquestion_total_refs",
    "avg_subquestion_poison_refs",
    "avg_subquestion_poison_ratio",
    "avg_planning_poison_ratio",
    "avg_research_poison_ratio",
    "source_poison_subquestion_count",
    "subquestion_asr_known_count",
    "subquestion_asr_poisoned_count",
    "subquestion_asr_poisoned_rate",
    "learning_with_source_url_count",
    "learning_with_source_subtask_count",
    "learning_poison_link_count",
    "learning_poison_link_rate",
    "learning_asr_known_count",
    "learning_asr_poisoned_count",
    "learning_asr_poisoned_rate",
    "source_subtask_asr_poisoned_learning_count",
    "source_subtask_to_learning_propagation_rate",
]

GROUP_FIELDNAMES = [
    "group_key",
    "run_count",
    "subquestion_count",
    "learning_count",
    "avg_subquestion_depth",
    "max_subquestion_depth",
    "avg_subquestion_total_refs",
    "avg_subquestion_poison_refs",
    "avg_subquestion_poison_ratio",
    "avg_planning_poison_ratio",
    "avg_research_poison_ratio",
    "source_poison_subquestion_count",
    "subquestion_asr_poisoned_count",
    "subquestion_asr_poisoned_rate",
    "learning_poison_link_count",
    "learning_poison_link_rate",
    "learning_asr_poisoned_count",
    "learning_asr_poisoned_rate",
    "source_subtask_asr_poisoned_learning_count",
    "source_subtask_to_learning_propagation_rate",
]

BY_SETTING_DOC_COUNT_FIELDNAMES = [
    "setting",
    "doc_count",
    *GROUP_FIELDNAMES[1:],
]

BY_DEPTH_FIELDNAMES = [
    "depth",
    "subquestion_count",
    "avg_total_refs",
    "avg_poison_refs",
    "avg_poison_ratio",
    "avg_planning_poison_ratio",
    "avg_research_poison_ratio",
    "source_poison_subquestion_count",
    "asr_known_count",
    "asr_poisoned_count",
    "asr_poisoned_rate",
]

CLAIM_MATCH_FIELDNAMES = [
    "scope",
    "key",
    "match_count",
]

TOP_SUBQUESTION_FIELDNAMES = [
    "run_id",
    "setting",
    "doc_count",
    "subquestion_id",
    "depth",
    "subquestion_text",
    "total_refs",
    "poison_refs",
    "poison_ratio",
    "asr_poisoned",
    "asr_best_status",
    "matched_claim_count",
    "matched_types",
]

TOP_LEARNING_FIELDNAMES = [
    "run_id",
    "setting",
    "doc_count",
    "learning_id",
    "learning_text",
    "learning_source_url",
    "learning_link_is_poison",
    "source_subtask_id",
    "source_subtask_asr_poisoned",
    "asr_poisoned",
    "asr_best_status",
    "matched_claim_count",
    "matched_types",
]

CORE_STORY_FIELDNAMES = [
    "group_type",
    "group_key",
    "run_count",
    "retrieval_poison_ratio_mean",
    "subquestion_asr_rate",
    "learning_asr_rate",
    "source_poison_to_subq_asr_rate",
    "poison_link_to_learning_asr_rate",
    "subtask_to_learning_propagation_rate",
    "child_poisoned_parent_poisoned_rate",
    "child_poisoned_parent_clean_rate",
    "child_poisoned_given_parent_poisoned_rate",
    "child_poisoned_given_parent_clean_rate",
    "parent_child_poison_rate_delta",
    "depth_weighted_subquestion_asr_rate",
    "deep_subquestion_asr_rate",
    "shallow_subquestion_asr_rate",
    "deep_minus_shallow_asr_gap",
]

FINAL_SUMMARY_FIELDNAMES = [
    "run_id",
    "setting",
    "doc_count",
    "group_type",
    "group_key",
    "run_count",
    "retrieval_poison_ratio_mean",
    "subquestion_asr_rate",
    "learning_asr_rate",
    "subtask_to_learning_propagation_rate",
]


QUALITY_CHECK_DESCRIPTIONS = {
    "task_missing_text_count": "Rows where subquestion text is empty.",
    "learning_missing_text_count": "Rows where learning text is empty.",
    "task_missing_graph_path_count": "Rows where subquestion is missing graph_path.",
    "learning_missing_graph_path_count": "Rows where learning is missing graph_path.",
    "task_poison_ratio_out_of_range_count": "Rows where subquestion poison_ratio is not in [0,1].",
    "task_planning_poison_ratio_out_of_range_count": "Rows where subquestion planning_poison_ratio is not in [0,1].",
    "task_research_poison_ratio_out_of_range_count": "Rows where subquestion research_poison_ratio is not in [0,1].",
    "task_poison_refs_gt_total_refs_count": "Rows where subquestion poison_refs exceeds total_refs.",
    "learning_source_subtask_missing_count": "Rows where learning is missing source subtask ID.",
    "learning_source_url_missing_count": "Rows where learning is missing source URL.",
    "task_asr_status_inconsistent_count": "Rows where subquestion asr_poisoned and asr_best_status are semantically inconsistent.",
    "learning_asr_status_inconsistent_count": "Rows where learning asr_poisoned and asr_best_status are semantically inconsistent.",
    "task_match_without_claim_ids_count": "Rows where subquestion status is matched but matched_claim_ids is empty.",
    "learning_match_without_claim_ids_count": "Rows where learning status is matched but matched_claim_ids is empty.",
}


def _parse_bool(value: object) -> bool | None:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return None


def _parse_int(value: object) -> int:
    text = str(value).strip()
    return int(text) if text else 0


def _parse_float(value: object) -> float:
    text = str(value).strip()
    return float(text) if text else 0.0


def _parse_json_list(value: object) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "sum": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "stddev": 0.0,
        }
    return {
        "count": float(len(values)),
        "sum": float(sum(values)),
        "mean": float(sum(values) / len(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "stddev": float(pstdev(values)) if len(values) > 1 else 0.0,
    }


def _normalize_task_rows(task_rows: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for row in task_rows:
        item = {
            "run_id": str(row.get("run_id", "")),
            "setting": str(row.get("setting", "")),
            "doc_count": _parse_int(row.get("doc_count")),
            "subquestion_id": str(row.get("subquestion_id", "")),
            "depth": _parse_int(row.get("depth")),
            "parent_subquestion_id": str(row.get("parent_subquestion_id", "")),
            "subquestion_text": str(row.get("subquestion_text", "")),
            "graph_path": str(row.get("graph_path", "")),
            "planning_web_refs": _parse_int(row.get("planning_web_refs")),
            "planning_local_refs": _parse_int(row.get("planning_local_refs")),
            "planning_total_refs": _parse_int(row.get("planning_total_refs")),
            "research_web_refs": _parse_int(row.get("research_web_refs")),
            "research_local_refs": _parse_int(row.get("research_local_refs")),
            "research_total_refs": _parse_int(row.get("research_total_refs")),
            "total_refs": _parse_int(row.get("total_refs")),
            "planning_poison_refs": _parse_int(row.get("planning_poison_refs")),
            "research_poison_refs": _parse_int(row.get("research_poison_refs")),
            "poison_refs": _parse_int(row.get("poison_refs")),
            "poison_ratio": _parse_float(row.get("poison_ratio")),
            "planning_poison_ratio": _parse_float(row.get("planning_poison_ratio")),
            "research_poison_ratio": _parse_float(row.get("research_poison_ratio")),
            "asr_poisoned": _parse_bool(row.get("asr_poisoned")),
            "asr_best_status": str(row.get("asr_best_status", "")),
            "asr_matched_types": _parse_json_list(row.get("asr_matched_types")),
            "asr_matched_claim_ids": _parse_json_list(row.get("asr_matched_claim_ids")),
            "asr_justifications": _parse_json_list(row.get("asr_justifications")),
        }
        item["source_poison_present"] = item["poison_refs"] > 0
        normalized.append(item)
    return normalized


def _normalize_learning_rows(learning_rows: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for row in learning_rows:
        item = {
            "run_id": str(row.get("run_id", "")),
            "setting": str(row.get("setting", "")),
            "doc_count": _parse_int(row.get("doc_count")),
            "learning_id": str(row.get("learning_id", "")),
            "learning_text": str(row.get("learning_text", "")),
            "learning_source_url": str(row.get("learning_source_url", "")),
            "learning_link_is_poison": _parse_bool(row.get("learning_link_is_poison")),
            "source_subtask_id": str(row.get("source_subtask_id", "")),
            "source_subtask_text": str(row.get("source_subtask_text", "")),
            "source_subtask_asr_poisoned": _parse_bool(row.get("source_subtask_asr_poisoned")),
            "graph_path": str(row.get("graph_path", "")),
            "asr_poisoned": _parse_bool(row.get("asr_poisoned")),
            "asr_best_status": str(row.get("asr_best_status", "")),
            "asr_matched_types": _parse_json_list(row.get("asr_matched_types")),
            "asr_matched_claim_ids": _parse_json_list(row.get("asr_matched_claim_ids")),
            "asr_justifications": _parse_json_list(row.get("asr_justifications")),
        }
        item["has_source_url"] = bool(item["learning_source_url"])
        item["has_source_subtask"] = bool(item["source_subtask_id"])
        normalized.append(item)
    return normalized


def _task_matrix(tasks: list[dict]) -> dict[str, int]:
    matrix = {
        "source0_asr0": 0,
        "source0_asr1": 0,
        "source1_asr0": 0,
        "source1_asr1": 0,
        "asr_unknown": 0,
    }
    for task in tasks:
        asr_poisoned = task["asr_poisoned"]
        if asr_poisoned is None:
            matrix["asr_unknown"] += 1
            continue
        source_flag = 1 if task["source_poison_present"] else 0
        asr_flag = 1 if asr_poisoned else 0
        matrix[f"source{source_flag}_asr{asr_flag}"] += 1
    return matrix


def _parent_asr_matrix(tasks: list[dict]) -> dict[str, int]:
    matrix = {
        "child1_parent1": 0,
        "child1_parent0": 0,
        "child1_parent_unknown": 0,
        "child0_parent1": 0,
        "child0_parent0": 0,
        "child0_parent_unknown": 0,
        "child_unknown": 0,
        "parent_missing": 0,
    }

    task_by_id = {task["subquestion_id"]: task for task in tasks}
    for task in tasks:
        parent_id = task["parent_subquestion_id"]
        if not parent_id:
            continue

        child_asr = task["asr_poisoned"]
        if child_asr is None:
            matrix["child_unknown"] += 1
            continue

        parent_task = task_by_id.get(parent_id)
        if parent_task is None:
            matrix["parent_missing"] += 1
            continue

        parent_asr = parent_task["asr_poisoned"]
        if parent_asr is None:
            matrix[f"child{1 if child_asr else 0}_parent_unknown"] += 1
            continue

        matrix[f"child{1 if child_asr else 0}_parent{1 if parent_asr else 0}"] += 1
    return matrix


def _learning_link_matrix(learnings: list[dict]) -> dict[str, int]:
    matrix = {
        "link0_asr0": 0,
        "link0_asr1": 0,
        "link1_asr0": 0,
        "link1_asr1": 0,
        "asr_unknown": 0,
        "link_unknown": 0,
    }
    for learning in learnings:
        link_flag = learning["learning_link_is_poison"]
        asr_flag = learning["asr_poisoned"]
        if link_flag is None:
            matrix["link_unknown"] += 1
            continue
        if asr_flag is None:
            matrix["asr_unknown"] += 1
            continue
        matrix[f"link{1 if link_flag else 0}_asr{1 if asr_flag else 0}"] += 1
    return matrix


def _learning_source_matrix(learnings: list[dict]) -> dict[str, int]:
    matrix = {
        "source0_asr0": 0,
        "source0_asr1": 0,
        "source1_asr0": 0,
        "source1_asr1": 0,
        "source_unknown": 0,
        "asr_unknown": 0,
    }
    for learning in learnings:
        source_flag = learning["source_subtask_asr_poisoned"]
        asr_flag = learning["asr_poisoned"]
        if source_flag is None:
            matrix["source_unknown"] += 1
            continue
        if asr_flag is None:
            matrix["asr_unknown"] += 1
            continue
        matrix[f"source{1 if source_flag else 0}_asr{1 if asr_flag else 0}"] += 1
    return matrix


def _task_metrics(tasks: list[dict]) -> dict[str, object]:
    child_parent_ids = {task["parent_subquestion_id"] for task in tasks if task["parent_subquestion_id"]}
    roots = [task for task in tasks if not task["parent_subquestion_id"]]
    leaves = [task for task in tasks if task["subquestion_id"] not in child_parent_ids]
    deep_tasks = [task for task in tasks if task["depth"] >= 3]
    asr_known = [task for task in tasks if task["asr_poisoned"] is not None]
    asr_poisoned = [task for task in asr_known if task["asr_poisoned"]]
    explicit = [task for task in tasks if task["asr_best_status"] == "matched_explicit"]
    implicit = [task for task in tasks if task["asr_best_status"] == "matched_implicit"]
    absent = [task for task in tasks if task["asr_best_status"] == "absent"]
    parent_asr_matrix = _parent_asr_matrix(tasks)

    child_poisoned_parent_known = parent_asr_matrix["child1_parent1"] + parent_asr_matrix["child1_parent0"]
    child_poisoned_parent_poisoned_rate = _safe_ratio(
        parent_asr_matrix["child1_parent1"],
        child_poisoned_parent_known,
    )
    child_poisoned_parent_clean_rate = _safe_ratio(
        parent_asr_matrix["child1_parent0"],
        child_poisoned_parent_known,
    )

    parent_poisoned_child_known = parent_asr_matrix["child1_parent1"] + parent_asr_matrix["child0_parent1"]
    parent_clean_child_known = parent_asr_matrix["child1_parent0"] + parent_asr_matrix["child0_parent0"]
    child_poisoned_given_parent_poisoned_rate = _safe_ratio(
        parent_asr_matrix["child1_parent1"],
        parent_poisoned_child_known,
    )
    child_poisoned_given_parent_clean_rate = _safe_ratio(
        parent_asr_matrix["child1_parent0"],
        parent_clean_child_known,
    )

    asr_known_with_depth = [task for task in asr_known if task["depth"] > 0]
    depth_weight_total = sum(task["depth"] for task in asr_known_with_depth)
    depth_weight_poison = sum(task["depth"] for task in asr_known_with_depth if task["asr_poisoned"])
    depth_weighted_subquestion_asr_rate = _safe_ratio(depth_weight_poison, depth_weight_total)

    deep_known = [task for task in asr_known if task["depth"] >= 3]
    shallow_known = [task for task in asr_known if task["depth"] <= 2]
    deep_poison = [task for task in deep_known if task["asr_poisoned"]]
    shallow_poison = [task for task in shallow_known if task["asr_poisoned"]]
    deep_subquestion_asr_rate = _safe_ratio(len(deep_poison), len(deep_known))
    shallow_subquestion_asr_rate = _safe_ratio(len(shallow_poison), len(shallow_known))

    return {
        "count": len(tasks),
        "depth": _numeric_summary([float(task["depth"]) for task in tasks]),
        "total_refs": _numeric_summary([float(task["total_refs"]) for task in tasks]),
        "poison_refs": _numeric_summary([float(task["poison_refs"]) for task in tasks]),
        "poison_ratio": _numeric_summary([task["poison_ratio"] for task in tasks]),
        "planning_poison_ratio": _numeric_summary([task["planning_poison_ratio"] for task in tasks]),
        "research_poison_ratio": _numeric_summary([task["research_poison_ratio"] for task in tasks]),
        "planning_total_refs_sum": sum(task["planning_total_refs"] for task in tasks),
        "research_total_refs_sum": sum(task["research_total_refs"] for task in tasks),
        "planning_poison_refs_sum": sum(task["planning_poison_refs"] for task in tasks),
        "research_poison_refs_sum": sum(task["research_poison_refs"] for task in tasks),
        "source_poison_count": sum(1 for task in tasks if task["source_poison_present"]),
        "zero_ref_count": sum(1 for task in tasks if task["total_refs"] == 0),
        "root_count": len(roots),
        "leaf_count": len(leaves),
        "deep_count": len(deep_tasks),
        "max_depth": max((task["depth"] for task in tasks), default=0),
        "asr_known_count": len(asr_known),
        "asr_poisoned_count": len(asr_poisoned),
        "asr_poisoned_rate": _safe_ratio(len(asr_poisoned), len(asr_known)),
        "matched_explicit_count": len(explicit),
        "matched_implicit_count": len(implicit),
        "absent_count": len(absent),
        "source_vs_asr_matrix": _task_matrix(tasks),
        "parent_asr_matrix": parent_asr_matrix,
        "child_poisoned_parent_known_count": child_poisoned_parent_known,
        "child_poisoned_parent_poisoned_rate": child_poisoned_parent_poisoned_rate,
        "child_poisoned_parent_clean_rate": child_poisoned_parent_clean_rate,
        "parent_poisoned_child_known_count": parent_poisoned_child_known,
        "parent_clean_child_known_count": parent_clean_child_known,
        "child_poisoned_given_parent_poisoned_rate": child_poisoned_given_parent_poisoned_rate,
        "child_poisoned_given_parent_clean_rate": child_poisoned_given_parent_clean_rate,
        "parent_child_poison_rate_delta": (
            child_poisoned_given_parent_poisoned_rate - child_poisoned_given_parent_clean_rate
        ),
        "depth_weighted_subquestion_asr_rate": depth_weighted_subquestion_asr_rate,
        "deep_subquestion_asr_rate": deep_subquestion_asr_rate,
        "shallow_subquestion_asr_rate": shallow_subquestion_asr_rate,
        "deep_minus_shallow_asr_gap": (deep_subquestion_asr_rate - shallow_subquestion_asr_rate),
    }


def _learning_metrics(learnings: list[dict]) -> dict[str, object]:
    asr_known = [learning for learning in learnings if learning["asr_poisoned"] is not None]
    asr_poisoned = [learning for learning in asr_known if learning["asr_poisoned"]]
    explicit = [learning for learning in learnings if learning["asr_best_status"] == "matched_explicit"]
    implicit = [learning for learning in learnings if learning["asr_best_status"] == "matched_implicit"]
    absent = [learning for learning in learnings if learning["asr_best_status"] == "absent"]
    with_source_url = [learning for learning in learnings if learning["has_source_url"]]
    with_source_subtask = [learning for learning in learnings if learning["has_source_subtask"]]
    poison_links = [learning for learning in learnings if learning["learning_link_is_poison"] is True]
    source_poison = [learning for learning in learnings if learning["source_subtask_asr_poisoned"] is True]
    propagated = [
        learning
        for learning in learnings
        if learning["source_subtask_asr_poisoned"] is True and learning["asr_poisoned"] is True
    ]

    return {
        "count": len(learnings),
        "with_source_url_count": len(with_source_url),
        "with_source_url_rate": _safe_ratio(len(with_source_url), len(learnings)),
        "with_source_subtask_count": len(with_source_subtask),
        "with_source_subtask_rate": _safe_ratio(len(with_source_subtask), len(learnings)),
        "poison_link_count": len(poison_links),
        "poison_link_rate": _safe_ratio(len(poison_links), len(learnings)),
        "source_subtask_asr_poisoned_count": len(source_poison),
        "source_subtask_asr_poisoned_rate": _safe_ratio(len(source_poison), len(learnings)),
        "asr_known_count": len(asr_known),
        "asr_poisoned_count": len(asr_poisoned),
        "asr_poisoned_rate": _safe_ratio(len(asr_poisoned), len(asr_known)),
        "matched_explicit_count": len(explicit),
        "matched_implicit_count": len(implicit),
        "absent_count": len(absent),
        "link_vs_asr_matrix": _learning_link_matrix(learnings),
        "source_vs_learning_asr_matrix": _learning_source_matrix(learnings),
        "source_subtask_to_learning_propagation_rate": _safe_ratio(len(propagated), len(source_poison)),
    }


def _claim_match_stats(tasks: list[dict], learnings: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    by_scope: dict[str, Counter] = {
        "task_claim_ids": Counter(),
        "task_claim_types": Counter(),
        "learning_claim_ids": Counter(),
        "learning_claim_types": Counter(),
        "combined_claim_ids": Counter(),
        "combined_claim_types": Counter(),
    }

    for task in tasks:
        for claim_id in task["asr_matched_claim_ids"]:
            by_scope["task_claim_ids"][claim_id] += 1
            by_scope["combined_claim_ids"][claim_id] += 1
        for claim_type in task["asr_matched_types"]:
            by_scope["task_claim_types"][claim_type] += 1
            by_scope["combined_claim_types"][claim_type] += 1

    for learning in learnings:
        for claim_id in learning["asr_matched_claim_ids"]:
            by_scope["learning_claim_ids"][claim_id] += 1
            by_scope["combined_claim_ids"][claim_id] += 1
        for claim_type in learning["asr_matched_types"]:
            by_scope["learning_claim_types"][claim_type] += 1
            by_scope["combined_claim_types"][claim_type] += 1

    rows: list[dict] = []
    json_payload: dict[str, list[dict]] = {}
    for scope, counter in by_scope.items():
        items = [{"key": key, "match_count": count} for key, count in counter.most_common()]
        json_payload[scope] = items
        for item in items:
            rows.append(
                {
                    "scope": scope,
                    "key": item["key"],
                    "match_count": item["match_count"],
                }
            )
    return rows, json_payload


def _quality_checks(tasks: list[dict], learnings: list[dict]) -> dict[str, int]:
    return {
        "task_missing_text_count": sum(1 for task in tasks if not task["subquestion_text"].strip()),
        "learning_missing_text_count": sum(1 for learning in learnings if not learning["learning_text"].strip()),
        "task_missing_graph_path_count": sum(1 for task in tasks if not task["graph_path"].strip()),
        "learning_missing_graph_path_count": sum(1 for learning in learnings if not learning["graph_path"].strip()),
        "task_poison_ratio_out_of_range_count": sum(
            1 for task in tasks if task["poison_ratio"] < 0.0 or task["poison_ratio"] > 1.0
        ),
        "task_planning_poison_ratio_out_of_range_count": sum(
            1 for task in tasks if task["planning_poison_ratio"] < 0.0 or task["planning_poison_ratio"] > 1.0
        ),
        "task_research_poison_ratio_out_of_range_count": sum(
            1 for task in tasks if task["research_poison_ratio"] < 0.0 or task["research_poison_ratio"] > 1.0
        ),
        "task_poison_refs_gt_total_refs_count": sum(
            1 for task in tasks if task["poison_refs"] > task["total_refs"]
        ),
        "learning_source_subtask_missing_count": sum(
            1 for learning in learnings if not learning["source_subtask_id"].strip()
        ),
        "learning_source_url_missing_count": sum(
            1 for learning in learnings if not learning["learning_source_url"].strip()
        ),
        "task_asr_status_inconsistent_count": sum(
            1
            for task in tasks
            if (task["asr_poisoned"] is True and task["asr_best_status"] == "absent")
            or (task["asr_poisoned"] is False and task["asr_best_status"] in {"matched_explicit", "matched_implicit"})
        ),
        "learning_asr_status_inconsistent_count": sum(
            1
            for learning in learnings
            if (learning["asr_poisoned"] is True and learning["asr_best_status"] == "absent")
            or (
                learning["asr_poisoned"] is False
                and learning["asr_best_status"] in {"matched_explicit", "matched_implicit"}
            )
        ),
        "task_match_without_claim_ids_count": sum(
            1
            for task in tasks
            if task["asr_best_status"] in {"matched_explicit", "matched_implicit"}
            and not task["asr_matched_claim_ids"]
        ),
        "learning_match_without_claim_ids_count": sum(
            1
            for learning in learnings
            if learning["asr_best_status"] in {"matched_explicit", "matched_implicit"}
            and not learning["asr_matched_claim_ids"]
        ),
    }


def _group_rows(
    *,
    run_count: int,
    tasks: list[dict],
    learnings: list[dict],
    group_key: str,
) -> dict[str, object]:
    task_metrics = _task_metrics(tasks)
    learning_metrics = _learning_metrics(learnings)
    return {
        "group_key": group_key,
        "run_count": run_count,
        "subquestion_count": task_metrics["count"],
        "learning_count": learning_metrics["count"],
        "avg_subquestion_depth": _format_float(task_metrics["depth"]["mean"]),
        "max_subquestion_depth": int(task_metrics["max_depth"]),
        "avg_subquestion_total_refs": _format_float(task_metrics["total_refs"]["mean"]),
        "avg_subquestion_poison_refs": _format_float(task_metrics["poison_refs"]["mean"]),
        "avg_subquestion_poison_ratio": _format_float(task_metrics["poison_ratio"]["mean"]),
        "avg_planning_poison_ratio": _format_float(task_metrics["planning_poison_ratio"]["mean"]),
        "avg_research_poison_ratio": _format_float(task_metrics["research_poison_ratio"]["mean"]),
        "source_poison_subquestion_count": int(task_metrics["source_poison_count"]),
        "subquestion_asr_poisoned_count": int(task_metrics["asr_poisoned_count"]),
        "subquestion_asr_poisoned_rate": _format_float(task_metrics["asr_poisoned_rate"]),
        "learning_poison_link_count": int(learning_metrics["poison_link_count"]),
        "learning_poison_link_rate": _format_float(learning_metrics["poison_link_rate"]),
        "learning_asr_poisoned_count": int(learning_metrics["asr_poisoned_count"]),
        "learning_asr_poisoned_rate": _format_float(learning_metrics["asr_poisoned_rate"]),
        "source_subtask_asr_poisoned_learning_count": int(learning_metrics["source_subtask_asr_poisoned_count"]),
        "source_subtask_to_learning_propagation_rate": _format_float(
            learning_metrics["source_subtask_to_learning_propagation_rate"]
        ),
    }


def _core_story_row(
    *,
    group_type: str,
    group_key: str,
    run_count: int,
    tasks: list[dict],
    learnings: list[dict],
    clean_subq_asr_rate: float,
    clean_learning_asr_rate: float,
) -> dict[str, object]:
    task_metrics = _task_metrics(tasks)
    learning_metrics = _learning_metrics(learnings)

    retrieval_poison_ratio_mean = float(task_metrics["poison_ratio"]["mean"])

    source_matrix = task_metrics["source_vs_asr_matrix"]
    source_poison_to_subq_asr_rate = _safe_ratio(
        source_matrix["source1_asr1"],
        source_matrix["source1_asr1"] + source_matrix["source1_asr0"],
    )

    link_matrix = learning_metrics["link_vs_asr_matrix"]
    poison_link_to_learning_asr_rate = _safe_ratio(
        link_matrix["link1_asr1"],
        link_matrix["link1_asr1"] + link_matrix["link1_asr0"],
    )

    subq_asr_rate = float(task_metrics["asr_poisoned_rate"])
    learning_asr_rate = float(learning_metrics["asr_poisoned_rate"])

    return {
        "group_type": group_type,
        "group_key": group_key,
        "run_count": run_count,
        "retrieval_poison_ratio_mean": _format_float(retrieval_poison_ratio_mean),
        "subquestion_asr_rate": _format_float(subq_asr_rate),
        "learning_asr_rate": _format_float(learning_asr_rate),
        "source_poison_to_subq_asr_rate": _format_float(source_poison_to_subq_asr_rate),
        "poison_link_to_learning_asr_rate": _format_float(poison_link_to_learning_asr_rate),
        "subtask_to_learning_propagation_rate": _format_float(
            float(learning_metrics["source_subtask_to_learning_propagation_rate"])
        ),
        "child_poisoned_parent_poisoned_rate": _format_float(
            float(task_metrics["child_poisoned_parent_poisoned_rate"])
        ),
        "child_poisoned_parent_clean_rate": _format_float(
            float(task_metrics["child_poisoned_parent_clean_rate"])
        ),
        "child_poisoned_given_parent_poisoned_rate": _format_float(
            float(task_metrics["child_poisoned_given_parent_poisoned_rate"])
        ),
        "child_poisoned_given_parent_clean_rate": _format_float(
            float(task_metrics["child_poisoned_given_parent_clean_rate"])
        ),
        "parent_child_poison_rate_delta": _format_float(
            float(task_metrics["parent_child_poison_rate_delta"])
        ),
        "depth_weighted_subquestion_asr_rate": _format_float(
            float(task_metrics["depth_weighted_subquestion_asr_rate"])
        ),
        "deep_subquestion_asr_rate": _format_float(float(task_metrics["deep_subquestion_asr_rate"])),
        "shallow_subquestion_asr_rate": _format_float(float(task_metrics["shallow_subquestion_asr_rate"])),
        "deep_minus_shallow_asr_gap": _format_float(float(task_metrics["deep_minus_shallow_asr_gap"])),
    }


def _build_run_rows(
    *,
    tasks: list[dict],
    learnings: list[dict],
    runs: list[dict],
) -> list[dict]:
    tasks_by_run: dict[str, list[dict]] = defaultdict(list)
    learnings_by_run: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        tasks_by_run[task["run_id"]].append(task)
    for learning in learnings:
        learnings_by_run[learning["run_id"]].append(learning)

    rows: list[dict] = []
    for run in runs:
        run_id = str(run["run_id"])
        run_tasks = tasks_by_run.get(run_id, [])
        run_learnings = learnings_by_run.get(run_id, [])
        task_metrics = _task_metrics(run_tasks)
        learning_metrics = _learning_metrics(run_learnings)
        rows.append(
            {
                "run_id": run_id,
                "setting": str(run["setting"]),
                "doc_count": int(run["doc_count"]),
                "subquestion_count": int(task_metrics["count"]),
                "learning_count": int(learning_metrics["count"]),
                "avg_subquestion_depth": _format_float(task_metrics["depth"]["mean"]),
                "max_subquestion_depth": int(task_metrics["max_depth"]),
                "root_subquestion_count": int(task_metrics["root_count"]),
                "leaf_subquestion_count": int(task_metrics["leaf_count"]),
                "deep_subquestion_count": int(task_metrics["deep_count"]),
                "avg_subquestion_total_refs": _format_float(task_metrics["total_refs"]["mean"]),
                "avg_subquestion_poison_refs": _format_float(task_metrics["poison_refs"]["mean"]),
                "avg_subquestion_poison_ratio": _format_float(task_metrics["poison_ratio"]["mean"]),
                "avg_planning_poison_ratio": _format_float(task_metrics["planning_poison_ratio"]["mean"]),
                "avg_research_poison_ratio": _format_float(task_metrics["research_poison_ratio"]["mean"]),
                "source_poison_subquestion_count": int(task_metrics["source_poison_count"]),
                "subquestion_asr_known_count": int(task_metrics["asr_known_count"]),
                "subquestion_asr_poisoned_count": int(task_metrics["asr_poisoned_count"]),
                "subquestion_asr_poisoned_rate": _format_float(task_metrics["asr_poisoned_rate"]),
                "learning_with_source_url_count": int(learning_metrics["with_source_url_count"]),
                "learning_with_source_subtask_count": int(learning_metrics["with_source_subtask_count"]),
                "learning_poison_link_count": int(learning_metrics["poison_link_count"]),
                "learning_poison_link_rate": _format_float(learning_metrics["poison_link_rate"]),
                "learning_asr_known_count": int(learning_metrics["asr_known_count"]),
                "learning_asr_poisoned_count": int(learning_metrics["asr_poisoned_count"]),
                "learning_asr_poisoned_rate": _format_float(learning_metrics["asr_poisoned_rate"]),
                "source_subtask_asr_poisoned_learning_count": int(
                    learning_metrics["source_subtask_asr_poisoned_count"]
                ),
                "source_subtask_to_learning_propagation_rate": _format_float(
                    learning_metrics["source_subtask_to_learning_propagation_rate"]
                ),
            }
        )
    return rows


def _top_subquestions(tasks: list[dict], limit: int = 50) -> list[dict]:
    ordered = sorted(
        tasks,
        key=lambda task: (
            task["asr_poisoned"] is not True,
            -task["poison_ratio"],
            -len(task["asr_matched_claim_ids"]),
            -task["poison_refs"],
            task["run_id"],
            task["subquestion_id"],
        ),
    )
    rows: list[dict] = []
    for task in ordered[:limit]:
        rows.append(
            {
                "run_id": task["run_id"],
                "setting": task["setting"],
                "doc_count": task["doc_count"],
                "subquestion_id": task["subquestion_id"],
                "depth": task["depth"],
                "subquestion_text": task["subquestion_text"],
                "total_refs": task["total_refs"],
                "poison_refs": task["poison_refs"],
                "poison_ratio": _format_float(task["poison_ratio"]),
                "asr_poisoned": "" if task["asr_poisoned"] is None else str(task["asr_poisoned"]),
                "asr_best_status": task["asr_best_status"],
                "matched_claim_count": len(task["asr_matched_claim_ids"]),
                "matched_types": json.dumps(task["asr_matched_types"], ensure_ascii=False),
            }
        )
    return rows


def _top_learnings(learnings: list[dict], limit: int = 50) -> list[dict]:
    ordered = sorted(
        learnings,
        key=lambda learning: (
            learning["asr_poisoned"] is not True,
            learning["learning_link_is_poison"] is not True,
            -len(learning["asr_matched_claim_ids"]),
            learning["run_id"],
            learning["learning_id"],
        ),
    )
    rows: list[dict] = []
    for learning in ordered[:limit]:
        rows.append(
            {
                "run_id": learning["run_id"],
                "setting": learning["setting"],
                "doc_count": learning["doc_count"],
                "learning_id": learning["learning_id"],
                "learning_text": learning["learning_text"],
                "learning_source_url": learning["learning_source_url"],
                "learning_link_is_poison": (
                    "" if learning["learning_link_is_poison"] is None else str(learning["learning_link_is_poison"])
                ),
                "source_subtask_id": learning["source_subtask_id"],
                "source_subtask_asr_poisoned": (
                    "" if learning["source_subtask_asr_poisoned"] is None else str(learning["source_subtask_asr_poisoned"])
                ),
                "asr_poisoned": "" if learning["asr_poisoned"] is None else str(learning["asr_poisoned"]),
                "asr_best_status": learning["asr_best_status"],
                "matched_claim_count": len(learning["asr_matched_claim_ids"]),
                "matched_types": json.dumps(learning["asr_matched_types"], ensure_ascii=False),
            }
        )
    return rows


def _build_final_summary_rows(by_run_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in by_run_rows:
        rows.append(
            {
                "run_id": str(row.get("run_id", "")),
                "setting": str(row.get("setting", "")),
                "doc_count": int(row.get("doc_count", 0)),
                "group_type": "run",
                "group_key": str(row.get("run_id", "")),
                "run_count": 1,
                "retrieval_poison_ratio_mean": str(row.get("avg_subquestion_poison_ratio", "0.000000")),
                "subquestion_asr_rate": str(row.get("subquestion_asr_poisoned_rate", "0.000000")),
                "learning_asr_rate": str(row.get("learning_asr_poisoned_rate", "0.000000")),
                "subtask_to_learning_propagation_rate": str(
                    row.get("source_subtask_to_learning_propagation_rate", "0.000000")
                ),
            }
        )
    return rows


def _build_report_markdown(
    *,
    manifest: dict,
    overall: dict,
    by_setting_rows: list[dict],
    by_doc_count_rows: list[dict],
    by_setting_doc_count_rows: list[dict],
    by_run_rows: list[dict],
    top_subquestion_rows: list[dict],
    top_learning_rows: list[dict],
    claim_match_payload: dict[str, list[dict]],
    top_claim_id_details: list[dict],
    core_story_rows: list[dict],
) -> str:
    lines: list[str] = []
    lines.append("# new_graph_asr_audit Statistics Report")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Query text: {manifest.get('query_text', '')}")
    lines.append(f"- Query ID: {manifest.get('query_id', '')}")
    lines.append(f"- Run count: {overall['overview']['run_count']}")
    lines.append(f"- Injected claim count: {overall['overview']['injected_claim_count']}")
    lines.append(f"- Subquestion count: {overall['subquestions']['count']}")
    lines.append(f"- Learning count: {overall['learnings']['count']}")
    lines.append(
        f"- Subquestion ASR poisoned rate: {_format_float(overall['subquestions']['asr_poisoned_rate'])}"
    )
    lines.append(
        f"- Learning ASR poisoned rate: {_format_float(overall['learnings']['asr_poisoned_rate'])}"
    )
    lines.append(
        f"- Subquestion avg poison ratio: {_format_float(overall['subquestions']['poison_ratio']['mean'])}"
    )
    lines.append(
        f"- Learning poison link rate: {_format_float(overall['learnings']['poison_link_rate'])}"
    )
    lines.append("")
    lines.append("## Document Details (by run)")
    lines.append("- Each run is shown individually rather than aggregated by setting / doc_count.")
    lines.append("- Each run corresponds to a report and its graph audit results.")
    for row in by_run_rows:
        lines.append(
            f"- {row['run_id']} [{row['setting']} / doc_count={row['doc_count']}]: "
            f"subquestions={row['subquestion_count']}, learnings={row['learning_count']}, "
            f"subq_asr_rate={row['subquestion_asr_poisoned_rate']}, "
            f"learning_asr_rate={row['learning_asr_poisoned_rate']}, "
            f"avg_poison_ratio={row['avg_subquestion_poison_ratio']}"
        )
    lines.append(“”)
    lines.append(“## Key Explanation Metrics (Compact)”)
    lines.append(“- retrieval_poison_ratio_weighted: Weighted poison ratio during retrieval (sum(poison_refs)/sum(total_refs)).”)
    lines.append(“- subquestion_asr_rate: Fraction of subquestions judged as poisoned by ASR.”)
    lines.append(“- learning_asr_rate: Fraction of learnings judged as poisoned by ASR.”)
    lines.append(“- source_poison_to_subq_asr_rate: Among source-poisoned subquestions, fraction judged by ASR.”)
    lines.append(“- poison_link_to_learning_asr_rate: Among poison-linked learnings, fraction judged by ASR.”)
    lines.append(“- child_poisoned_parent_poisoned_rate: Given child subquestion is ASR-poisoned, fraction where parent is also poisoned.”)
    lines.append(“- child_poisoned_parent_clean_rate: Given child subquestion is ASR-poisoned, fraction where parent is clean.”)
    lines.append(“- child_poisoned_given_parent_poisoned_rate: Given parent is poisoned, fraction where child is ASR-poisoned.”)
    lines.append(“- child_poisoned_given_parent_clean_rate: Given parent is clean, fraction where child is ASR-poisoned.”)
    lines.append(“- parent_child_poison_rate_delta: Difference of above two (larger = stronger parent-to-child propagation).”)
    lines.append(“- depth_weighted_subquestion_asr_rate: Depth-weighted subquestion ASR rate (deeper = higher weight).”)
    lines.append(“- deep_minus_shallow_asr_gap: Deep subquestion ASR rate minus shallow (>0 = deeper more affected).”)
    for row in core_story_rows:
        lines.append(
            f”- [{row['group_type']}] {row['group_key']}: retrieval_poison_ratio_mean={row['retrieval_poison_ratio_mean']}, “
            f”subq_asr_rate={row['subquestion_asr_rate']}, learning_asr_rate={row['learning_asr_rate']}, “
            f”child_poisoned_parent_poisoned_rate={row['child_poisoned_parent_poisoned_rate']}, “
            f”child_poisoned_parent_clean_rate={row['child_poisoned_parent_clean_rate']}, “
            f”child_poisoned_given_parent_poisoned_rate={row['child_poisoned_given_parent_poisoned_rate']}, “
            f”child_poisoned_given_parent_clean_rate={row['child_poisoned_given_parent_clean_rate']}, “
            f”parent_child_poison_rate_delta={row['parent_child_poison_rate_delta']}, “
            f”depth_weighted_subq_asr_rate={row['depth_weighted_subquestion_asr_rate']}, “
            f”deep_minus_shallow_asr_gap={row['deep_minus_shallow_asr_gap']}”
        )
    lines.append("")
    lines.append("## Consistency Matrices")
    lines.append("- Subquestion source poison vs ASR meaning:")
    lines.append("  - source0_asr0: Source clean and ASR not poisoned")
    lines.append("  - source0_asr1: Source clean but ASR poisoned")
    lines.append("  - source1_asr0: Source poisoned but ASR not poisoned")
    lines.append("  - source1_asr1: Source poisoned and ASR poisoned")
    lines.append("  - asr_unknown: ASR result missing")
    lines.append(
        f"- Subquestion source poison vs ASR: {json.dumps(overall['subquestions']['source_vs_asr_matrix'], ensure_ascii=False)}"
    )
    lines.append("- Subquestion parent-child ASR matrix (child=subquestion, parent=parent) meaning:")
    lines.append("  - child1_parent1: Child poisoned, parent poisoned")
    lines.append("  - child1_parent0: Child poisoned, parent not poisoned")
    lines.append("  - child1_parent_unknown: Child poisoned, parent ASR unknown")
    lines.append("  - child0_parent1: Child not poisoned, parent poisoned")
    lines.append("  - child0_parent0: Child not poisoned, parent not poisoned")
    lines.append("  - child0_parent_unknown: Child not poisoned, parent ASR unknown")
    lines.append("  - child_unknown: Child ASR unknown")
    lines.append("  - parent_missing: Parent node missing")
    lines.append(
        f"- Subquestion parent-child ASR matrix: {json.dumps(overall['subquestions']['parent_asr_matrix'], ensure_ascii=False)}"
    )
    lines.append("- Learning poison link vs ASR meaning:")
    lines.append("  - link0_asr0: Link clean and ASR not poisoned")
    lines.append("  - link0_asr1: Link clean but ASR poisoned")
    lines.append("  - link1_asr0: Link poisoned but ASR not poisoned")
    lines.append("  - link1_asr1: Link poisoned and ASR poisoned")
    lines.append("  - link_unknown: Link poison flag missing")
    lines.append("  - asr_unknown: ASR result missing")
    lines.append(
        f"- Learning poison link vs ASR: {json.dumps(overall['learnings']['link_vs_asr_matrix'], ensure_ascii=False)}"
    )
    lines.append("- Source subtask ASR vs Learning ASR meaning:")
    lines.append("  - source0_asr0: Source subtask not poisoned and learning not poisoned")
    lines.append("  - source0_asr1: Source subtask not poisoned but learning poisoned")
    lines.append("  - source1_asr0: Source subtask poisoned but learning not poisoned")
    lines.append("  - source1_asr1: Source subtask poisoned and learning poisoned")
    lines.append("  - source_unknown: Source subtask poison status missing")
    lines.append("  - asr_unknown: Learning ASR result missing")
    lines.append(
        f"- Source subtask ASR vs Learning ASR: {json.dumps(overall['learnings']['source_vs_learning_asr_matrix'], ensure_ascii=False)}"
    )
    lines.append("")
    lines.append("## Most Matched Claim IDs (Top 10)")
    for item in top_claim_id_details:
        lines.append(
            f"- {item['claim_id']}: matches={item['match_count']}, type={item['claim_type']}, text={item['text']}"
        )
    lines.append("")
    lines.append("## Claim Type Match Counts (Top 10)")
    for item in claim_match_payload["combined_claim_types"][:10]:
        lines.append(f"- {item['key']}: {item['match_count']}")
    lines.append("")
    lines.append("## High-Risk Subquestions")
    for row in top_subquestion_rows[:10]:
        lines.append(
            f"- {row['run_id']} / {row['subquestion_id']}: poison_ratio={row['poison_ratio']}, "
            f"asr={row['asr_best_status']}, matched_claims={row['matched_claim_count']}, text={row['subquestion_text']}"
        )
    lines.append("")
    lines.append("## High-Risk Learnings")
    for row in top_learning_rows[:10]:
        lines.append(
            f"- {row['run_id']} / {row['learning_id']}: link_poison={row['learning_link_is_poison']}, "
            f"asr={row['asr_best_status']}, matched_claims={row['matched_claim_count']}, text={row['learning_text']}"
        )
    lines.append("")
    lines.append("## Data Quality")
    for key, value in overall["quality_checks"].items():
        description = QUALITY_CHECK_DESCRIPTIONS.get(key, "(no description)")
        lines.append(f"- {key}: {value} ({description})")
    lines.append("")
    return "\n".join(lines)


def generate_statistics(
    *,
    summary_dir: Path,
    task_rows: list[dict],
    learning_rows: list[dict],
    manifest: dict,
    injected_claim_lookup: dict[str, dict[str, str]] | None = None,
) -> None:
    tasks = _normalize_task_rows(task_rows)
    learnings = _normalize_learning_rows(learning_rows)
    runs = [dict(run) for run in manifest.get("runs", [])]

    tasks_by_setting: dict[str, list[dict]] = defaultdict(list)
    learnings_by_setting: dict[str, list[dict]] = defaultdict(list)
    tasks_by_doc_count: dict[int, list[dict]] = defaultdict(list)
    learnings_by_doc_count: dict[int, list[dict]] = defaultdict(list)
    tasks_by_setting_doc_count: dict[tuple[str, int], list[dict]] = defaultdict(list)
    learnings_by_setting_doc_count: dict[tuple[str, int], list[dict]] = defaultdict(list)
    tasks_by_depth: dict[int, list[dict]] = defaultdict(list)

    run_lookup: dict[str, dict] = {str(run["run_id"]): run for run in runs}
    runs_by_setting: dict[str, set[str]] = defaultdict(set)
    runs_by_doc_count: dict[int, set[str]] = defaultdict(set)
    runs_by_setting_doc_count: dict[tuple[str, int], set[str]] = defaultdict(set)

    for run in runs:
        setting = str(run["setting"])
        doc_count = int(run["doc_count"])
        run_id = str(run["run_id"])
        runs_by_setting[setting].add(run_id)
        runs_by_doc_count[doc_count].add(run_id)
        runs_by_setting_doc_count[(setting, doc_count)].add(run_id)

    for task in tasks:
        tasks_by_setting[task["setting"]].append(task)
        tasks_by_doc_count[task["doc_count"]].append(task)
        tasks_by_setting_doc_count[(task["setting"], task["doc_count"])].append(task)
        tasks_by_depth[task["depth"]].append(task)

    for learning in learnings:
        learnings_by_setting[learning["setting"]].append(learning)
        learnings_by_doc_count[learning["doc_count"]].append(learning)
        learnings_by_setting_doc_count[(learning["setting"], learning["doc_count"])].append(learning)

    overall = {
        "overview": {
            "query_id": manifest.get("query_id", ""),
            "query_text": manifest.get("query_text", ""),
            "run_count": len(runs),
            "settings": sorted(runs_by_setting),
            "doc_counts": sorted(runs_by_doc_count),
            "skip_asr": bool(manifest.get("skip_asr")),
            "injected_claim_count": int(manifest.get("injected_claim_count", 0)),
        },
        "subquestions": _task_metrics(tasks),
        "learnings": _learning_metrics(learnings),
        "quality_checks": _quality_checks(tasks, learnings),
    }

    claim_match_rows, claim_match_payload = _claim_match_stats(tasks, learnings)
    overall["claim_matches"] = claim_match_payload

    claim_lookup = injected_claim_lookup or {}
    top_claim_id_details: list[dict] = []
    for item in claim_match_payload["combined_claim_ids"][:10]:
        claim_id = str(item["key"])
        claim_meta = claim_lookup.get(claim_id, {})
        top_claim_id_details.append(
            {
                "claim_id": claim_id,
                "match_count": int(item["match_count"]),
                "claim_type": str(claim_meta.get("claim_type", "unknown")),
                "text": str(claim_meta.get("text", "")),
            }
        )
    overall["top_claim_id_details"] = top_claim_id_details

    by_run_rows = _build_run_rows(tasks=tasks, learnings=learnings, runs=runs)
    by_setting_rows = [
        _group_rows(
            run_count=len(runs_by_setting[setting]),
            tasks=tasks_by_setting.get(setting, []),
            learnings=learnings_by_setting.get(setting, []),
            group_key=setting,
        )
        for setting in sorted(runs_by_setting)
    ]
    by_doc_count_rows = [
        _group_rows(
            run_count=len(runs_by_doc_count[doc_count]),
            tasks=tasks_by_doc_count.get(doc_count, []),
            learnings=learnings_by_doc_count.get(doc_count, []),
            group_key=str(doc_count),
        )
        for doc_count in sorted(runs_by_doc_count)
    ]
    by_setting_doc_count_rows = [
        {
            "setting": setting,
            "doc_count": doc_count,
            **{
                key: value
                for key, value in _group_rows(
                    run_count=len(runs_by_setting_doc_count[(setting, doc_count)]),
                    tasks=tasks_by_setting_doc_count.get((setting, doc_count), []),
                    learnings=learnings_by_setting_doc_count.get((setting, doc_count), []),
                    group_key=f"{setting}:{doc_count}",
                ).items()
                if key != "group_key"
            },
        }
        for setting, doc_count in sorted(runs_by_setting_doc_count)
    ]
    by_depth_rows = []
    for depth in sorted(tasks_by_depth):
        group_tasks = tasks_by_depth[depth]
        metrics = _task_metrics(group_tasks)
        by_depth_rows.append(
            {
                "depth": depth,
                "subquestion_count": metrics["count"],
                "avg_total_refs": _format_float(metrics["total_refs"]["mean"]),
                "avg_poison_refs": _format_float(metrics["poison_refs"]["mean"]),
                "avg_poison_ratio": _format_float(metrics["poison_ratio"]["mean"]),
                "avg_planning_poison_ratio": _format_float(metrics["planning_poison_ratio"]["mean"]),
                "avg_research_poison_ratio": _format_float(metrics["research_poison_ratio"]["mean"]),
                "source_poison_subquestion_count": metrics["source_poison_count"],
                "asr_known_count": metrics["asr_known_count"],
                "asr_poisoned_count": metrics["asr_poisoned_count"],
                "asr_poisoned_rate": _format_float(metrics["asr_poisoned_rate"]),
            }
        )

    top_subquestion_rows = _top_subquestions(tasks)
    top_learning_rows = _top_learnings(learnings)

    clean_row = next((row for row in by_setting_rows if row["group_key"] == "clean"), None)
    clean_subq_asr_rate = float(clean_row["subquestion_asr_poisoned_rate"]) if clean_row else 0.0
    clean_learning_asr_rate = float(clean_row["learning_asr_poisoned_rate"]) if clean_row else 0.0

    core_story_rows: list[dict] = []
    core_story_rows.append(
        _core_story_row(
            group_type="overall",
            group_key="all",
            run_count=len(runs),
            tasks=tasks,
            learnings=learnings,
            clean_subq_asr_rate=clean_subq_asr_rate,
            clean_learning_asr_rate=clean_learning_asr_rate,
        )
    )
    for row in by_setting_doc_count_rows:
        setting = str(row["setting"])
        doc_count = int(row["doc_count"])
        core_story_rows.append(
            _core_story_row(
                group_type="setting_doc_count",
                group_key=f"{setting}:{doc_count}",
                run_count=int(row["run_count"]),
                tasks=tasks_by_setting_doc_count.get((setting, doc_count), []),
                learnings=learnings_by_setting_doc_count.get((setting, doc_count), []),
                clean_subq_asr_rate=clean_subq_asr_rate,
                clean_learning_asr_rate=clean_learning_asr_rate,
            )
        )

    write_json(summary_dir / "new_stats_overall.json", overall)
    write_csv(summary_dir / "new_stats_by_run.csv", by_run_rows, BY_RUN_FIELDNAMES)
    write_csv(summary_dir / "new_stats_by_setting.csv", by_setting_rows, GROUP_FIELDNAMES)
    write_csv(summary_dir / "new_stats_by_doc_count.csv", by_doc_count_rows, GROUP_FIELDNAMES)
    write_csv(
        summary_dir / "new_stats_by_setting_doc_count.csv",
        by_setting_doc_count_rows,
        BY_SETTING_DOC_COUNT_FIELDNAMES,
    )
    write_csv(summary_dir / "new_stats_by_depth.csv", by_depth_rows, BY_DEPTH_FIELDNAMES)
    write_csv(summary_dir / "new_stats_claim_matches.csv", claim_match_rows, CLAIM_MATCH_FIELDNAMES)
    write_csv(
        summary_dir / "new_stats_top_subquestions.csv",
        top_subquestion_rows,
        TOP_SUBQUESTION_FIELDNAMES,
    )
    write_csv(
        summary_dir / "new_stats_top_learnings.csv",
        top_learning_rows,
        TOP_LEARNING_FIELDNAMES,
    )
    write_csv(
        summary_dir / "new_stats_core_story.csv",
        core_story_rows,
        CORE_STORY_FIELDNAMES,
    )

    final_summary_dir = summary_dir.parent / "new_summary_final"
    final_summary_rows = _build_final_summary_rows(by_run_rows)
    final_summary_payload = {
        "query_id": manifest.get("query_id", ""),
        "query_text": manifest.get("query_text", ""),
        "items": final_summary_rows,
        "overall": overall,
    }
    write_csv(
        final_summary_dir / "new_final_metrics.csv",
        final_summary_rows,
        FINAL_SUMMARY_FIELDNAMES,
    )
    write_json(
        final_summary_dir / "new_final_metrics.json",
        final_summary_payload,
    )
    write_json(
        final_summary_dir / "new_final_overall.json",
        overall,
    )

    report_markdown = _build_report_markdown(
        manifest=manifest,
        overall=overall,
        by_setting_rows=by_setting_rows,
        by_doc_count_rows=by_doc_count_rows,
        by_setting_doc_count_rows=by_setting_doc_count_rows,
        by_run_rows=by_run_rows,
        top_subquestion_rows=top_subquestion_rows,
        top_learning_rows=top_learning_rows,
        claim_match_payload=claim_match_payload,
        top_claim_id_details=top_claim_id_details,
        core_story_rows=core_story_rows,
    )
    (summary_dir / "new_stats_report.md").write_text(report_markdown, encoding="utf-8")
