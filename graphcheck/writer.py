from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import csv
import json


FIELDNAME_ZH_MAP = {
    "run_id": "Run ID",
    "setting": "Setting type",
    "doc_count": "Poison document count",
    "subquestion_count": "Subquestion count",
    "learning_count": "Learning count",
    "avg_subquestion_depth": "Avg subquestion depth",
    "max_subquestion_depth": "Max subquestion depth",
    "root_subquestion_count": "Root subquestion count",
    "leaf_subquestion_count": "Leaf subquestion count",
    "deep_subquestion_count": "Deep subquestion count",
    "avg_subquestion_total_refs": "Avg subquestion total refs",
    "avg_subquestion_poison_refs": "Avg subquestion poison refs",
    "avg_subquestion_poison_ratio": "Avg subquestion poison ratio",
    "avg_planning_poison_ratio": "Avg planning poison ratio",
    "avg_research_poison_ratio": "Avg research poison ratio",
    "source_poison_subquestion_count": "Source-poisoned subquestion count",
    "subquestion_asr_known_count": "Subquestion ASR determinable count",
    "subquestion_asr_poisoned_count": "Subquestion ASR poisoned count",
    "subquestion_asr_poisoned_rate": "Subquestion ASR poisoned rate",
    "learning_with_source_url_count": "Learning with source URL count",
    "learning_with_source_subtask_count": "Learning with source subtask count",
    "learning_poison_link_count": "Learning poison link count",
    "learning_poison_link_rate": "Learning poison link rate",
    "learning_asr_known_count": "Learning ASR determinable count",
    "learning_asr_poisoned_count": "Learning ASR poisoned count",
    "learning_asr_poisoned_rate": "Learning ASR poisoned rate",
    "source_subtask_asr_poisoned_learning_count": "Source subtask ASR poisoned learning count",
    "source_subtask_to_learning_propagation_rate": "Source subtask to learning propagation rate",
    "group_key": "Group key",
    "run_count": "Run count",
    "depth": "Depth",
    "avg_total_refs": "Avg total refs",
    "avg_poison_refs": "Avg poison refs",
    "avg_poison_ratio": "Avg poison ratio",
    "asr_known_count": "ASR determinable count",
    "asr_poisoned_count": "ASR poisoned count",
    "asr_poisoned_rate": "ASR poisoned rate",
    "scope": "Scope",
    "key": "Key",
    "match_count": "Match count",
    "subquestion_id": "Subquestion ID",
    "parent_subquestion_id": "Parent subquestion ID",
    "subquestion_text": "Subquestion text",
    "graph_path": "Graph path",
    "planning_web_refs": "Planning web refs",
    "planning_local_refs": "Planning local refs",
    "planning_total_refs": "Planning total refs",
    "research_web_refs": "Research web refs",
    "research_local_refs": "Research local refs",
    "research_total_refs": "Research total refs",
    "total_refs": "Total refs",
    "planning_poison_refs": "Planning poison refs",
    "research_poison_refs": "Research poison refs",
    "poison_refs": "Poison refs",
    "poison_ratio": "Poison ratio",
    "planning_poison_ratio": "Planning poison ratio",
    "research_poison_ratio": "Research poison ratio",
    "asr_poisoned": "ASR poisoned",
    "asr_best_status": "ASR best status",
    "asr_matched_types": "ASR matched types",
    "asr_matched_claim_ids": "ASR matched claim IDs",
    "asr_justifications": "ASR justifications",
    "learning_id": "Learning ID",
    "learning_text": "Learning text",
    "learning_source_url": "Learning source URL",
    "learning_link_is_poison": "Learning link is poison",
    "source_subtask_id": "Source subtask ID",
    "source_subtask_text": "Source subtask text",
    "source_subtask_asr_poisoned": "Source subtask ASR poisoned",
    "matched_claim_count": "Matched claim count",
    "matched_types": "Matched types",
    "group_type": "Group type",
    "retrieval_poison_ratio_weighted": "Retrieval poison ratio (weighted)",
    "retrieval_poison_ratio_mean": "Retrieval poison ratio (mean)",
    "subquestion_asr_rate": "Subquestion ASR rate",
    "learning_asr_rate": "Learning ASR rate",
    "source_poison_to_subq_asr_rate": "Source poison to subquestion ASR rate",
    "poison_link_to_learning_asr_rate": "Poison link to learning ASR rate",
    "subtask_to_learning_propagation_rate": "Subtask to learning propagation rate",
    "child_poisoned_parent_poisoned_rate": "Child poisoned parent poisoned rate",
    "child_poisoned_parent_clean_rate": "Child poisoned parent clean rate",
    "child_poisoned_given_parent_poisoned_rate": "Child poisoned given parent poisoned rate",
    "child_poisoned_given_parent_clean_rate": "Child poisoned given parent clean rate",
    "parent_child_poison_rate_delta": "Parent-child poison rate delta",
    "depth_weighted_subquestion_asr_rate": "Depth-weighted subquestion ASR rate",
    "deep_subquestion_asr_rate": "Deep subquestion ASR rate",
    "shallow_subquestion_asr_rate": "Shallow subquestion ASR rate",
    "deep_minus_shallow_asr_gap": "Deep minus shallow ASR gap",
    "subq_asr_uplift_vs_clean": "Subquestion ASR uplift vs clean",
    "learning_asr_uplift_vs_clean": "Learning ASR uplift vs clean",
}


def _display_fieldname(fieldname: str) -> str:
    zh = FIELDNAME_ZH_MAP.get(fieldname)
    return f"{fieldname}（{zh}）" if zh else fieldname


def _json_ready(value):
    if is_dataclass(value):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([_display_fieldname(name) for name in fieldnames])
        for row in rows:
            writer.writerow([row.get(name, "") for name in fieldnames])
