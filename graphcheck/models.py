from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class RunArtifacts:
    run_id: str
    setting: str
    doc_count: int
    report_json_path: Path
    graph_path: Path


@dataclass(slots=True)
class ExperimentContext:
    asr_output_dir: Path
    dataset_experiment_dir: Path
    category_name: str
    query_id: str
    query_text: str
    query_text_fragment: str
    runs: list[RunArtifacts]
    new_parsed_dir: Path
    new_scoring_dir: Path
    new_traces_dir: Path
    new_summary_dir: Path


@dataclass(slots=True)
class TaskRecord:
    item_id: str
    index: int
    run_id: str
    setting: str
    doc_count: int
    graph_path: str
    text: str
    depth: int
    indent: int
    parent_item_id: str | None = None
    planning_web_urls: list[str] = field(default_factory=list)
    planning_local_urls: list[str] = field(default_factory=list)
    research_web_urls: list[str] = field(default_factory=list)
    research_local_urls: list[str] = field(default_factory=list)
    planning_total_refs: int = 0
    planning_web_refs: int = 0
    planning_local_refs: int = 0
    research_total_refs: int = 0
    research_web_refs: int = 0
    research_local_refs: int = 0
    total_refs: int = 0
    planning_poison_refs: int = 0
    research_poison_refs: int = 0
    poison_refs: int = 0
    poison_ratio: float = 0.0
    planning_poison_ratio: float = 0.0
    research_poison_ratio: float = 0.0
    asr_poisoned: bool | None = None
    asr_best_status: str | None = None
    asr_matched_types: list[str] = field(default_factory=list)
    asr_matched_claim_ids: list[str] = field(default_factory=list)
    asr_justifications: list[str] = field(default_factory=list)


@dataclass(slots=True)
class LearningRecord:
    item_id: str
    index: int
    run_id: str
    setting: str
    doc_count: int
    graph_path: str
    raw_text: str
    text: str
    source_url: str | None
    source_subtask_text: str | None
    source_subtask_item_id: str | None = None
    source_subtask_asr_poisoned: bool | None = None
    link_is_poison: bool = False
    asr_poisoned: bool | None = None
    asr_best_status: str | None = None
    asr_matched_types: list[str] = field(default_factory=list)
    asr_matched_claim_ids: list[str] = field(default_factory=list)
    asr_justifications: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ItemTypeScore:
    claim_type: str
    status: str
    matched_injected_claim_ids: list[str]
    influence_justification: str | None


@dataclass(slots=True)
class ItemAggregateScore:
    poisoned: bool
    best_status: str
    matched_types: list[str]
    matched_claim_ids: list[str]
    justifications: list[str]
    per_type: dict[str, ItemTypeScore]


@dataclass(slots=True)
class ParsedGraph:
    run_id: str
    setting: str
    doc_count: int
    graph_path: Path
    tasks: list[TaskRecord]
    learnings: list[LearningRecord]
