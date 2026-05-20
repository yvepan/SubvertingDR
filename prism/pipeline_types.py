"""Data types and I/O helpers for the PRISM atomic ASR pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable
import json


MATCH_STATUSES = {
    "matched_explicit",
    "matched_implicit",
    "mentioned_only",
    "contradicted",
    "absent",
}

CLAIM_TYPES = (
    "factual",
    "causal",
    "evaluative",
    "prescriptive",
    "framing",
)


@dataclass(slots=True)
class QueryRecord:
    query_id: str
    query_text: str


@dataclass(slots=True)
class ReferenceDocument:
    query_id: str
    doc_id: str
    order: int
    source_type: str
    title: str
    content: str
    source_path: str | None = None


@dataclass(slots=True)
class ReportRun:
    query_id: str
    run_id: str
    setting: str
    doc_count: int
    final_report: str
    source_path: str | None = None


@dataclass(slots=True)
class AtomicClaim:
    claim_id: str
    text: str
    rationale: str
    claim_type: str = "factual"


@dataclass(slots=True)
class DocAtomicClaims:
    query_id: str
    doc_id: str
    order: int
    atomic_claims: list[AtomicClaim]


@dataclass(slots=True)
class ReportAtomicClaims:
    query_id: str
    run_id: str
    setting: str
    doc_count: int
    atomic_claims: list[AtomicClaim]


@dataclass(slots=True)
class CanonicalAtomicClaim:
    canonical_id: str
    text: str
    claim_type: str
    source_doc_ids: list[str]
    merged_claim_ids: list[str] = field(default_factory=list)
    first_source_order: int | None = None


@dataclass(slots=True)
class QueryCanonicalClaims:
    query_id: str
    query_text: str
    canonical_atomic_claims: list[CanonicalAtomicClaim]


@dataclass(slots=True)
class ClaimResult:
    report_claim_id: str
    text: str
    claim_type: str
    status: str
    matched_injected_claim_ids: list[str]
    evidence_strength: str | None = None
    influence_justification: str | None = None


@dataclass(slots=True)
class AtomicAsrResult:
    query_id: str
    run_id: str
    setting: str
    doc_count: int
    total_atomic_claims: int
    matched_atomic_claims: int
    atomic_asr: float
    claim_results: list[ClaimResult]
    injected_claim_count: int | None = None
    report_claim_count: int | None = None
    matched_explicit_count: int | None = None
    matched_implicit_count: int | None = None
    factual_total_claims: int | None = None
    factual_matched_claims: int | None = None
    factual_asr: float | None = None
    causal_total_claims: int | None = None
    causal_matched_claims: int | None = None
    causal_asr: float | None = None
    evaluative_total_claims: int | None = None
    evaluative_matched_claims: int | None = None
    evaluative_asr: float | None = None
    prescriptive_total_claims: int | None = None
    prescriptive_matched_claims: int | None = None
    prescriptive_asr: float | None = None
    framing_total_claims: int | None = None
    framing_matched_claims: int | None = None
    framing_asr: float | None = None
    impact_score: float | None = None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return {key: json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    return value


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_list(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def load_queries(path: Path) -> list[QueryRecord]:
    return [QueryRecord(**item) for item in read_json_list(path)]


def load_reference_documents(path: Path) -> list[ReferenceDocument]:
    documents: list[ReferenceDocument] = []
    for json_path in sorted(path.glob("*.json")):
        documents.append(ReferenceDocument(**read_json(json_path)))
    return documents


def load_report_runs(path: Path) -> list[ReportRun]:
    runs: list[ReportRun] = []
    for json_path in sorted(path.glob("*.json")):
        runs.append(ReportRun(**read_json(json_path)))
    return runs


def load_doc_atomic_claims(path: Path) -> list[DocAtomicClaims]:
    results: list[DocAtomicClaims] = []
    for json_path in sorted(path.glob("*.json")):
        payload = read_json(json_path)
        claims = [AtomicClaim(**claim) for claim in payload["atomic_claims"]]
        results.append(
            DocAtomicClaims(
                query_id=payload["query_id"],
                doc_id=payload["doc_id"],
                order=payload["order"],
                atomic_claims=claims,
            )
        )
    return results


def load_report_atomic_claims(path: Path) -> ReportAtomicClaims:
    payload = read_json(path)
    claims = [AtomicClaim(**claim) for claim in payload["atomic_claims"]]
    return ReportAtomicClaims(
        query_id=payload["query_id"],
        run_id=payload["run_id"],
        setting=payload["setting"],
        doc_count=payload["doc_count"],
        atomic_claims=claims,
    )


def load_query_canonical_claims(path: Path) -> QueryCanonicalClaims:
    payload = read_json(path)
    claims = [
        CanonicalAtomicClaim(**claim)
        for claim in payload["canonical_atomic_claims"]
    ]
    return QueryCanonicalClaims(
        query_id=payload["query_id"],
        query_text=payload["query_text"],
        canonical_atomic_claims=claims,
    )


def load_scoring_results(path: Path) -> list[AtomicAsrResult]:
    results: list[AtomicAsrResult] = []
    for json_path in sorted(path.glob("*.json")):
        payload = read_json(json_path)
        claim_results = [ClaimResult(**item) for item in payload["claim_results"]]
        results.append(
            AtomicAsrResult(
                query_id=payload["query_id"],
                run_id=payload["run_id"],
                setting=payload["setting"],
                doc_count=payload["doc_count"],
                total_atomic_claims=payload["total_atomic_claims"],
                matched_atomic_claims=payload["matched_atomic_claims"],
                atomic_asr=payload["atomic_asr"],
                claim_results=claim_results,
                injected_claim_count=payload.get("injected_claim_count"),
                report_claim_count=payload.get("report_claim_count"),
                matched_explicit_count=payload.get("matched_explicit_count"),
                matched_implicit_count=payload.get("matched_implicit_count"),
                factual_total_claims=payload.get("factual_total_claims"),
                factual_matched_claims=payload.get("factual_matched_claims"),
                factual_asr=payload.get("factual_asr"),
                causal_total_claims=payload.get("causal_total_claims"),
                causal_matched_claims=payload.get("causal_matched_claims"),
                causal_asr=payload.get("causal_asr"),
                evaluative_total_claims=payload.get("evaluative_total_claims"),
                evaluative_matched_claims=payload.get("evaluative_matched_claims"),
                evaluative_asr=payload.get("evaluative_asr"),
                prescriptive_total_claims=payload.get("prescriptive_total_claims"),
                prescriptive_matched_claims=payload.get("prescriptive_matched_claims"),
                prescriptive_asr=payload.get("prescriptive_asr"),
                framing_total_claims=payload.get("framing_total_claims"),
                framing_matched_claims=payload.get("framing_matched_claims"),
                framing_asr=payload.get("framing_asr"),
                impact_score=payload.get("impact_score"),
            )
        )
    return results


def chunk_lines(items: Iterable[str]) -> str:
    return "\n".join(items)
