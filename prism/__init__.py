"""PRISM — Poisoning Report Impact Severity Metric.

Public API
----------
Taxonomy and scoring helpers (used by tools/score_claim_csv.py):
  CLAIM_WEIGHTS, ClaimType, aggregate_prism_score, dimension_asr

Evaluator prompt templates (paper §5.2, Steps 1–3):
  DOCUMENT_CLAIM_EXTRACTION_SYSTEM, DOCUMENT_CLAIM_EXTRACTION_USER
  REPORT_CLAIM_EXTRACTION_SYSTEM,   REPORT_CLAIM_EXTRACTION_USER
  CLAIM_MATCHING_SYSTEM_TEMPLATE,   CLAIM_MATCHING_USER
  build_matching_system_prompt

Pipeline data types:
  QueryRecord, ReferenceDocument, ReportRun
  AtomicClaim, DocAtomicClaims, ReportAtomicClaims
  CanonicalAtomicClaim, QueryCanonicalClaims
  ClaimResult, AtomicAsrResult
  CLAIM_TYPES, MATCH_STATUSES

LLM client:
  LLMConfig, LLMJsonError, build_llm_config, generate_json

Pipeline stages (importable individually or via run_pipeline):
  extract_doc_atomic_claims   — Step 1
  build_query_canonical_claims — Step 1b
  score_report_atomic_asr     — Steps 2–3
  run_pipeline                — end-to-end orchestration
"""

from .evaluator_prompts import (
    CLAIM_MATCHING_SYSTEM_TEMPLATE,
    CLAIM_MATCHING_USER,
    DOCUMENT_CLAIM_EXTRACTION_SYSTEM,
    DOCUMENT_CLAIM_EXTRACTION_USER,
    REPORT_CLAIM_EXTRACTION_SYSTEM,
    REPORT_CLAIM_EXTRACTION_USER,
    build_matching_system_prompt,
)
from .metrics import aggregate_prism_score, dimension_asr
from .pipeline_types import (
    CLAIM_TYPES,
    MATCH_STATUSES,
    AtomicAsrResult,
    AtomicClaim,
    CanonicalAtomicClaim,
    ClaimResult,
    DocAtomicClaims,
    QueryCanonicalClaims,
    QueryRecord,
    ReferenceDocument,
    ReportAtomicClaims,
    ReportRun,
)
from .taxonomy import CLAIM_WEIGHTS, ClaimType

__all__ = [
    # Taxonomy and scoring
    "CLAIM_WEIGHTS",
    "ClaimType",
    "aggregate_prism_score",
    "dimension_asr",
    # Evaluator prompts (paper §5.2, Steps 1-3)
    "DOCUMENT_CLAIM_EXTRACTION_SYSTEM",
    "DOCUMENT_CLAIM_EXTRACTION_USER",
    "REPORT_CLAIM_EXTRACTION_SYSTEM",
    "REPORT_CLAIM_EXTRACTION_USER",
    "CLAIM_MATCHING_SYSTEM_TEMPLATE",
    "CLAIM_MATCHING_USER",
    "build_matching_system_prompt",
    # Pipeline data types
    "CLAIM_TYPES",
    "MATCH_STATUSES",
    "QueryRecord",
    "ReferenceDocument",
    "ReportRun",
    "AtomicClaim",
    "DocAtomicClaims",
    "ReportAtomicClaims",
    "CanonicalAtomicClaim",
    "QueryCanonicalClaims",
    "ClaimResult",
    "AtomicAsrResult",
]
