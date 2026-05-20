"""FORGE experiment helpers.

This package exposes the reusable pieces of the repository without bundling
private adversarial corpora or fully automated document-generation assets.
"""

from .chain import ChainEdge, ChainNode, build_linear_chain
from .construction import ForgePlan, ForgePlanStep, build_manual_review_plan
from .prompts import (
    CHAIN_DECOMPOSITION,
    CONCLUDING_DOCUMENT_FABRICATION,
    HUMAN_REVIEW_CHECKLIST,
    INTER_DOCUMENT_CHAINING_REVIEW,
    INTRA_DOCUMENT_FABRICATION_REVIEW,
    INTERMEDIATE_DOCUMENT_FABRICATION,
)

__all__ = [
    # Chain metadata
    "ChainEdge",
    "ChainNode",
    "ForgePlan",
    "ForgePlanStep",
    "build_linear_chain",
    "build_manual_review_plan",
    # Appendix B generation prompts (Steps 1, 2a, 2b)
    "CHAIN_DECOMPOSITION",
    "INTERMEDIATE_DOCUMENT_FABRICATION",
    "CONCLUDING_DOCUMENT_FABRICATION",
    "HUMAN_REVIEW_CHECKLIST",
    # Review/audit prompts
    "INTRA_DOCUMENT_FABRICATION_REVIEW",
    "INTER_DOCUMENT_CHAINING_REVIEW",
]
