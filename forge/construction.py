from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .chain import ChainEdge, ChainNode, build_linear_chain


@dataclass(frozen=True)
class ForgePlanStep:
    """A reviewed construction step, not an automatically generated document."""

    step_id: str
    instruction: str
    requires_review: bool = True


@dataclass(frozen=True)
class ForgePlan:
    """Structured metadata for a FORGE-style experiment plan."""

    query: str
    target_narrative: str
    nodes: list[ChainNode] = field(default_factory=list)
    edges: list[ChainEdge] = field(default_factory=list)
    steps: list[ForgePlanStep] = field(default_factory=list)


def build_manual_review_plan(query: str, target_narrative: str, claims: Iterable[str]) -> ForgePlan:
    """Build non-generative experiment metadata with explicit review gates.

    The public package intentionally stops at plan metadata. Document drafting
    and release decisions stay outside this helper so the repository can expose
    reproducible structure without shipping a turnkey attack generator.
    """

    nodes, edges = build_linear_chain(claims)
    steps = [
        ForgePlanStep(
            "decompose",
            "Apply CHAIN_DECOMPOSITION prompt to decompose the target narrative into"
            " claim nodes; review the resulting chain before proceeding.",
        ),
        ForgePlanStep(
            "fabricate",
            "Apply INTERMEDIATE_DOCUMENT_FABRICATION (p_1 … p_{j-1}) and"
            " CONCLUDING_DOCUMENT_FABRICATION (p_j) prompts; run"
            " INTRA_DOCUMENT_FABRICATION_REVIEW on each draft and sign off against"
            " HUMAN_REVIEW_CHECKLIST before marking this step complete.",
        ),
        ForgePlanStep(
            "chain",
            "Apply INTER_DOCUMENT_CHAINING_REVIEW to the full document set; confirm"
            " that every inter-document link is coherent and no circular support exists.",
        ),
    ]
    return ForgePlan(
        query=query,
        target_narrative=target_narrative,
        nodes=nodes,
        edges=edges,
        steps=steps,
    )
