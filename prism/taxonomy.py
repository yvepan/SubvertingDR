from __future__ import annotations

from enum import StrEnum


class ClaimType(StrEnum):
    FACTUAL = "factual"
    CAUSAL = "causal"
    EVALUATIVE = "evaluative"
    PRESCRIPTIVE = "prescriptive"
    FRAMING = "framing"


PRISM_PAPER_WEIGHTS: dict[ClaimType, float] = {
    ClaimType.FACTUAL: 4.0,
    ClaimType.PRESCRIPTIVE: 5.0,
    ClaimType.EVALUATIVE: 6.0,
    ClaimType.CAUSAL: 7.0,
    ClaimType.FRAMING: 8.0,
}

EQUAL_CLAIM_WEIGHTS: dict[ClaimType, float] = {
    claim_type: 1.0 for claim_type in ClaimType
}

CLAIM_WEIGHTS = PRISM_PAPER_WEIGHTS

WEIGHT_PRESETS: dict[str, dict[ClaimType, float]] = {
    "prism": PRISM_PAPER_WEIGHTS,
    "equal": EQUAL_CLAIM_WEIGHTS,
}
