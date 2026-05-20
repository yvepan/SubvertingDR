from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .taxonomy import CLAIM_WEIGHTS, ClaimType


def _as_claim_type(value: str | ClaimType) -> ClaimType:
    return value if isinstance(value, ClaimType) else ClaimType(str(value).lower())


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def dimension_asr(records: Iterable[Mapping[str, object]], claim_type: str | ClaimType) -> float:
    """Return infected / total for one claim dimension."""

    target = _as_claim_type(claim_type)
    total = 0
    infected = 0
    for record in records:
        raw_claim_type = str(record.get("claim_type", "")).strip().lower()
        if not raw_claim_type:
            continue
        if _as_claim_type(raw_claim_type) != target:
            continue
        total += 1
        infected += int(_as_bool(record.get("infected", False)))
    return infected / total if total else 0.0


@dataclass(frozen=True)
class ClaimDimensionScore:
    claim_type: ClaimType
    total_claims: int = 0
    infected_claims: int = 0
    total_weight: float = 0.0
    infected_weight: float = 0.0

    @property
    def asr(self) -> float:
        return self.infected_claims / self.total_claims if self.total_claims else 0.0


@dataclass(frozen=True)
class PrismScore:
    score: float
    total_claims: int
    infected_claims: int
    total_weight: float
    infected_weight: float
    dimensions: dict[ClaimType, ClaimDimensionScore] = field(default_factory=dict)


def score_claim_records(
    records: Iterable[Mapping[str, object]],
    weights: Mapping[ClaimType, float] | None = None,
) -> PrismScore:
    """Score normalized claim records using weighted claim mass."""

    selected_weights = dict(weights or CLAIM_WEIGHTS)
    dimensions = {
        claim_type: {
            "total_claims": 0,
            "infected_claims": 0,
            "total_weight": 0.0,
            "infected_weight": 0.0,
        }
        for claim_type in selected_weights
    }
    total_claims = 0
    infected_claims = 0
    total_weight = 0.0
    infected_weight = 0.0

    for record in records:
        raw_claim_type = str(record.get("claim_type", "")).strip().lower()
        if not raw_claim_type:
            continue
        claim_type = _as_claim_type(raw_claim_type)
        weight = float(selected_weights.get(claim_type, 0.0))
        if weight <= 0:
            continue

        infected = _as_bool(record.get("infected", False))
        dimensions.setdefault(
            claim_type,
            {
                "total_claims": 0,
                "infected_claims": 0,
                "total_weight": 0.0,
                "infected_weight": 0.0,
            },
        )
        dimensions[claim_type]["total_claims"] += 1
        dimensions[claim_type]["total_weight"] += weight
        total_claims += 1
        total_weight += weight

        if infected:
            dimensions[claim_type]["infected_claims"] += 1
            dimensions[claim_type]["infected_weight"] += weight
            infected_claims += 1
            infected_weight += weight

    dimension_scores = {
        claim_type: ClaimDimensionScore(
            claim_type=claim_type,
            total_claims=int(values["total_claims"]),
            infected_claims=int(values["infected_claims"]),
            total_weight=float(values["total_weight"]),
            infected_weight=float(values["infected_weight"]),
        )
        for claim_type, values in dimensions.items()
    }
    score = infected_weight / total_weight if total_weight else 0.0
    return PrismScore(
        score=score,
        total_claims=total_claims,
        infected_claims=infected_claims,
        total_weight=total_weight,
        infected_weight=infected_weight,
        dimensions=dimension_scores,
    )


def aggregate_prism_score(
    records: Iterable[Mapping[str, object]],
    weights: Mapping[ClaimType, float] | None = None,
) -> float:
    """Compute PRISM as infected weighted claim mass over total claim mass."""

    return score_claim_records(records, weights=weights).score
