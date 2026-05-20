from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from .metrics import PrismScore, aggregate_prism_score, score_claim_records
from .taxonomy import ClaimType


def load_atomic_claim_csv(path: Path | str) -> list[dict[str, object]]:
    """Load a normalized claim CSV for PRISM metric aggregation.

    Expected columns are `claim_type` and `infected`. Existing ASR pipeline
    outputs can be converted to this shape before calling the metric helpers.
    """

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        value = str(row.get("infected", "")).strip().lower()
        row["infected"] = value in {"1", "true", "yes", "y"}
    return rows


def score_claim_csv(path: Path | str, weights: dict[ClaimType, float] | None = None) -> float:
    return aggregate_prism_score(load_atomic_claim_csv(path), weights=weights)


def score_claim_csv_detailed(
    path: Path | str,
    weights: dict[ClaimType, float] | None = None,
) -> PrismScore:
    return score_claim_records(load_atomic_claim_csv(path), weights=weights)


def group_records(
    records: Iterable[dict[str, object]],
    key: str,
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        value = str(record.get(key, "")).strip() or "<missing>"
        grouped[value].append(record)
    return dict(grouped)
