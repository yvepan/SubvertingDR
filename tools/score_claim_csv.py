from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from prism.evaluator import group_records, load_atomic_claim_csv
from prism.metrics import PrismScore, score_claim_records
from prism.taxonomy import CLAIM_WEIGHTS, WEIGHT_PRESETS


def _score_label(weighting: str) -> str:
    return "prism_paper" if weighting == "prism" else "asr_equal"


def _score_to_dict(group: str, score: PrismScore, score_label: str) -> dict[str, object]:
    return {
        "group": group,
        score_label: score.score,
        "total_claims": score.total_claims,
        "infected_claims": score.infected_claims,
        "total_weight": score.total_weight,
        "infected_weight": score.infected_weight,
    }


def _print_text(group: str, score: PrismScore, score_label: str) -> None:
    prefix = "" if group == "overall" else f"{group}."
    for claim_type in CLAIM_WEIGHTS:
        dimension = score.dimensions.get(claim_type)
        if not dimension:
            continue
        print(
            f"{prefix}{claim_type.value}_asr={dimension.asr:.6f} "
            f"({dimension.infected_claims}/{dimension.total_claims})"
        )
    print(f"{prefix}{score_label}={score.score:.6f}")
    print(f"{prefix}infected_weight={score.infected_weight:.6f}")
    print(f"{prefix}total_weight={score.total_weight:.6f}")


def _print_csv(rows: list[dict[str, object]], score_label: str) -> None:
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=["group", score_label, "total_claims", "infected_claims", "total_weight", "infected_weight"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute PRISM-style ASR from a normalized claim CSV."
    )
    parser.add_argument("claim_csv", type=Path)
    parser.add_argument(
        "--weighting",
        choices=sorted(WEIGHT_PRESETS),
        default="prism",
        help="Use paper PRISM weights or equal-weight diagnostic ASR.",
    )
    parser.add_argument(
        "--group-by",
        choices=["none", "query_id", "report_id"],
        default="none",
        help="Optionally compute scores separately by query or report.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "csv", "json"],
        default="text",
        help="Output format for score summaries.",
    )
    args = parser.parse_args()

    records = load_atomic_claim_csv(args.claim_csv)
    weights = WEIGHT_PRESETS[args.weighting]
    score_label = _score_label(args.weighting)

    if args.group_by == "none":
        scored_groups = [("overall", score_claim_records(records, weights=weights))]
    else:
        scored_groups = [
            (group, score_claim_records(subrecords, weights=weights))
            for group, subrecords in sorted(group_records(records, args.group_by).items())
        ]

    rows = [_score_to_dict(group, score, score_label) for group, score in scored_groups]
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif args.format == "csv":
        _print_csv(rows, score_label)
    else:
        for index, (group, score) in enumerate(scored_groups):
            if index:
                print()
            _print_text(group, score, score_label)


if __name__ == "__main__":
    main()
