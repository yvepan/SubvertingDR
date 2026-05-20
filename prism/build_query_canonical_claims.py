"""Step 1b: incrementally merge document-level atomic claims into a per-query canonical set."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from prism.llm_utils import build_llm_config, generate_json
from prism.pipeline_types import (
    CLAIM_TYPES,
    CanonicalAtomicClaim,
    QueryCanonicalClaims,
    chunk_lines,
    ensure_dir,
    load_doc_atomic_claims,
    load_queries,
    read_json,
    write_json,
)


MERGE_SYSTEM_PROMPT = """You incrementally merge document-level atomic claims into an order-aware canonical claim set.

Rules:
- Process claims from the current document against the existing canonical set.
- Only merge claims that have the same `claim_type`.
- Merge only when two claims express the same proposition without losing substantive detail.
- Do not merge if the new claim adds a new number, threshold, actor, mechanism, temporal marker, consequence, deployment detail, or comparative benchmark.
- When a claim is novel, create a new canonical claim using concise factual wording.
- Keep every canonical claim atomic and testable.
- Return valid JSON only.
"""


def _build_merge_prompt(
    query_text: str,
    existing_claims_text: str,
    current_doc_text: str,
) -> str:
    return f"""Query:
{query_text}

Existing canonical claims:
{existing_claims_text}

Current document claims:
{current_doc_text}

Return JSON with this shape:
{{
  "operations": [
    {{
      "claim_id": "doc_claim_id",
      "action": "merge" | "add",
      "existing_canonical_id": "temp_canonical_id or empty string",
      "canonical_text": "required when action=add, otherwise empty string",
      "reason": "short explanation"
    }}
  ]
}}
"""


def _validate_merge_payload(
    payload: dict[str, Any],
    allowed_claim_ids: set[str],
    allowed_existing_ids: set[str],
) -> None:
    operations = payload.get("operations")
    if not isinstance(operations, list):
        raise ValueError("operations must be a list.")
    if len(operations) != len(allowed_claim_ids):
        raise ValueError("There must be exactly one operation per current document claim.")

    seen_claim_ids: set[str] = set()
    for item in operations:
        if not isinstance(item, dict):
            raise ValueError("Each operation must be an object.")
        claim_id = str(item.get("claim_id", "")).strip()
        action = str(item.get("action", "")).strip()
        existing_id = str(item.get("existing_canonical_id", "")).strip()
        canonical_text = str(item.get("canonical_text", "")).strip()
        if claim_id not in allowed_claim_ids:
            raise ValueError("Unknown claim_id in operations.")
        if claim_id in seen_claim_ids:
            raise ValueError("Duplicate claim_id in operations.")
        if action not in {"merge", "add"}:
            raise ValueError("action must be 'merge' or 'add'.")
        if action == "merge" and existing_id not in allowed_existing_ids:
            raise ValueError("existing_canonical_id must point to an existing canonical claim.")
        if action == "add" and not canonical_text:
            raise ValueError("canonical_text is required when action=add.")
        seen_claim_ids.add(claim_id)


def _query_claims_from_payload(payload: dict[str, Any]) -> QueryCanonicalClaims:
    return QueryCanonicalClaims(
        query_id=payload["query_id"],
        query_text=payload["query_text"],
        canonical_atomic_claims=[CanonicalAtomicClaim(**item) for item in payload["canonical_atomic_claims"]],
    )


def build_query_canonical_claims(
    *,
    data_dir: Path,
    output_dir: Path,
    trace_dir: Path,
    model: str,
    base_url: str,
    api_key: str | None,
    api_key_env: str | None,
    query_ids: set[str] | None = None,
) -> list[QueryCanonicalClaims]:
    queries = {item.query_id: item for item in load_queries(data_dir / "queries.json")}
    doc_claims = load_doc_atomic_claims(output_dir / "atomic_units" / "doc_level")
    claims_by_query: dict[str, list] = defaultdict(list)
    for item in doc_claims:
        claims_by_query[item.query_id].append(item)

    client_config = build_llm_config(
        model=model,
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
    )

    query_output_dir = ensure_dir(output_dir / "atomic_units" / "query_level")
    ensure_dir(trace_dir)
    results: list[QueryCanonicalClaims] = []

    for query_id, query in queries.items():
        if query_ids and query_id not in query_ids:
            continue
        output_path = query_output_dir / f"{query_id}.json"
        if output_path.exists():
            results.append(_query_claims_from_payload(read_json(output_path)))
            continue

        grouped_claims = sorted(claims_by_query.get(query_id, []), key=lambda item: item.order)
        if not grouped_claims:
            continue

        canonical_records: list[dict[str, Any]] = []
        next_canonical_index = 1

        for doc_claim in grouped_claims:
            if not canonical_records:
                for claim in doc_claim.atomic_claims:
                    canonical_records.append(
                        {
                            "temp_id": f"temp_canonical_{next_canonical_index:02d}",
                            "text": claim.text,
                            "claim_type": claim.claim_type,
                            "source_doc_ids": [doc_claim.doc_id],
                            "merged_claim_ids": [claim.claim_id],
                            "first_source_order": doc_claim.order,
                        }
                    )
                    next_canonical_index += 1
                continue

            for claim_type in CLAIM_TYPES:
                current_claims = [claim for claim in doc_claim.atomic_claims if claim.claim_type == claim_type]
                if not current_claims:
                    continue

                existing_records = [item for item in canonical_records if item["claim_type"] == claim_type]
                if not existing_records:
                    for claim in current_claims:
                        canonical_records.append(
                            {
                                "temp_id": f"temp_canonical_{next_canonical_index:02d}",
                                "text": claim.text,
                                "claim_type": claim.claim_type,
                                "source_doc_ids": [doc_claim.doc_id],
                                "merged_claim_ids": [claim.claim_id],
                                "first_source_order": doc_claim.order,
                            }
                        )
                        next_canonical_index += 1
                    continue

                existing_claims_text = chunk_lines(
                    [
                        (
                            f"- {item['temp_id']}: {item['text']} | "
                            f"claim_type={item['claim_type']} | "
                            f"source_doc_ids={','.join(item['source_doc_ids'])} | "
                            f"first_source_order={item['first_source_order']}"
                        )
                        for item in existing_records
                    ]
                )
                current_doc_text = chunk_lines(
                    [
                        f"- {claim.claim_id}: {claim.text} | claim_type: {claim.claim_type} | rationale: {claim.rationale}"
                        for claim in current_claims
                    ]
                )
                allowed_claim_ids = {claim.claim_id for claim in current_claims}
                allowed_existing_ids = {item["temp_id"] for item in existing_records}
                trace_path = trace_dir / f"{query_id}_order_{doc_claim.order:02d}_{claim_type}.json"
                payload = generate_json(
                    config=client_config,
                    system_prompt=MERGE_SYSTEM_PROMPT,
                    user_prompt=_build_merge_prompt(
                        query.query_text,
                        existing_claims_text,
                        current_doc_text,
                    ),
                    validator=lambda item: _validate_merge_payload(
                        item,
                        allowed_claim_ids=allowed_claim_ids,
                        allowed_existing_ids=allowed_existing_ids,
                    ),
                    trace_path=trace_path,
                )
                claim_lookup = {claim.claim_id: claim for claim in current_claims}
                record_lookup = {item["temp_id"]: item for item in existing_records}

                for operation in payload["operations"]:
                    claim_id = operation["claim_id"]
                    action = operation["action"]
                    if action == "merge":
                        target = record_lookup[operation["existing_canonical_id"]]
                        if doc_claim.doc_id not in target["source_doc_ids"]:
                            target["source_doc_ids"].append(doc_claim.doc_id)
                        if claim_id not in target["merged_claim_ids"]:
                            target["merged_claim_ids"].append(claim_id)
                        continue

                    claim = claim_lookup[claim_id]
                    canonical_records.append(
                        {
                            "temp_id": f"temp_canonical_{next_canonical_index:02d}",
                            "text": str(operation["canonical_text"]).strip(),
                            "claim_type": claim.claim_type,
                            "source_doc_ids": [doc_claim.doc_id],
                            "merged_claim_ids": [claim.claim_id],
                            "first_source_order": doc_claim.order,
                        }
                    )
                    next_canonical_index += 1

        canonical_claims = [
            CanonicalAtomicClaim(
                canonical_id=f"{query_id}_canonical_{index:02d}",
                text=item["text"],
                claim_type=item["claim_type"],
                source_doc_ids=item["source_doc_ids"],
                merged_claim_ids=item["merged_claim_ids"],
                first_source_order=item["first_source_order"],
            )
            for index, item in enumerate(canonical_records, start=1)
        ]
        result = QueryCanonicalClaims(
            query_id=query_id,
            query_text=query.query_text,
            canonical_atomic_claims=canonical_claims,
        )
        write_json(output_path, result)
        results.append(result)

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build query-level canonical claims.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env", default="PRISM_API_KEY")
    parser.add_argument("--query-id", action="append", dest="query_ids")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    build_query_canonical_claims(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        trace_dir=args.trace_dir,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        query_ids=set(args.query_ids) if args.query_ids else None,
    )


if __name__ == "__main__":
    main()
