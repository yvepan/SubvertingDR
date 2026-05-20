"""Step 1 of the PRISM pipeline: extract atomic poisoning points from each adversarial document."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from prism.evaluator_prompts import DOCUMENT_CLAIM_EXTRACTION_SYSTEM, DOCUMENT_CLAIM_EXTRACTION_USER
from prism.llm_utils import build_llm_config, generate_json
from prism.pipeline_types import (
    CLAIM_TYPES,
    AtomicClaim,
    DocAtomicClaims,
    QueryRecord,
    ensure_dir,
    load_queries,
    load_reference_documents,
    read_json,
    write_json,
)


DOC_ATOMIC_CLAIM_TARGET = 10


def _build_prompt(query: QueryRecord, document_title: str, document_content: str) -> str:
    return DOCUMENT_CLAIM_EXTRACTION_USER.format(
        query=query.query_text,
        document_title=document_title,
        document_content=document_content,
    )


def _validate_atomic_claim_payload(payload: dict[str, Any]) -> None:
    claims = payload.get("atomic_claims")
    if not isinstance(claims, list):
        raise ValueError("atomic_claims must be a list.")
    if len(claims) != DOC_ATOMIC_CLAIM_TARGET:
        raise ValueError(f"atomic_claims must contain exactly {DOC_ATOMIC_CLAIM_TARGET} claims.")

    seen: set[str] = set()
    present_types: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("Each atomic claim must be an object.")
        text = str(claim.get("text", "")).strip()
        rationale = str(claim.get("rationale", "")).strip()
        claim_type = str(claim.get("claim_type", "")).strip().lower()
        if not text or not rationale or claim_type not in CLAIM_TYPES:
            raise ValueError("Each atomic claim must contain non-empty text, rationale, and a valid claim_type.")
        norm = " ".join(text.lower().split())
        if norm in seen:
            raise ValueError("Duplicate atomic claims are not allowed.")
        seen.add(norm)
        present_types.add(claim_type)

    missing_types = [claim_type for claim_type in CLAIM_TYPES if claim_type not in present_types]
    if missing_types:
        raise ValueError(f"atomic_claims must include all claim types at least once. Missing: {', '.join(missing_types)}")


def _load_valid_cached_doc_claims(output_path: Path) -> DocAtomicClaims | None:
    if not output_path.exists():
        return None

    payload = read_json(output_path)
    try:
        _validate_atomic_claim_payload(payload)
    except ValueError:
        return None
    return _doc_claims_from_payload(payload)


def _doc_claims_from_payload(payload: dict[str, Any]) -> DocAtomicClaims:
    return DocAtomicClaims(
        query_id=payload["query_id"],
        doc_id=payload["doc_id"],
        order=payload["order"],
        atomic_claims=[AtomicClaim(**item) for item in payload["atomic_claims"]],
    )


def extract_doc_atomic_claims(
    *,
    data_dir: Path,
    output_dir: Path,
    trace_dir: Path,
    model: str,
    base_url: str,
    api_key: str | None,
    api_key_env: str | None,
    query_ids: set[str] | None = None,
) -> list[DocAtomicClaims]:
    queries = {item.query_id: item for item in load_queries(data_dir / "queries.json")}
    client_config = build_llm_config(
        model=model,
        base_url=base_url,
        api_key=api_key,
        api_key_env=api_key_env,
    )

    results: list[DocAtomicClaims] = []
    doc_output_dir = ensure_dir(output_dir / "atomic_units" / "doc_level")
    ensure_dir(trace_dir)

    for query_id, query in queries.items():
        if query_ids and query_id not in query_ids:
            continue

        reference_dir = data_dir / "references" / query_id
        for document in load_reference_documents(reference_dir):
            output_path = doc_output_dir / f"{document.doc_id}.json"
            cached_output = _load_valid_cached_doc_claims(output_path)
            if cached_output is not None:
                results.append(cached_output)
                continue

            trace_path = trace_dir / f"{document.doc_id}.json"
            response = generate_json(
                config=client_config,
                system_prompt=DOCUMENT_CLAIM_EXTRACTION_SYSTEM,
                user_prompt=_build_prompt(query, document.title, document.content),
                validator=_validate_atomic_claim_payload,
                trace_path=trace_path,
            )
            atomic_claims = [
                AtomicClaim(
                    claim_id=f"{document.doc_id}_claim_{index:02d}",
                    text=str(item["text"]).strip(),
                    rationale=str(item["rationale"]).strip(),
                    claim_type=str(item["claim_type"]).strip().lower(),
                )
                for index, item in enumerate(response["atomic_claims"], start=1)
            ]
            output = DocAtomicClaims(
                query_id=document.query_id,
                doc_id=document.doc_id,
                order=document.order,
                atomic_claims=atomic_claims,
            )
            write_json(output_path, output)
            results.append(output)

    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract document-level atomic claims.")
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
    extract_doc_atomic_claims(
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
