import asyncio
import logging
import os
import re
from typing import Any

from ..utils.bm25 import get_top_k_hybrid


def _to_virtual_url(base_name: str, ext: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9\-]+", "-", base_name).strip("-").lower()
    return f"http://research/{slug}{ext}"


async def get_local_document_url_items(researcher) -> list[dict[str, str]]:
    """Build virtual URL candidates for web poison documents."""
    logger = logging.getLogger("research")
    local_url_items: list[dict[str, str]] = []

    doc_path = getattr(researcher.cfg, "doc_path_web_poison", None)
    if not doc_path:
        return local_url_items

    logger.info(f"Web poison: using path {doc_path}")

    try:
        paths_to_process = [doc_path] if isinstance(doc_path, (str, bytes, os.PathLike)) else doc_path

        for path in paths_to_process:
            if os.path.isfile(path):
                filename = os.path.basename(path)
                base_name, ext = os.path.splitext(filename)
                
                body_text = ""
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        body_text = f.read(500)
                except Exception:
                    pass
                
                local_url_items.append({
                    "href": _to_virtual_url(base_name, ext),
                    "title": base_name,
                    "body": body_text,
                })
                continue

            if not os.path.isdir(path):
                continue

            for root, _, files in os.walk(path):
                for file in files:
                    base_name, ext = os.path.splitext(file)
                    if ext.lower() not in [".pdf", ".txt", ".doc", ".docx", ".pptx", ".csv", ".xls", ".xlsx", ".md", ".html"]:
                        continue

                    body_text = ""
                    try:
                        file_path = os.path.join(root, file)
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            body_text = f.read(500)
                    except Exception:
                        pass

                    local_url_items.append({
                        "href": _to_virtual_url(base_name, ext),
                        "title": base_name,
                        "body": body_text,
                    })

        logger.info(f"Web poison: generated {len(local_url_items)} virtual URLs")
    except Exception as e:
        logger.error(f"Error getting local document URLs: {e}")

    return local_url_items


async def filter_url_items_by_title_similarity(
    query: str,
    url_items: list[dict[str, Any]],
    researcher,
    dynamic_top_k: bool = True,
    summary_max_chars: int | None = None,
) -> list[dict[str, Any]]:
    """Apply the same hybrid top-k candidate ranking used by web research."""
    logger = logging.getLogger("research")

    if not url_items:
        return []

    if not dynamic_top_k:
        return url_items

    real_url_count = 0
    virtual_url_count = 0
    for item in url_items:
        href = item.get("href") or item.get("url", "")
        if href.startswith("http://research/"):
            virtual_url_count += 1
        else:
            real_url_count += 1

    top_k = real_url_count
    if top_k == 0:
        return url_items

    try:
        enriched = []
        for item in url_items:
            title = item.get("title", "")
            if not title:
                href = item.get("href") or item.get("url", "")
                if href.startswith("http://research/"):
                    title = href.replace("http://research/", "").replace("-", " ")
                else:
                    title = item.get("body", "")[:100]

            enriched.append({**item, "title": title})

        top_items = await get_top_k_hybrid(
            query,
            enriched,
            k=top_k,
            embeddings=researcher.memory.get_embeddings(),
            summary_max_chars=summary_max_chars,
        )

        selected_real = sum(
            1 for item in top_items if not (item.get("href") or item.get("url", "")).startswith("http://research/")
        )
        selected_virtual = sum(
            1 for item in top_items if (item.get("href") or item.get("url", "")).startswith("http://research/")
        )

        logger.info(f"Hybrid filtering: real URLs={real_url_count}, virtual URLs={virtual_url_count}, top-k={top_k}")
        logger.info(f"Filtering result: selected {len(top_items)} of {len(url_items)} URLs (real={selected_real}, poison={selected_virtual})")
        return top_items
    except Exception as e:
        logger.error(f"Hybrid filtering failed: {e}; returning all URLs")
        return url_items


async def get_planning_search_results(query: str, researcher, query_domains: list[str] | None = None) -> list[dict[str, Any]]:
    """Collect planning sources using the same candidate assembly and filtering as web research."""
    logger = logging.getLogger("research")
    if query_domains is None:
        query_domains = []

    all_url_items: list[dict[str, Any]] = []

    for retriever_class in researcher.retrievers:
        if "mcpretriever" in retriever_class.__name__.lower():
            continue

        try:
            retriever = retriever_class(query, query_domains=query_domains)
            search_results = await asyncio.to_thread(
                retriever.search,
                max_results=researcher.cfg.max_search_results_per_query,
            )

            for result in search_results:
                href = result.get("href") or result.get("url")
                if not href:
                    continue
                all_url_items.append({
                    "href": href,
                    "title": result.get("title", ""),
                    "body": result.get("body", "") or result.get("content", ""),
                })
        except Exception as e:
            logger.error(f"Error searching with {retriever_class.__name__}: {e}")

    if researcher.report_source in ["local", "hybrid"]:
        local_url_items = await get_local_document_url_items(researcher)
        all_url_items.extend(local_url_items)
        logger.info(f"Web poison: added {len(local_url_items)} virtual URLs to planning search results")

    enable_similarity_filter = getattr(researcher.cfg, "enable_url_similarity_filter", False)
    if enable_similarity_filter:
        logger.info(f"[Planning stage] node subtask query={query[:80]}")
        logger.info(f"Planning stage candidate ranking enabled (total URLs: {len(all_url_items)})")
        filtered_items = await filter_url_items_by_title_similarity(
            query,
            all_url_items,
            researcher,
            dynamic_top_k=True,
            summary_max_chars=200,
        )
    else:
        logger.info("Planning stage candidate ranking disabled; using all URLs")
        filtered_items = all_url_items

    seen: set[str] = set()
    unique_items: list[dict[str, Any]] = []
    for item in filtered_items:
        href = item.get("href") or item.get("url")
        if not href or href in seen:
            continue
        seen.add(href)
        unique_items.append(item)

    return unique_items
