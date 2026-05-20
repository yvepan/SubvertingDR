"""Root Query Anchoring (RQA) defense helper (paper §7.3).

RQA is a planning-layer defense that constrains recursive subtask drift by
co-conditioning every follow-up query on the original root query q_0.  The
paper shows it reduces PRISM from 38.5% to 18.3% on the 10-query defense
subset while raising utility (RACE) from 0.500 to 0.617.

Implementation
--------------
The defense operates at two recursive injection points inside
``gpt_researcher/skills/deep_research.py``:

  1. ``DeepResearch.generate_search_queries()`` (called once per subtask) —
     before generating SERP sub-queries, ``_anchor_query()`` prepends q_0 to
     the subtask string so the planner's retrieval stays semantically tied to
     the original objective.

  2. ``DeepResearch.generate_research_plan()`` (called to produce follow-up
     questions from retrieved evidence) — the same anchoring is applied so
     that poisoned retrieved documents cannot steer the follow-up question list
     away from q_0.

At both points the call is:

    planning_query = self._anchor_query(root_query, query)

where ``_anchor_query`` delegates here and is a no-op when the defense is
disabled (``enable_deep_research_defense=False`` in config, or ``--disable-
defense`` on the CLI).

This module exposes ``build_root_query_anchored_query`` as the sole primitive
so the anchoring logic is testable independently of the async research loop.
"""

from __future__ import annotations


def build_root_query_anchored_query(root_query: str | None, sub_query: str | None) -> str:
    """Return a query that co-conditions sub_query on root_query.

    Concatenates root_query and sub_query with a single space so the
    resulting string carries both the original research objective and the
    current subtask focus.  This prevents a poisoned subtask from drifting
    the retrieval entirely away from the root query.

    Edge cases:
    - If root_query is empty or None, returns sub_query unchanged.
    - If sub_query is empty, None, or identical to root_query, returns root_query.
    """
    root = (root_query or "").strip()
    sub = (sub_query or "").strip()
    if not root:
        return sub
    if not sub or sub.lower() == root.lower():
        return root
    return f"{root} {sub}"
