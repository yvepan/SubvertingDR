from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ChainNode:
    """A narrative claim node used to describe inter-document chaining."""

    node_id: str
    claim: str


@dataclass(frozen=True)
class ChainEdge:
    """A directed support relation between two claim nodes."""

    source_id: str
    target_id: str
    relation: str = "supports"


def build_linear_chain(claims: Iterable[str], prefix: str = "c") -> tuple[list[ChainNode], list[ChainEdge]]:
    """Create a simple linear claim chain for reproducible experiment metadata."""

    nodes = [
        ChainNode(node_id=f"{prefix}{index}", claim=claim)
        for index, claim in enumerate(claims, start=1)
        if claim.strip()
    ]
    edges = [
        ChainEdge(source_id=nodes[index].node_id, target_id=nodes[index + 1].node_id)
        for index in range(len(nodes) - 1)
    ]
    return nodes, edges
