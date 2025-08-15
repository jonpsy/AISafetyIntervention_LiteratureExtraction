from __future__ import annotations

from .merge_types import (
    NodeComparisonInput,
    NodeViewForComparison,
    NodeAggregate,
)
from .merge_indexer import MergeIndex


def _node_view(agg: NodeAggregate) -> NodeViewForComparison:
    contexts: list[str] = []
    for e in agg.linked_edges:
        if e.rationale:
            contexts.append(e.get_context_for_node(agg.node_key))

    return NodeViewForComparison(
        text=agg.text,
        aliases=agg.aliases,
        context=contexts + agg.notes,
        source_metadata=agg.sources,
        linked_edges=agg.linked_edges,
    )


def build_node_comparison_input(
    index: MergeIndex, key_a: str, key_b: str
) -> NodeComparisonInput:
    a = index.nodes[key_a]
    b = index.nodes[key_b]
    return NodeComparisonInput(node_a=_node_view(a), node_b=_node_view(b))
