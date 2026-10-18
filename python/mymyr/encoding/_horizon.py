"""The input of FlatHorizonEncoder: the nodes and edges of a batch of horizons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mymyr import State
    from mymyr._typing import SupportsDLPack


@dataclass(frozen=True)
class HorizonView:
    """G horizons: each a root state and the nodes below it, as rows of one batch of states, with the edges between
    them. ``FlatHorizonEncoder.encode`` takes this object or any other with these attributes.

    - ``nodes``: the node rows, an array [M, W] of state words (64-bit words or 32-bit halves [M, 2W], of any DLPack
      framework) or a list of States. Graph g owns the rows ``node_offsets[g]`` to ``node_offsets[g + 1]``; the first
      of them is the graph's root, so node k of graph g is row ``node_offsets[g] + k``.
    - ``node_offsets``: int32 [G + 1], from 0, strictly ascending (every graph has a root): the last entry is at most M.
    - ``edge_offsets``: int32 [G + 1], from 0, ascending: the edges of graph g are the entries ``edge_offsets[g]`` to
      ``edge_offsets[g + 1]`` of the edge arrays.
    - ``edge_src``, ``edge_dst``: int32 [E], graph-local node indices: edge e leads from node ``edge_src[e]`` to node
      ``edge_dst[e]``.
    - ``edge_schema``: int32 [E], the action schema of each edge; ``edge_binding``: int32 [E, L], the objects of its
      ground action (the first arity-of-the-schema entries of a row are read; L at least the table's label width).
      ``rl.expand`` gives the schema and binding rows of a state's successors in exactly this form.
    - ``task_ids``: the instance of each graph, int32 [G]; None for a table of one instance.

    The edges are used as given: a node may have several incoming edges or none, and the encoder neither sorts nor
    merges them. An action relation holds one tuple per edge and a parent relation one tuple per edge, in the order of
    the edges of the graph. The edge arrays that the schema does not read (the encoder needs the edges of a schema with
    action or parent relations only, ``edge_schema`` and ``edge_binding`` of one with action relations only) may be
    None. The arrays are read where they are; they must stay unchanged during the call.
    """

    nodes: SupportsDLPack | Sequence[State]
    node_offsets: SupportsDLPack
    edge_offsets: SupportsDLPack
    edge_src: SupportsDLPack | None = None
    edge_dst: SupportsDLPack | None = None
    edge_schema: SupportsDLPack | None = None
    edge_binding: SupportsDLPack | None = None
    task_ids: SupportsDLPack | Sequence[int] | None = None

    def __post_init__(self) -> None:
        for name in ("nodes", "node_offsets", "edge_offsets"):
            if getattr(self, name) is None:
                raise TypeError(f"mymyr: encoding: HorizonView.{name} must be given")
