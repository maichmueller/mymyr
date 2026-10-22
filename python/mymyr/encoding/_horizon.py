"""The input of FlatHorizonEncoder: the nodes and edges of a batch of horizons, and the builders of it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mymyr._core import _encoding as _core

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mymyr import State, Task, TaskHandle
    from mymyr._typing import SupportsDLPack
    from mymyr.datasets import StateSpace
    from mymyr.rl import TaskTable, ThreadPool


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
    - ``truncated``: uint8 [G], set where ``horizon`` stopped a graph at ``max_nodes``; None when not known. The encoders
      do not read it.
    - ``parent_offsets`` and ``parent_edges``: optional int32 [G + 1] and [P] canonical parent pairs, supplied together.
      The slice for graph g contains one graph-local edge index per distinct (source, destination), ordered by that
      pair. Builders prepare these arrays so that encoding parent relations needs no sort. They must describe the
      complete pair set of the unchanged edges; omit both when editing edges or constructing an arbitrary view.

    The edges may come in any order and be repeated, and a node may have several incoming edges or none: the encoder
    works from their canonical form, so the batch does not depend on how the view was made. An action relation holds one
    tuple per distinct (destination, objects) of its edges, ordered by destination and then objects; the parent relation
    one tuple per distinct (source, destination), ordered by it, a self-loop included; the sibling and cousin relations
    are read from the parent edges without the self-loops. Under ``root_policy='exclude'`` an edge into the root has no
    action tuple and an edge that touches the root no parent, sibling or cousin tuple. The edge arrays that the schema
    does not read (the encoder needs the edges of a schema with action, parent, sibling or cousin relations only,
    ``edge_schema`` and ``edge_binding`` of one with action relations only) may be None. The arrays are read where they
    are; they must stay unchanged during the call.
    """

    nodes: SupportsDLPack | Sequence[State]
    node_offsets: SupportsDLPack
    edge_offsets: SupportsDLPack
    edge_src: SupportsDLPack | None = None
    edge_dst: SupportsDLPack | None = None
    edge_schema: SupportsDLPack | None = None
    edge_binding: SupportsDLPack | None = None
    task_ids: SupportsDLPack | Sequence[int] | None = None
    truncated: SupportsDLPack | None = None
    parent_offsets: SupportsDLPack | None = None
    parent_edges: SupportsDLPack | None = None

    def __post_init__(self) -> None:
        for name in ("nodes", "node_offsets", "edge_offsets"):
            if getattr(self, name) is None:
                raise TypeError(f"mymyr: encoding: HorizonView.{name} must be given")


def horizon(
    table: TaskTable | Task | TaskHandle,
    roots: SupportsDLPack | Sequence[State],
    depth: int = 1,
    *,
    task_ids: SupportsDLPack | Sequence[int] | None = None,
    max_nodes: int | None = None,
    truncate: bool = False,
    merge: bool = True,
    threads: int = 1,
    pool: ThreadPool | None = None,
) -> HorizonView:
    """The horizons of the states ``roots`` (rows of one instance each, ``task_ids``): the states within ``depth``
    actions of a root as nodes, and the actions between them as edges, as a HorizonView for
    ``FlatHorizonEncoder.encode``. ``depth=0`` gives root-only graphs.

    Each graph is built on its own, breadth first. The root is node 0; then, level by level, the successors of the nodes
    of the level before, in node order, each node's successors in the canonical order of ``rl.expand`` (by action schema,
    then by objects). A successor is a node when its state has not been reached in this graph before (``merge=True``,
    first seen wins: the node's number is that of its first appearance), otherwise the edge leads to the node of that
    state, so a state with several predecessors is one node with several incoming edges and a cycle is an edge to an
    earlier node. There is one edge per (node, successor row), so two actions that lead to the same state are two
    edges. The edges of a graph are ordered by destination node, then by action schema, then by objects; edges that
    agree keep the order of their source nodes and of ``rl.expand``. The nodes of the deepest level are not expanded.
    With ``merge=False`` every successor row is a node of its own and the graph is a tree. The graphs, their numbering and
    their order do not depend on ``threads`` or ``pool`` nor on the process.

    Memory. All the graphs of a call are built at once and held in memory. While they are built every node takes
    ``8 * W + 8`` bytes (W state words, its graph and number) and, when merging, 8 more for its hash and 8 to 16 for the
    hash table; every edge takes ``4 * (4 + L)`` bytes (L action label columns). The result (``8 * W * nodes + 4 * (3 +
    L) * edges + 4 * (P + G + 1)`` bytes, P distinct parent pairs and G graphs) is then written from them, so the peak is
    about the sum of the two; expansion works in chunks,
    so nothing else grows with the number of nodes. The number of nodes of a graph is at most the number of
    states within ``depth`` actions of its root, which can grow exponentially with the depth. ``max_nodes`` bounds it:
    a graph that would hold more nodes than ``max_nodes`` (the root included) raises ValueError, or with
    ``truncate=True`` stops growing instead: a successor that is not yet a node is left out together with its edge
    (successors that are nodes already still get their edges), and ``truncated`` of the result is 1 for that graph.
    ``max_nodes=None`` is unbounded. A table of several instances needs ``task_ids``. ``threads`` (0: one per hardware
    thread) or ``pool`` (a ``mymyr.rl.ThreadPool``, one batch at a time) split the expansion.

    Raises ValueError ("mymyr: encoding: ...") for invalid states or task ids, a graph above ``max_nodes``, and more than
    2^31 - 1 nodes or edges in the result, and for numeric tables (horizons of numeric states are not encoded).
    """
    return HorizonView(**_core._build_horizon(
        table, roots, task_ids, depth=depth, max_nodes=max_nodes, truncate=truncate, merge=merge, threads=threads,
        pool=pool,
    ))


def horizon_of_space(
    space: StateSpace,
    roots: SupportsDLPack | Sequence[int] | None = None,
    depth: int = 1,
    *,
    max_nodes: int | None = None,
    truncate: bool = False,
    merge: bool = True,
) -> HorizonView:
    """The horizons of states of a state space, by the same rules as ``horizon`` but over the space's recorded
    transitions instead of expanding states: the graph of ``roots[g]`` (state ids of the space; None: every state, in id
    order) has the states within ``depth`` transitions as nodes, in the order of the space's transitions of each state,
    and one edge per transition with its action label. The space needs its transition labels
    (``labels=True`` when generated) and its task is the one to encode with (``FlatHorizonEncoder(space.task)``);
    ``max_nodes``, ``truncate``, ``merge`` and the memory are as in ``horizon``."""
    return HorizonView(**_core._build_horizon_of_space(
        space, roots, depth=depth, max_nodes=max_nodes, truncate=truncate, merge=merge,
    ))
