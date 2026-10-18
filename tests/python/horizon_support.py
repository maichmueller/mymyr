"""Horizon views of the tests: the states within `depth` actions of some roots, built with rl.expand."""

import numpy as np

from mymyr import encoding, rl


def expand_horizons(table, roots, task_ids=None, depth=1, max_children=None):
    """A HorizonView of the horizons of `roots` (a state array [G, W]): the nodes of graph g are its
    root, then the states of each level in the order of their parents and the canonical order of the actions (the first
    `max_children` actions of a node, if given); the edges are the expansion rows, one per child, in that order.
    Equal states reached twice are separate nodes."""
    roots = np.ascontiguousarray(roots, dtype=np.uint64)
    graphs = len(roots)
    ids = np.zeros(graphs, np.int32) if task_ids is None else np.asarray(task_ids, np.int32)
    nodes = [[roots[g]] for g in range(graphs)]
    edges = [[] for _ in range(graphs)]
    frontier = [(g, 0) for g in range(graphs)]
    for _ in range(depth):
        if not frontier:
            break
        states = np.stack([nodes[g][k] for g, k in frontier])
        expansion = rl.expand(table, states, ids[[g for g, _ in frontier]])
        succ = np.asarray(expansion.succ)
        schema, binding, offsets = np.asarray(expansion.schema), np.asarray(expansion.binding), np.asarray(expansion.offsets)
        assert succ.shape[1] == roots.shape[1], "the roots have the width of the expansion"
        following = []
        for i, (g, k) in enumerate(frontier):
            first, last = int(offsets[i]), int(offsets[i + 1])
            if max_children is not None:
                last = min(last, first + max_children)
            for row in range(first, last):
                nodes[g].append(succ[row])
                edges[g].append((k, len(nodes[g]) - 1, schema[row], binding[row]))
                following.append((g, len(nodes[g]) - 1))
        frontier = following
    width = table.label_width
    node_offsets = np.cumsum([0] + [len(n) for n in nodes]).astype(np.int32)
    edge_offsets = np.cumsum([0] + [len(e) for e in edges]).astype(np.int32)
    flat = [edge for g in edges for edge in g]
    return encoding.HorizonView(
        nodes=np.stack([n for g in nodes for n in g]) if graphs else np.zeros((0, roots.shape[1]), np.uint64),
        node_offsets=node_offsets,
        edge_offsets=edge_offsets,
        edge_src=np.array([e[0] for e in flat], np.int32),
        edge_dst=np.array([e[1] for e in flat], np.int32),
        edge_schema=np.array([e[2] for e in flat], np.int32),
        edge_binding=np.array([e[3] for e in flat], np.int32).reshape(len(flat), width),
        task_ids=ids,
    )
