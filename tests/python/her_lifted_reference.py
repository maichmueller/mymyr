"""A pure-Python reference of the goal templates of lifted hindsight relabelling, on mymyr Tasks (a test oracle).

It follows the semantics documented for ``mymyr.rl.lifted_goal_templates`` step by step with plain Python: goal
literals, the literal graph, connected subsets in lexicographic order (a depth-first search over index tuples with a
simple reachability prune, every subset checked for connectivity), component draws from a Python port of the
Philox-4x32-10 counter RNG, products, the per-size quota, and ``GroundCondition.lift``.
"""

import itertools
import math

MASK32 = 0xFFFFFFFF
K_TEMPLATES = 2  # the RNG purpose of template draws


def philox4x32_10(counter, k0, k1):
    c0, c1, c2, c3 = counter
    for _ in range(10):
        p0 = 0xD2511F53 * c0
        p1 = 0xCD9E8D57 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0) & MASK32, p1 & MASK32, ((p0 >> 32) ^ c3 ^ k1) & MASK32, p0 & MASK32
        k0 = (k0 + 0x9E3779B9) & MASK32
        k1 = (k1 + 0xBB67AE85) & MASK32
    return c0, c1, c2, c3


def bits64(seed, env, draw, purpose):
    b = philox4x32_10((draw & MASK32, (draw >> 32) & MASK32, env & MASK32, purpose), seed & MASK32, (seed >> 32) & MASK32)
    return b[0] | (b[1] << 32)


def below(x, n):
    return (((x >> 32) * n) + (((x & MASK32) * n) >> 32)) >> 32


def _bits(x):
    while x:
        low = x & -x
        yield low.bit_length() - 1
        x ^= low


def _reach(start, allowed, adj):
    """The vertices reachable from `start` within the vertex set `allowed` (bitsets)."""
    seen, frontier = start, start
    while frontier:
        nxt = 0
        for v in _bits(frontier):
            nxt |= adj[v]
        frontier = nxt & allowed & ~seen
        seen |= frontier
    return seen


def connected_subsets(m, adj, k, limit):
    """The first `limit` connected k-subsets of vertices 0..m-1 in lexicographic order of their index tuples."""
    out = []
    prefix = []

    def rec(start, pbits):
        for j in range(start, m - (k - len(prefix)) + 1):
            p = pbits | (1 << j)
            prefix.append(j)
            if len(prefix) == k:
                if _reach(1 << prefix[0], p, adj) == p:
                    out.append(tuple(prefix))
            else:
                # the prefix must lie in one component of prefix + later vertices, of at least k vertices
                later = ((1 << m) - 1) & ~((1 << (j + 1)) - 1)
                seen = _reach(1 << prefix[0], p | later, adj)
                if seen & p == p and bin(seen).count("1") >= k:
                    rec(j + 1, p)
            prefix.pop()
            if len(out) >= limit:
                return

    rec(0, 0)
    return out


def subgoals(task, instance, *, max_subgoals_per_size=2, max_size=10, max_subsets_per_size=100, max_component_samples=5,
             max_products=1000, seed=0):
    """Goal literal index tuples of the templates of `task` as instance `instance`, sizes largest first."""
    lits = list(task.goal_condition.literals)
    n = len(lits)
    objs = [set(l.atom.object_indices) for l in lits]
    neighbours = [[j for j in range(n) if j != i and objs[i] & objs[j]] for i in range(n)]
    comps, seen = [], set()
    for i in range(n):
        if i in seen:
            continue
        comp, stack = {i}, [i]
        while stack:
            for u in neighbours[stack.pop()]:
                if u not in comp:
                    comp.add(u)
                    stack.append(u)
        seen |= comp
        comps.append(sorted(comp))
    subsets = []
    for comp in comps:
        local = {v: j for j, v in enumerate(comp)}
        adj = [sum(1 << local[u] for u in neighbours[v]) for v in comp]
        found = []
        for k in range(1, min(len(comp), max_size) + 1):
            found += [tuple(comp[j] for j in s) for s in connected_subsets(len(comp), adj, k, max_subsets_per_size)]
        subsets.append(found)
    total = len(subsets)
    sampled = []
    for c in range(1, min(max_size, total) + 1):
        picks = set()
        for s in range(min(max_component_samples, math.comb(total, c))):
            perm = list(range(total))
            for d in range(c):
                j = d + below(bits64(seed, instance, (c << 32) | (s * c + d), K_TEMPLATES), total - d)
                perm[d], perm[j] = perm[j], perm[d]
            picks.add(tuple(sorted(perm[:c])))
        for pick in sorted(picks):
            for combo in itertools.islice(itertools.product(*[subsets[i] for i in pick]), max_products):
                sampled.append(sum(combo, ()))
    by_size = {}
    for sg in sampled:
        if len(sg) <= max_size:
            by_size.setdefault(len(sg), []).append(sg)
    return [sg for size in sorted(by_size, reverse=True) for sg in by_size[size][:max_subgoals_per_size]]


def declares_equality(task):
    return any(r.lstrip(":") == "equality" for r in task.formalism.requirements)


def templates(task, instance, *, distinct=True, **options):
    """(goal literal index tuples, lifted ConjunctiveConditions) of `task` as instance `instance`."""
    lits = list(task.goal_condition.literals)
    tuples = subgoals(task, instance, **options)
    add = distinct is True or (distinct == "equality" and declares_equality(task))
    return tuples, [task.ground_condition([lits[i] for i in sg]).lift(add) for sg in tuples]
