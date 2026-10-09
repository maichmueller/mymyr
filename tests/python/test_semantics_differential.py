"""Differential test of the planning semantics: random small tasks against a reference written here.

The tasks mix typed and union-typed (`either`) parameters, objects and constants, negative, quantified, derived and
numeric preconditions (unary minus included), axioms with numeric bodies, conditional and universal effects, numeric
effects in conditional effects, numeric values without an initial value (that an `assign` can give one), division by
zero, and zero and fractional action costs. The reference implements the semantics mymyr documents (a breadth-first
search and Dijkstra over its own states) and is compared with mymyr on: the applicable actions and successors of every
reachable state, the number of reachable states (also from mymyr.datasets.state_space), the optimal plan cost of A*
with every admissible heuristic, the admissibility of h_max and h² for every reachable state, and the validity and cost
of GBFS plans. With a CUDA device: device A*, device heuristics and the device state space as well.
"""

import heapq
import itertools
import math
import os
import random

import pytest

import mymyr
import mymyr.datasets
from mymyr import search

if not hasattr(mymyr, "Domain"):
    pytest.skip("built without the loki front end", allow_module_level=True)

NAN = float("nan")
MAX_STATES = 1500  # tasks with more reachable states are replaced by the next seed
SEEDS = range(int(os.environ.get("MYMYR_DIFFERENTIAL_TASKS", "120")))

PARENT = {"ta": "object", "tb": "ta", "tc": "object"}
PARAM_TYPES = ["ta", "tb", "tc", "object", ("tb", "tc"), ("ta", "tb"), ("tb", "tc", "ta")]
OBJECT_TYPES = ["ta", "tb", "tc", ("tb", "tc")]


def supertypes(t):
    out = []
    while t != "object":
        out.append(t)
        t = PARENT[t]
    return out + ["object"]


def type_text(t):
    return t if isinstance(t, str) else "(either " + " ".join(t) + ")"


def num_text(x):
    return format(x, "g")


# ------------------------------------------------------------------------------------------------ expressions


def ex_text(e):
    if e[0] == "n":
        return num_text(e[1])
    if e[0] == "fn":
        return "(" + " ".join((e[1], *e[2])) + ")"
    if e[0] == "neg":
        return f"(- {ex_text(e[1])})"
    return f"({e[0]} {ex_text(e[1])} {ex_text(e[2])})"


def item_text(it):
    if it[0] == "lit":
        atom = "(" + " ".join((it[2], *it[3])) + ")"
        return atom if it[1] else f"(not {atom})"
    if it[0] == "cmp":
        return f"({it[1]} {ex_text(it[2])} {ex_text(it[3])})"
    body = " ".join(item_text(x) for x in it[3])
    return f"({it[0]} ({it[1]} - {type_text(it[2])}) (and {body}))"


def conj_text(items):
    return "(and " + " ".join(item_text(x) for x in items) + ")"


class Task:
    """A generated task: its PDDL text and the reference semantics."""

    def __init__(self, seed):
        rng = random.Random(seed)
        self.rng = rng
        self.objects = {f"o{i}": rng.choice(OBJECT_TYPES) for i in range(4)}
        self.constants = {"k": rng.choice(["tb", ("tb", "tc")])}
        self.all = {**self.constants, **self.objects}
        self.costs = rng.random() < 0.75
        self.metric = self.costs and rng.random() < 0.7
        self.static_atoms = {("s", (o,)) for o in self.all if rng.random() < 0.5}
        self.static_values = {}
        for o in self.all:
            if rng.random() < 0.85:
                self.static_values[("w", (o,))] = rng.choice([0, 0.5, 1])
            if rng.random() < 0.85:
                self.static_values[("p", (o,))] = rng.choice([0, 0, 0.25, 0.5, 1, 1.5])
        self.axiom = self.make_axiom()
        self.actions = [self.make_action(i) for i in range(rng.randint(3, 5))]
        self.fluent_preds = {a[0] for act in self.actions for b in act["blocks"] for a in b["add"] + b["del"]}
        self.fluent_funcs = {e[1] for act in self.actions for b in act["blocks"] for e in b["num"]}
        atoms = set()
        for o in self.all:
            if rng.random() < 0.4:
                atoms.add(("f", (o,)))
            for o2 in self.all:
                if rng.random() < 0.12:
                    atoms.add(("g", (o, o2)))
        values = {}
        for o in self.all:
            if rng.random() < 0.7:
                values[("v", (o,))] = rng.choice([0, 0.5, 1, 2])
        if rng.random() < 0.8:
            values[("c",)] = rng.choice([0, 0.25, 1])
        # predicates and functions no effect changes are static
        for a in list(atoms):
            if a[0] not in self.fluent_preds:
                atoms.discard(a)
                self.static_atoms.add(a)
        for k in list(values):
            if k[0] not in self.fluent_funcs:
                self.static_values[k if len(k) == 2 else (k[0], ())] = values.pop(k)
        self.init_atoms = frozenset(atoms)
        self.init_values = {self.key(k): v for k, v in values.items()}
        self.goal = self.make_goal()

    @staticmethod
    def key(k):
        return k if len(k) == 2 else (k[0], ())

    # ---------------------------------------------------------------------------------------------- generation

    def objects_of(self, t):
        names = t if isinstance(t, tuple) else (t,)
        out = []
        for o, d in self.all.items():
            declared = d if isinstance(d, tuple) else (d,)
            if any(n in supertypes(x) for x in declared for n in names):
                out.append(o)
        return out

    def term(self, params):
        return self.rng.choice(params) if params and self.rng.random() < 0.85 else "k"

    def num_expr(self, params, depth=0):
        rng = self.rng
        r = rng.random()
        if r < 0.3 or depth > 1:
            return rng.choice([("n", rng.choice([0, 0.5, 1, 2])), ("fn", "c", ()), ("fn", "v", (self.term(params),))])
        if r < 0.45:
            return ("neg", self.num_expr(params, depth + 1))
        if r < 0.55:
            return ("*", ("n", 2), self.num_expr(params, depth + 1))
        if r < 0.65:
            return ("/", self.num_expr(params, depth + 1), ("fn", "w", (self.term(params),)))
        return (rng.choice("+-"), self.num_expr(params, depth + 1), self.num_expr(params, depth + 1))

    def comparison(self, params):
        rng = self.rng
        lhs = self.num_expr(params)
        if rng.random() < 0.3:
            lhs = ("neg", ("fn", "c", ()))  # a unary minus over a nullary function: a nullary constraint
        return ("cmp", rng.choice(["<", "<=", ">", ">=", "="]), lhs, ("n", rng.choice([0, 0.5, 1, 2])))

    def literal(self, params, preds=("f", "g", "s")):
        rng = self.rng
        p = rng.choice(preds)
        args = (self.term(params),) if p != "g" else (self.term(params), self.term(params))
        return ("lit", p == "s" or rng.random() < 0.65, p, args)

    def condition(self, params, n, derived=True, numeric=True, quantified=True):
        rng = self.rng
        items = []
        for _ in range(n):
            r = rng.random()
            if r < 0.15 and numeric:
                items.append(self.comparison(params))
            elif r < 0.27 and derived:
                items.append(("lit", rng.random() < 0.6, "d", (self.term(params),)))
            elif r < 0.37 and quantified:
                z = f"?q{len(items)}"
                body = [("lit", True, "g", (self.term(params), z))] if rng.random() < 0.5 else [("lit", True, "f", (z,))]
                items.append((rng.choice(["exists", "forall"]), z, rng.choice(PARAM_TYPES), body))
            else:
                items.append(self.literal(params))
        return items

    def make_axiom(self):
        rng = self.rng
        x = ["?x"]
        disjuncts = [[("lit", True, "f", ("?x",))]]
        if rng.random() < 0.6:
            disjuncts.append([("lit", True, "s", ("?x",)), self.comparison(x)])
        if rng.random() < 0.6:
            disjuncts.append([("exists", "?y", "object", [("lit", True, "g", ("?x", "?y")), ("lit", True, "s", ("?y",))])])
        return disjuncts

    def numeric_effect(self, params):
        """(op, function, args, expression) and the guard that keeps the values bounded."""
        rng = self.rng
        target = ("c", ()) if rng.random() < 0.4 else ("v", (self.term(params),))
        t = ("fn", target[0], target[1])
        op = rng.choice(["assign", "assign", "increase", "decrease", "scale-up", "scale-down"])
        if op == "assign":
            e = self.num_expr(params)
            result = e
        elif op in ("increase", "decrease"):
            e = ("n", rng.choice([0.5, 1])) if rng.random() < 0.7 else ("fn", "v", (self.term(params),))
            result = ("+" if op == "increase" else "-", t, e)
        elif op == "scale-up":
            e = ("n", 2)
            result = ("*", t, e)
        else:
            e = ("fn", "w", (self.term(params),))  # 0 (undefined), 0.5 or 1
            result = ("*", ("n", 2), t)
        guard = [("cmp", "<=", result, ("n", 4)), ("cmp", ">=", result, ("n", -4))]
        return (op, target[0], target[1], e), guard

    def block(self, params, conditional):
        rng = self.rng
        b = {"forall": [], "cond": [], "add": [], "del": [], "num": [], "cost": None}
        scope = list(params)
        if conditional and rng.random() < 0.4:
            z = "?z"
            b["forall"].append((z, rng.choice(PARAM_TYPES)))
            scope.append(z)
        if conditional and not (b["forall"] and rng.random() < 0.4):  # a forall effect needs no condition
            b["cond"] = self.condition(scope, rng.randint(1, 2), quantified=False)
        used = set()
        for _ in range(rng.randint(1, 3)):
            lit = self.literal(scope, ("f", "g"))
            if (lit[2], lit[3]) in used:
                continue
            used.add((lit[2], lit[3]))
            (b["add"] if lit[1] else b["del"]).append((lit[2], lit[3]))
        targets = set()
        for _ in range(rng.choice([0, 0, 1, 1, 2])):
            eff, guard = self.numeric_effect(scope)
            if (eff[1], eff[2]) in targets:
                continue  # one effect per lifted target and block (loki merges several)
            targets.add((eff[1], eff[2]))
            b["num"].append(eff)
            b["cond"] += guard
        if self.costs and (not conditional or rng.random() < 0.3):
            b["cost"] = ("n", rng.choice([0, 0, 0.25, 0.5, 1, 2])) if rng.random() < 0.6 else ("fn", "p", (self.term(scope),))
        return b

    def make_action(self, i):
        rng = self.rng
        params = [f"?x{j}" for j in range(rng.randint(0, 2))]
        types = [rng.choice(PARAM_TYPES) for _ in params]
        blocks = [self.block(params, False)]
        guard = blocks[0].pop("cond")
        blocks[0]["cond"] = []
        for _ in range(rng.choice([0, 1, 1, 2])):
            blocks.append(self.block(params, True))
        pre = self.condition(params, rng.randint(0, 3)) + guard
        return {"name": f"a{i}", "params": params, "types": types, "pre": pre, "blocks": blocks}

    def make_goal(self):
        rng = self.rng
        objs = list(self.all)
        items = []
        for _ in range(rng.randint(1, 3)):
            r = rng.random()
            if r < 0.15:
                items.append(("cmp", rng.choice([">=", "<="]), ("fn", "c", ()), ("n", rng.choice([0.5, 1]))))
            elif r < 0.25:
                items.append(("lit", True, "d", (rng.choice(objs),)))
            else:
                p = rng.choice(["f", "g"])
                args = (rng.choice(objs),) if p == "f" else (rng.choice(objs), rng.choice(objs))
                items.append(("lit", rng.random() < 0.8, p, args))
        if rng.random() < 0.25:
            return items
        # most goals hold in a state a short random walk reaches: those tasks are solvable
        state = as_state(self, self.initial())
        for _ in range(rng.randint(1, 6)):
            succ = self.successors(state)
            if not succ:
                break
            state = as_state(self, succ[rng.choice(sorted(succ))][0])
        reachable = []
        for it in items:
            if it[0] == "cmp":
                flipped = ("cmp", {">=": "<", "<=": ">"}[it[1]], it[2], it[3])
                it = it if self.conj([it], state, {}) else flipped if self.conj([flipped], state, {}) else None
            elif not self.conj([it], state, {}):
                it = ("lit", not it[1], it[2], it[3])
            if it is not None:
                reachable.append(it)
        return reachable

    # ---------------------------------------------------------------------------------------------- PDDL

    def pddl(self):
        def block_effects(b):
            parts = [f"({p} {' '.join(a)})" for p, a in b["add"]]
            parts += [f"(not ({p} {' '.join(a)}))" for p, a in b["del"]]
            parts += [f"({op} ({fn}{''.join(' ' + x for x in args)}) {ex_text(e)})" for op, fn, args, e in b["num"]]
            if b["cost"] is not None:
                parts.append(f"(increase (total-cost) {ex_text(b['cost'])})")
            return parts

        actions = []
        for a in self.actions:
            params = " ".join(f"{p} - {type_text(t)}" for p, t in zip(a["params"], a["types"], strict=True))
            effects = block_effects(a["blocks"][0])
            for b in a["blocks"][1:]:
                e = f"(and {' '.join(block_effects(b))})"
                if b["cond"]:
                    e = f"(when {conj_text(b['cond'])} {e})"
                if b["forall"]:
                    z, t = b["forall"][0]
                    e = f"(forall ({z} - {type_text(t)}) {e})"
                effects.append(e)
            actions.append(f"(:action {a['name']} :parameters ({params}) :precondition {conj_text(a['pre'])} "
                           f":effect (and {' '.join(effects)}))")
        axiom = "(:derived (d ?x - object) (or " + " ".join(conj_text(c) for c in self.axiom) + "))"
        functions = "(v ?x - object) (c) (w ?x - object) (p ?x - object)" + (" (total-cost)" if self.costs else "")
        domain = (
            "(define (domain diff) (:requirements :typing :negative-preconditions :disjunctive-preconditions :conditional-effects "
            ":universal-preconditions :existential-preconditions :derived-predicates :numeric-fluents :action-costs) "
            "(:types tb - ta ta tc) "
            f"(:constants {' '.join(f'{o} - {type_text(t)}' for o, t in self.constants.items())}) "
            "(:predicates (f ?x - object) (g ?x - object ?y - object) (s ?x - object) (d ?x - object)) "
            f"(:functions {functions}) {axiom} {' '.join(actions)})"
        )
        init = [f"({p} {' '.join(a)})" for p, a in sorted(self.init_atoms | self.static_atoms)]
        values = {**self.static_values, **self.init_values}
        init += [f"(= ({fn}{''.join(' ' + x for x in args)}) {num_text(v)})" for (fn, args), v in sorted(values.items())]
        if self.costs:
            init.append("(= (total-cost) 0)")
        metric = "(:metric minimize (total-cost))" if self.metric else ""
        objects = " ".join(f"{o} - {type_text(t)}" for o, t in self.objects.items())
        problem = (f"(define (problem p) (:domain diff) (:objects {objects}) (:init {' '.join(init)}) "
                   f"(:goal {conj_text(self.goal)}) {metric})")
        return domain, problem

    # ---------------------------------------------------------------------------------------------- reference

    def value(self, state, fn, args):
        k = (fn, tuple(args))
        if fn in self.fluent_funcs:
            v = state[1].get(k)
        else:
            v = self.static_values.get(k)
        return NAN if v is None else v

    def ev(self, e, state, env):
        k = e[0]
        if k == "n":
            return e[1]
        if k == "fn":
            return self.value(state, e[1], [env.get(x, x) for x in e[2]])
        if k == "neg":
            return -self.ev(e[1], state, env)
        a, b = self.ev(e[1], state, env), self.ev(e[2], state, env)
        if k == "+":
            return a + b
        if k == "-":
            return a - b
        if k == "*":
            return a * b
        return NAN if b == 0 else a / b

    def holds(self, pred, args, state):
        if pred == "d":
            return any(self.conj(c, state, {"?x": args[0]}) for c in self.axiom)
        if pred in self.fluent_preds:
            return (pred, args) in state[0]
        return (pred, args) in self.static_atoms

    def conj(self, items, state, env):
        for it in items:
            if it[0] == "lit":
                if self.holds(it[2], tuple(env.get(x, x) for x in it[3]), state) != it[1]:
                    return False
            elif it[0] == "cmp":
                a, b = self.ev(it[2], state, env), self.ev(it[3], state, env)
                if math.isnan(a) or math.isnan(b):
                    return False
                ok = {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "=": a == b}[it[1]]
                if not ok:
                    return False
            else:
                q, var, t, body = it
                results = (self.conj(body, state, {**env, var: o}) for o in self.objects_of(t))
                if not (any(results) if q == "exists" else all(results)):
                    return False
        return True

    def successors(self, state):
        """{action text: (child, cost)}."""
        out = {}
        for a in self.actions:
            for binding in itertools.product(*(self.objects_of(t) for t in a["types"])):
                env = dict(zip(a["params"], binding, strict=True))
                if not self.conj(a["pre"], state, env):
                    continue
                r = self.apply(a, state, env)
                if r is not None:
                    out["(" + " ".join((a["name"], *binding)) + ")"] = r
        return out

    def apply(self, a, state, env):
        adds, dels, writes, family = set(), set(), [], {}
        cost = 0
        for b in a["blocks"]:
            envs = [env]
            if b["forall"]:
                z, t = b["forall"][0]
                envs = [{**env, z: o} for o in self.objects_of(t)]
            for e in envs:
                if not self.conj(b["cond"], state, e):
                    continue
                for op, fn, args, expr in b["num"]:
                    k = (fn, tuple(e.get(x, x) for x in args))
                    f = 1 if op == "assign" else 2 if op in ("increase", "decrease") else 3
                    if k in family and (family[k] != f or f == 1):
                        return None  # conflicting effects on one target
                    family[k] = f
                    v = self.ev(expr, state, e)
                    if math.isnan(v) or (op == "scale-down" and v == 0):
                        return None  # an undefined effect
                    if op != "assign" and k not in state[1]:
                        return None  # only assign gives an undefined function a value
                    writes.append((k, op, v))
                if b["cost"] is not None:
                    v = self.ev(b["cost"], state, e)
                    if math.isnan(v):
                        return None
                    cost += v
                adds |= {(p, tuple(e.get(x, x) for x in args)) for p, args in b["add"]}
                dels |= {(p, tuple(e.get(x, x) for x in args)) for p, args in b["del"]}
        nums = dict(state[1])
        for k, op, v in writes:
            old = nums.get(k)
            nums[k] = {"assign": lambda: v, "increase": lambda: old + v, "decrease": lambda: old - v,
                       "scale-up": lambda: old * v, "scale-down": lambda: old / v}[op]() + 0.0
        atoms = (state[0] - dels) | adds
        return (frozenset(atoms), tuple(sorted(nums.items()))), (cost if self.costs else 1)

    def initial(self):
        return (self.init_atoms, tuple(sorted(self.init_values.items())))

    def is_goal(self, state):
        return self.conj(self.goal, state, {})


def as_state(t, state):
    return (state[0], dict(state[1]))


class Reference:
    """The reachable states of a task under the reference semantics, with the transitions and goal distances."""

    def __init__(self, t):
        self.t = t
        s0 = t.initial()
        self.index = {s0: 0}
        self.states = [s0]
        self.edges = []
        for s in self.states:
            succ = t.successors(as_state(t, s))
            self.edges.append(succ)
            for child, _ in succ.values():
                if child not in self.index:
                    self.index[child] = len(self.states)
                    self.states.append(child)
            if len(self.states) > MAX_STATES:
                raise OverflowError
        self.goal = [t.is_goal(as_state(t, s)) for s in self.states]
        # h*: Dijkstra from the goal states over the reversed transitions
        rev = [[] for _ in self.states]
        for i, succ in enumerate(self.edges):
            for child, cost in succ.values():
                rev[self.index[child]].append((i, cost))
        self.hstar = [math.inf] * len(self.states)
        heap = []
        for i, g in enumerate(self.goal):
            if g:
                self.hstar[i] = 0
                heap.append((0, i))
        heapq.heapify(heap)
        while heap:
            d, i = heapq.heappop(heap)
            if d > self.hstar[i]:
                continue
            for j, c in rev[i]:
                if d + c < self.hstar[j]:
                    self.hstar[j] = d + c
                    heapq.heappush(heap, (d + c, j))

    def plan_cost(self, plan):
        """The cost of a plan text list under the reference, or None if it is not a valid plan."""
        i, cost = 0, 0
        for a in plan:
            if a not in self.edges[i]:
                return None
            child, c = self.edges[i][a]
            i, cost = self.index[child], cost + c
        return cost if self.goal[i] else None


def action_text(a):
    """An action as the PDDL writes it: without the parameters normalization added (existential variables)."""
    return "(" + " ".join([a.name, *(str(o) for o in a.objects[: a.original_arity])]) + ")"


def mymyr_key(t, task, state):
    atoms = set()
    for o in t.all:
        if "f" in t.fluent_preds and state.holds(f"(f {o})"):
            atoms.add(("f", (o,)))
        for o2 in t.all:
            if "g" in t.fluent_preds and state.holds(f"(g {o} {o2})"):
                atoms.add(("g", (o, o2)))
    nums = {}
    for name, v in zip(task.numeric_names, state.numeric_values(), strict=True):
        if not math.isnan(v):
            parts = name.strip("()").split()
            nums[(parts[0], tuple(parts[1:]))] = v + 0.0
    return frozenset(atoms), tuple(sorted(nums.items()))


def generated(count):
    """(seed, task, mymyr task, reference) for `count` tasks within MAX_STATES reachable states."""
    seed = 0
    out = []
    while len(out) < count:
        t = Task(seed)
        domain, problem = t.pddl()
        try:
            ref = Reference(t)
        except OverflowError:
            seed += 1
            continue
        task = mymyr.Task(mymyr.Domain.from_string(domain).instantiate_string(problem))
        out.append((seed, t, task, ref))
        seed += 1
    return out


@pytest.fixture(scope="module")
def tasks():
    return generated(len(SEEDS))


def close(a, b):
    return (math.isinf(a) and math.isinf(b)) or abs(a - b) <= 1e-9 * max(1.0, abs(b))


def explore(t, task, ref):
    """mymyr's state of every reference state (checking the successors on the way)."""
    states = {0: task.initial_state}
    assert mymyr_key(t, task, task.initial_state) == ref.states[0]
    for i in range(len(ref.states)):
        s = states[i]
        want = ref.edges[i]
        got = {}
        for action, child in task.successors(s):
            got.setdefault(action_text(action), set()).add(mymyr_key(t, task, child))
        assert set(got) == set(want), (i, sorted(got), sorted(want))
        for name, (child, _) in want.items():
            assert got[name] == {child}, (i, name)
            j = ref.index[child]
            if j not in states:
                for action, c in task.successors(s):
                    if action_text(action) == name:
                        states[j] = c
                        break
        assert task.is_goal(s) == ref.goal[i]
    return [states[i] for i in range(len(ref.states))]


def test_semantics_against_the_reference(tasks):
    total_states = 0
    for seed, t, task, ref in tasks:
        states = explore(t, task, ref)
        total_states += len(states)
        space = mymyr.datasets.state_space(task, remove_if_unsolvable=False)
        assert space is not None and space.num_states == len(ref.states), seed
        want = ref.hstar[0]
        for kind in ("blind", "max", "h2", "perfect"):
            r = search.astar(task, heuristic=kind)
            if math.isinf(want):
                assert not r.solved, (seed, kind)
                continue
            assert r.solved, (seed, kind, r.status, r.message)
            assert close(r.cost, want), (seed, kind, r.cost, want)
            plan = [action_text(a) for a in r.plan]
            assert ref.plan_cost(plan) is not None and close(ref.plan_cost(plan), want), (seed, kind)
        for kind in ("max", "h2"):
            h = search.Heuristic(task, kind)
            for i, s in enumerate(states):
                assert h(s) <= ref.hstar[i] + 1e-9, (seed, kind, i, h(s), ref.hstar[i])
        r = search.gbfs(task, heuristic="ff")
        assert r.solved == (not math.isinf(want)), seed
        if r.solved:
            cost = ref.plan_cost([action_text(a) for a in r.plan])
            assert cost is not None and close(r.cost, cost), (seed, r.cost, cost)
    print(f"differential: {len(tasks)} tasks, {total_states} states")


def test_device_against_the_reference(tasks):
    mc = pytest.importorskip("mymyr.cuda", reason="mymyr was built without the CUDA backend", exc_type=ImportError)
    if not mc.available():
        pytest.skip("no visible CUDA device (set CUDA_VISIBLE_DEVICES)")
    ctx = mc.Context(0, max_bytes=1 << 30)
    searched = spaces = 0
    for seed, t, task, ref in tasks:
        want = ref.hstar[0]
        try:
            r = mc.astar(task, heuristic="max", ctx=ctx)
        except ValueError:
            r = None  # a task the device does not run
        if r is not None:
            searched += 1
            if math.isinf(want):
                assert not r.solved, seed
            else:
                assert r.solved and close(r.cost, want), (seed, r.status, r.cost, want)
        try:
            space = mymyr.datasets.state_space(task, device=ctx, remove_if_unsolvable=False)
        except ValueError:
            space = None
        if space is not None:
            spaces += 1
            assert space.num_states == len(ref.states), seed
        try:
            h = mc.Heuristic(task, "max", ctx=ctx)
        except ValueError:
            continue
        states = explore(t, task, ref)
        for i, v in enumerate(h.evaluate(states)):
            assert v <= ref.hstar[i] + 1e-9, (seed, i, v, ref.hstar[i])
    ctx.synchronize()
    print(f"device differential: {len(tasks)} tasks, {searched} device searches, {spaces} device state spaces")
    assert searched >= len(tasks) // 2 and spaces >= len(tasks) // 2
