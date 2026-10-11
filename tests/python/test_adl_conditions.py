"""Conditions of the ADL fragment (negation, disjunction, implication, quantifiers and their nesting, equality,
constants, numeric comparisons) in goals, preconditions, conditional effects (also under forall effects) and derived
predicate bodies, against an evaluation of the formula as written.

For every shape and position the task is built from PDDL, and in every state the base actions reach (every combination
of the fluents the formulas read) the goal test, the derived atoms, the applicable instances of the action under test
and their successors are compared with the formula evaluated here. Search finds plans for guarded forall goals that
are valid against the formula, and the device paths (with a visible CUDA device) agree with the CPU.

Numeric comparisons follow the documented semantics: `not` turns a comparison into the complementary one, so with an
undefined side both are false, and a universal quantifier holds when no binding satisfies the negation of its body
(a comparison with an undefined side does not falsify it).
"""

import math
import re

import pytest

import mymyr
from mymyr import search

if not hasattr(mymyr, "Domain"):
    pytest.skip("built without the loki front end", allow_module_level=True)

SUPERTYPE = {"a": "object", "b": "a", "c": "object"}
OBJECTS = {"o1": "a", "o2": "b", "k": "b", "c1": "c"}  # k and c1 are constants
STATIC = {("s", ("o1",)), ("s", ("k",)), ("r", ("o1", "o2")), ("r", ("o2", "k")), ("r", ("k", "k")), ("r", ("o2", "o1"))}
VALUES = {("f", ("o1",)): 1.0, ("f", ("o2",)): 3.0}  # (f k) is undefined
FLUENT = {"p", "q", "g", "h", "m"}

DOMAIN = """(define (domain adl)
 (:requirements :typing :adl :derived-predicates :numeric-fluents)
 (:types b - a a c - object)
 (:constants k - b c1 - c)
 (:predicates (s ?x - a) (r ?x - a ?y - a) (p ?x - a) (q ?x - object) (g) (h) (m ?x - a) (d ?x - a) (e))
 (:functions (f ?x - a))
 {derived}
 (:action set-p :parameters (?x - a) :precondition (not (p ?x)) :effect (p ?x))
 (:action del-p :parameters (?x - a) :precondition (p ?x) :effect (not (p ?x)))
 (:action set-q :parameters (?x - object) :precondition (not (q ?x)) :effect (q ?x))
 (:action flip :parameters () :precondition (and) :effect (and (when (g) (not (g))) (when (not (g)) (g))))
 (:action act :parameters (?x - a) :precondition {pre} :effect (and (h) {eff})))
"""

PROBLEM = """(define (problem adl-p) (:domain adl)
 (:objects o1 - a o2 - b)
 (:init (s o1) (s k) (r o1 o2) (r o2 k) (r k k) (r o2 o1) (= (f o1) 1) (= (f o2) 3))
 (:goal {goal}))
"""

# formulas over a free term X of type a
SHAPES = {
    "forall-guarded": "(forall (?y - a) (imply (s ?y) (p ?y)))",
    "forall-free": "(forall (?y - a) (imply (r X ?y) (p ?y)))",
    "forall-or": "(forall (?y - b) (or (p ?y) (q ?y)))",
    "forall-equality": "(forall (?y - a) (or (= ?y X) (not (p ?y))))",
    "forall-either": "(forall (?y - (either b c)) (q ?y))",
    "forall-object": "(forall (?y - object) (imply (q ?y) (or (= ?y c1) (= ?y X))))",
    "forall-two": "(forall (?y ?z - a) (imply (and (r ?y ?z) (p ?y)) (p ?z)))",
    "exists-and": "(exists (?y - a) (and (r X ?y) (not (p ?y))))",
    "exists-two": "(exists (?y - a ?z - object) (and (r ?y X) (q ?z) (not (= ?z ?y))))",
    "not-exists": "(not (exists (?y - a) (and (s ?y) (p ?y))))",
    "or-ands": "(or (and (g) (p X)) (and (not (g)) (q X)))",
    "imply": "(imply (p X) (and (q X) (g)))",
    "not-and": "(not (and (p X) (g)))",
    "not-or": "(not (or (p X) (q c1)))",
    "nested-forall-exists": "(forall (?y - a) (imply (p ?y) (exists (?z - a) (and (r ?y ?z) (q ?z)))))",
    "nested-exists-forall": "(exists (?y - a) (and (q ?y) (forall (?z - a) (imply (r ?y ?z) (p ?z)))))",
    "nested-forall-forall": "(forall (?y - a) (imply (p ?y) (forall (?z - a) (imply (r ?y ?z) (q ?z)))))",
    "nested-negations": "(not (forall (?y - a) (or (not (p ?y)) (exists (?z - a) (and (r ?y ?z) (not (q ?z)))))))",
    "or-quantifiers": "(or (forall (?y - a) (p ?y)) (exists (?y - b) (and (q ?y) (not (p X)))))",
    "constants": "(or (p k) (and (q k) (not (= X k))))",
    "numeric-forall": "(forall (?y - a) (imply (p ?y) (>= (f ?y) 2)))",
    "numeric-exists": "(exists (?y - a) (and (p ?y) (< (f ?y) 2)))",
    "numeric-or": "(or (g) (> (f X) 2))",
    "numeric-not": "(not (and (p X) (<= (f X) 1)))",
}

CONTEXTS = ["goal", "goal-and", "pre", "when", "forall-when", "derived", "derived-goal"]


def fill(shape, x):
    return re.sub(r"\bX\b", x, shape)


def build(shape, context):
    parts = {"derived": "", "pre": "(and)", "eff": "", "goal": "(h)"}
    if context == "goal":
        parts["goal"] = fill(shape, "o1")
    elif context == "goal-and":
        parts["goal"] = "(and (q o2) " + fill(shape, "o2") + ")"
    elif context == "pre":
        parts["pre"] = fill(shape, "?x")
    elif context == "when":
        parts["eff"] = "(when " + fill(shape, "?x") + " (m ?x))"
    elif context == "forall-when":
        parts["eff"] = "(forall (?w - a) (when (and (not (= ?w ?x)) " + fill(shape, "?w") + ") (m ?w)))"
    else:
        parts["derived"] = "(:derived (d ?x - a) " + fill(shape, "?x") + ")"
        if context == "derived-goal":
            parts["goal"] = "(and (not (g)) (forall (?y - a) (imply (s ?y) (d ?y))))"
    return DOMAIN.format(**parts), PROBLEM.format(goal=parts["goal"]), parts


# ------------------------------------------------------------------------------------------------ reference


def parse(text):
    tokens = re.findall(r"\(|\)|[^\s()]+", text)
    pos = 0

    def node():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok != "(":
            return tok
        out = []
        while tokens[pos] != ")":
            out.append(node())
        pos += 1
        return out

    return node()


def typed_list(items):
    """[(variable, type)] of a PDDL typed list; untyped variables are objects."""
    out, pending = [], []
    i = 0
    while i < len(items):
        if items[i] == "-":
            for v in pending:
                out.append((v, items[i + 1]))
            pending = []
            i += 2
        else:
            pending.append(items[i])
            i += 1
    return out + [(v, "object") for v in pending]


def is_a(obj, t):
    if isinstance(t, list):  # (either ...)
        return any(is_a(obj, x) for x in t[1:])
    x = OBJECTS[obj]
    while True:
        if x == t:
            return True
        if x == "object":
            return t == "object"
        x = SUPERTYPE[x]


COMPARE = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "=": lambda a, b: a == b,
}
COMPLEMENT = {"<": ">=", "<=": ">", ">": "<=", ">=": "<"}


class Reference:
    def __init__(self, fluents, derived_rule):
        self.atoms = STATIC | fluents
        self.derived_rule = derived_rule  # (:derived (d ?x - a) body) as (variable, body), or None

    def term(self, t, env):
        return env.get(t, t)

    def value(self, e, env):
        if not isinstance(e, list):
            return float(e)
        return VALUES.get((e[0], tuple(self.term(t, env) for t in e[1:])))

    def bindings(self, params, env):
        out = [dict(env)]
        for v, t in typed_list(params):
            out = [{**b, v: o} for b in out for o in OBJECTS if is_a(o, t)]
        return out

    def atom(self, pred, args):
        if pred == "d":
            if self.derived_rule is None:
                return False
            var, body = self.derived_rule
            return self.holds(body, {var: args[0]})
        return (pred, args) in self.atoms

    def holds(self, phi, env, pos=True):
        op = phi[0]
        if op in ("and", "or"):
            parts = [self.holds(x, env, pos) for x in phi[1:]]
            return all(parts) if (op == "and") == pos else any(parts)
        if op == "not":
            return self.holds(phi[1], env, not pos)
        if op == "imply":
            return self.holds(["or", ["not", phi[1]], phi[2]], env, pos)
        if op in ("exists", "forall"):
            bindings = self.bindings(phi[1], env)
            if (op == "exists") == pos:
                return any(self.holds(phi[2], b, pos) for b in bindings)
            return not any(self.holds(phi[2], b, not pos) for b in bindings)
        if op in COMPARE and (op != "=" or isinstance(phi[1], list) or isinstance(phi[2], list)):
            a, b = self.value(phi[1], env), self.value(phi[2], env)
            if a is None or b is None or math.isnan(a) or math.isnan(b):
                return False
            if not pos:
                return COMPARE[COMPLEMENT[op]](a, b) if op != "=" else a != b
            return COMPARE[op](a, b)
        if op == "=":
            return (self.term(phi[1], env) == self.term(phi[2], env)) == pos
        return self.atom(op, tuple(self.term(t, env) for t in phi[1:])) == pos


# ------------------------------------------------------------------------------------------------ mymyr side


def atom_key(a):
    parts = str(a).strip("()").split()
    return (parts[0], tuple(parts[1:]))


def base_states(task):
    """The states the actions other than act reach: every combination of p, q and g."""
    start = task.initial_state
    seen, frontier = {start}, [start]
    while frontier:
        s = frontier.pop()
        for a in task.applicable_actions(s):
            if str(a).startswith("(act"):
                continue
            t = task.apply(s, a)
            if t not in seen:
                seen.add(t)
                frontier.append(t)
    return seen


def check(shape, context):
    domain, problem, parts = build(shape, context)
    task = mymyr.Task(mymyr.Domain.from_string(domain).instantiate_string(problem))
    goal = parse(parts["goal"])
    derived_rule = None
    if parts["derived"]:
        derived_rule = ("?x", parse(fill(shape, "?x")))
    a_objects = [o for o in OBJECTS if is_a(o, "a")]
    states = base_states(task)
    assert len(states) == 2 ** 3 * 2 ** 4 * 2
    for s in states:
        fluents = {atom_key(a) for a in s.atoms()}
        ref = Reference(fluents, derived_rule)
        assert task.is_goal(s) == ref.holds(goal, {}), (shape, context, sorted(fluents))
        derived = {atom_key(a) for a in task.derived_atoms(s) if atom_key(a)[0] in ("d", "e")}
        assert derived == {("d", (o,)) for o in a_objects if ref.atom("d", (o,))}, (shape, context, sorted(fluents))
        # an existential precondition adds its variables to the action's parameters: one instance per witness
        acts = [(atom_key(a)[1][0], a) for a in task.applicable_actions(s) if str(a).startswith("(act")]
        pre = parse(parts["pre"])
        assert {o for o, _ in acts} == {o for o in a_objects if ref.holds(pre, {"?x": o})}, (shape, context, sorted(fluents))
        for o, a in acts:
            want = set(fluents) | {("h", ())}
            if context == "when" and ref.holds(parse(fill(shape, o)), {}):
                want.add(("m", (o,)))
            if context == "forall-when":
                want |= {("m", (w,)) for w in a_objects if w != o and ref.holds(parse(fill(shape, w)), {})}
            assert {atom_key(x) for x in task.apply(s, a).atoms()} == want, (shape, context, str(a), sorted(fluents))


@pytest.mark.parametrize("context", CONTEXTS)
@pytest.mark.parametrize("shape", list(SHAPES))
def test_conditions_against_the_formula(shape, context):
    check(SHAPES[shape], context)


def test_fast_init_and_loki_init_agree():
    for shape in ("forall-guarded", "nested-negations", "numeric-forall"):
        for context in ("goal", "forall-when"):
            domain, problem, _ = build(SHAPES[shape], context)
            d = mymyr.Domain.from_string(domain)
            fast = d.instantiate_string(problem).to_text()
            assert fast == d.instantiate_string(problem, fast_init=False).to_text()


def test_helper_predicates_are_derived_axiom_predicates():
    domain, problem, _ = build(SHAPES["forall-free"], "forall-when")
    task = mymyr.Domain.from_string(domain).instantiate_string(problem)
    names = {p.name: p for p in task.predicates}
    helpers = sorted(n for n in names if n.startswith("axiom_"))
    assert helpers and all(names[n].kind == "derived" for n in helpers)
    # the forall effect's condition is lifted into a rule over its free variables (?w and ?x)
    assert any(names[n].arity == 2 for n in helpers)


GOAL = "(and (forall (?y - a) (imply (s ?y) (p ?y))) (forall (?y - (either b c)) (imply (not (= ?y k)) (q ?y))) (not (g)))"


@pytest.mark.parametrize("algorithm", ["astar", "gbfs", "brfs"])
def test_search_plans_satisfy_guarded_forall_goals(algorithm):
    domain, problem = DOMAIN.format(derived="", pre="(and)", eff=""), PROBLEM.format(goal=GOAL)
    task = mymyr.Task(mymyr.Domain.from_string(domain).instantiate_string(problem))
    result = getattr(search, algorithm)(task)
    assert result.solved
    state = task.initial_state
    assert not Reference({atom_key(a) for a in state.atoms()}, None).holds(parse(GOAL), {})
    for action in result.plan:
        state = task.apply(state, action)
    assert Reference({atom_key(a) for a in state.atoms()}, None).holds(parse(GOAL), {})
    if algorithm != "gbfs":
        assert len(result.plan) == 4  # (set-p o1), (set-p k), (set-q o2), (set-q c1)


def test_universal_recursion_through_a_derived_predicate_is_not_stratifiable():
    domain = DOMAIN.format(derived="(:derived (d ?x - a) (forall (?y - a) (imply (r ?x ?y) (d ?y))))", pre="(and)", eff="")
    with pytest.raises(ValueError, match="stratifiable"):
        mymyr.Task(mymyr.Domain.from_string(domain).instantiate_string(PROBLEM.format(goal="(d o1)")))


# ------------------------------------------------------------------------------------------------ device paths


DEVICE_CASES = [
    ("forall-guarded", "goal"),
    ("nested-negations", "goal-and"),
    ("or-quantifiers", "pre"),
    ("forall-free", "forall-when"),
    ("nested-exists-forall", "derived-goal"),
    ("numeric-forall", "goal"),
]


@pytest.mark.parametrize("shape,context", DEVICE_CASES)
def test_device_paths_agree_with_the_cpu(shape, context):
    mc = pytest.importorskip("mymyr.cuda", reason="mymyr was built without the CUDA backend", exc_type=ImportError)
    if not mc.available():
        pytest.skip("no visible CUDA device (set CUDA_VISIBLE_DEVICES)")
    import mymyr.datasets

    domain, problem, _ = build(SHAPES[shape], context)
    task = mymyr.Task(mymyr.Domain.from_string(domain).instantiate_string(problem))
    ctx = mc.Context(0, max_bytes=1 << 30)
    cpu = search.brfs(task, fingerprint=True, stop_at_goal=False)
    gpu = mc.brfs(task, ctx=ctx, fingerprint=True, stop_at_goal=False)
    assert (gpu.states, gpu.generated, gpu.goal_states, gpu.fingerprint) == (cpu.states, cpu.generated, cpu.goal_states, cpu.fingerprint)
    space = mymyr.datasets.state_space(task, remove_if_unsolvable=False)
    device_space = mymyr.datasets.state_space(task, device=ctx, remove_if_unsolvable=False)
    assert device_space.num_states == space.num_states
    assert sum(device_space.is_goal(i) for i in range(device_space.num_states)) == sum(space.is_goal(i) for i in range(space.num_states))
    ctx.synchronize()
