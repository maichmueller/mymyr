"""Write the mifrost parity fixtures to tests/data/encoding/mifrost/.

Run with the interpreter that has mifrost 0.6.0 and pymimir 0.16.3 installed, isolated and from the repository root:

    python -I scripts/export_mifrost_encoding_fixture.py [--out DIR] [--only F1,F5] [--introspect] [--small-goals]

This is the only place that imports mifrost; the tests read the JSON files and need neither mifrost nor pymimir. The
output is deterministic: running the script twice gives byte-identical files. ``--small-goals`` instead prints the
small semantic fixture ``tests/data/encoding/mifrost_flat_goals.json`` (mifrost's neutral engine, no planner).

Fixtures
--------
One file per task, ``f<NN>_<slug>.json`` (``f01_blocks_smedium.json`` ...). Keys:

``format``
    Format version (1).
``provenance``
    ``producer`` (``"mifrost"``), ``mifrost_version``, ``mifrost_source_checkout`` (the mifrost commit the 0.6.0
    wheel was cut from; the installed binary itself carries no commit), ``pymimir_version``, ``python``, ``seed``.
``task``
    ``id``, ``slug``, ``covers`` (what the task exercises), ``domain`` and ``problem`` (paths relative to the
    repository root) and ``objects`` (names in mifrost's order: sorted by name, constants included).
``static_atoms``
    Sorted names of the task's static atoms, as ``"(on a b)"``.
``states``
    Records ``{"id": "s0", "path": [...], "fluent": [...], "derived": [...]}``. ``path`` is the action names from the
    initial state (``[]`` for ``s0``); mymyr replays them by name with ``Task.apply``. ``fluent`` and ``derived`` are
    the sorted names of the state's fluent and derived atoms. ``s1`` to ``s4`` end self-avoiding seeded random walks
    of up to 2, 4, 6 and 8 steps; the five states are pairwise different where the task allows.
``catalogs``
    A pool, key to catalog. A flat catalog lists the relations in mifrost's order: ``names``, ``arities`` (encoded
    arity, slots included), ``logical_arities``, ``sources`` (``state``, ``action``, ``goal``, ``goal_derivation``,
    ``history``, ...), ``slots`` (per relation the slot role ids of its positions, indexing ``slot_role_names``) and
    ``slot_role_names``. An HGraph catalog lists ``node_types`` and ``edge_types``. A catalog is domain-scoped, so it
    is the same for every state of the task.
``configs``
    ``{config_id: {"encoder": class name, "options": constructor keywords (enum values by name), "catalog": key}}``.
``encodings``
    One record per (config, state), in config order and then state order: ``config``, ``state`` (a state id),
    ``inputs`` and ``outputs``.
``tuples``
    A pool, key to a list of tuples (see ``relations`` below).

``inputs`` holds only what the encoder was given, always as literal strings; a key is absent when not passed.
    ``goals``
        Literals replacing the problem's goal: ``"(on c b)"`` or ``"(not (on c b))"``.
    ``actions``
        Ground actions given to the action lane, as ``"(unstack a b)"``: the schema name and every object of the ground
        action. The objects include the parameters the normalizer adds, e.g. the existential witness of
        ``(step n2 n3 n2)``, and the action entities and action relations of mifrost use the same form.
    ``subgoal_layers``
        A list of literal lists, one per level.
    ``history_subgoals``
        A list of ``[dt, [literal, ...]]`` steps, oldest first; ``history_max_steps`` is an integer. mifrost keeps
        the newest steps.
    ``dag``
        Horizon inputs: ``nodes`` (action-name paths from the initial state; node 0 is the root) and ``edges``
        (``[parent, child, action]``).
    ``successor``
        Transition inputs: ``{"path": [...], "action": "(...)"}`` for the successor of the current state.

``outputs`` is the decoded encoding of one graph, by name (all of it absent if ``error`` is present):
    ``entities``
        ``by_role``: role name to the entity names in entity order; ``runs``: ``[role, count]`` pairs that give the
        role of each entity id in order, so that the id-ordered names are rebuilt by taking, run by run, the next
        names of that role. Roles are ``object``, ``predicate_virtual``, ``goal``, ``subgoal``, ``action``,
        ``history``, ``history_target`` and ``state``.
    ``relations``
        Relation name to ``{"emit": key, "sorted": key}``; relations without tuples are absent, and ``sorted`` is
        absent when it equals ``emit``. A key indexes the file-level ``tuples`` pool, which holds each distinct
        list once. A tuple is a list of entity names; ``emit`` keeps mifrost's emission order and ``sorted`` is the
        same list in lexicographic order of the names. The slot roles of each position are in the catalog.
    ``targets``
        ``groups``: group names, indexed by group id; ``names``: mifrost's target names; ``rows``: one
        ``[entity id, group id, candidate id]`` per target; ``entities``: ``[entity id, mifrost group code]`` per
        distinct target entity (the code is mifrost's own numbering of target kinds). Absent without targets.
    ``history``
        One ``[entity id, dt]`` per history step entity.
    ``graph_arrays``
        The remaining ``__graph__/...`` arrays, verbatim in mifrost ids (``node_sizes``, ``object_indices``, ...).
    ``hgraph``
        HGraph only (instead of ``entities``, ``relations`` and ``history``): ``nodes`` (node type to node names)
        and ``edges`` (edge type ``"src|position|dst"`` to ``[src name, dst name]`` pairs in mifrost order).
    ``error``
        The message, if mifrost rejected the inputs.

An encoder test reads a fixture like this: load the task from ``task.domain`` and ``task.problem``, replay each
state's ``path`` and compare the reached state's atoms with ``fluent`` and ``derived``, build the encoding for a
record from its ``config`` and ``inputs`` and compare the outcome with ``outputs``. Entity ids and tuple order are
mifrost's and differ from mymyr's by design; compare tuples as multisets of names (``sorted``) and the catalog as an
ordered list.

Configurations
--------------
C1 default flat; C2 all goal derivations (plain, satisfied, unsatisfied), nullary relations kept, relation-major packing;
C3 action and goal targets with predicate virtual nodes and explicit actions (goal-conditioned on odd states);
C4_L1 and C4_L2 subgoal layers at levels 1 and 2 (layer one is ``[sg]``, layer two ``[ssg]``); C5A history with
targets; C5B history without targets and ``history_max_steps``; C7A to C7F horizon encoders (full/include,
delta/exclude, delta/encode_only with parent relation and actions, action/exclude, the rooted encoder with defaults and
with sibling and cousin relations); C8A and C8B the transition and transition-effects encoders; C9A to C9C HGraph
(default, nullary predicates, actions). Horizon, transition and HGraph configs encode two states per task. The id
C6 is unused.

Every option the configurations use exists in mifrost 0.6.0. Behaviours that shaped the selection of configurations
are listed in ``FINDINGS`` and printed by ``--introspect``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "tests/data/encoding/mifrost"
FORMAT = 1
SEED = 20261011
MIFROST_VERSION = "0.6.0"
PYMIMIR_VERSION = "0.16.3"
MIFROST_SOURCE_CHECKOUT = "91b152c21819678a7818865b230774a3b6e800e3"
WALK_LENGTHS = (2, 4, 6, 8)
MAX_ACTIONS = 6
MAX_CHILDREN = 3
RAISE = False

ROLE_NAMES = (
    "object",
    "predicate_virtual",
    "goal",
    "subgoal",
    "action",
    "history",
    "history_target",
    "state",
)

# Behaviours of mifrost 0.6.0 that shape the selection of configurations; `--introspect` prints them.
#  * Every option the configurations use exists; none had to be left out.
#  * HGraphEncoder(add_nullary_predicates=True) rejects a nullary goal literal ("Node feature dim mismatch"), so C9B
#    only runs on tasks whose goals have no nullary literal.
#  * Packing relation arguments relation-major only differs from graph-major for batches; single-graph fixtures
#    record it as an option (C2) but cannot observe it.
FINDINGS = {
    "missing options": "none",
    "HGraphEncoder nullary predicates": "C9B only on tasks without nullary goal literals",
    "relation-major packing": "recorded as an option of C2; not observable on one graph",
}


def pddl(task_dir, problem):
    base = f"tests/data/pddl/{task_dir}"
    return f"{base}/domain.pddl", f"{base}/{problem}.pddl"


def ipc(name, problem):
    base = f"tests/data/ipc/{name}-ipc/test"
    return f"{base}/domain.pddl", f"{base}/{problem}.pddl"


FLAT = ("C1", "C2", "C3", "C4_L1", "C4_L2", "C5A", "C5B")
HORIZON = ("C7A", "C7B", "C7C", "C7D", "C7E", "C7F", "C8A", "C8B")
HGRAPH = ("C9A", "C9B", "C9C")

TASKS = [
    dict(id="F1", slug="blocks_smedium", covers="blocks, three objects, nullary handempty",
         files=pddl("blocks", "smedium"), configs=FLAT + HORIZON + HGRAPH),
    dict(id="F2", slug="blocks8", covers="blocks, eight objects", files=pddl("blocks", "probBLOCKS-8-0"),
         configs=("C1", "C2", "C3", "C4_L1", "C5A", "C5B", "C7B", "C8A", "C9A", "C9B", "C9C")),
    dict(id="F3", slug="logistics6", covers="typed domain, several action schemas",
         files=pddl("logistics00", "probLOGISTICS-6-1"), configs=("C1", "C2", "C3", "C4_L1", "C5A", "C7A", "C8B")),
    dict(id="F4", slug="childsnack_p01_easy", covers="constants, many statics",
         files=ipc("childsnack", "p01-easy"), configs=("C1", "C2", "C3", "C4_L1")),
    dict(id="F5", slug="philosophers_p03", covers="derived predicates, =, auxiliary axioms",
         files=pddl("philosophers", "p03-phil4"), configs=("C1", "C2", "C3", "C4_L1", "C7B", "C8B")),
    dict(id="F6", slug="rovers_p01_easy", covers="many statics", files=ipc("rovers", "p01-easy"),
         configs=("C1", "C2", "C3")),
    dict(id="F7", slug="satellite_p01_easy", covers="typed domain", files=ipc("satellite", "p01-easy"),
         configs=("C1", "C2", "C3", "C4_L1", "C7D")),
    dict(id="F8", slug="features", covers="negative goals, derived predicate, nullary predicate and action",
         files=pddl("features", "p01"), configs=FLAT + HORIZON + ("C9A", "C9C")),
    dict(id="F9", slug="relay", covers="existential preconditions", files=pddl("relay", "p01"),
         configs=FLAT + ("C7A", "C7C", "C8A", "C9A", "C9C")),
    dict(id="F10", slug="blocks_flags", covers="blocks with nullary fluents and goals",
         files=pddl("blocks_flags", "p01"), configs=FLAT + ("C7C", "C8B", "C9A")),
]

HGRAPH_STATES = ("s0", "s3")

GOAL_PLAIN_SAT_UNSAT = ["plain", "satisfied", "unsatisfied"]
GOAL_ADDED = ["added_satisfied", "added_unsatisfied"]

# config id -> (encoder class name, options, kind). `kind` selects the call shape in `encode_record`.
CONFIGS = {
    "C1": ("FlatRelationEncoder", {}, "flat"),
    "C2": ("FlatRelationEncoder", dict(goal_derivations=GOAL_PLAIN_SAT_UNSAT, ignore_zero_arity_relations=False,
                                        pack_relation_args_relation_major=True), "flat"),
    "C3": ("FlatRelationEncoder", dict(use_predicate_virtual_nodes=True, target_sources=["action", "goal"]), "flat"),
    "C4_L1": ("FlatRelationEncoder", dict(max_goal_level=1, target_sources=["goal", "subgoal"],
                                           goal_derivations=GOAL_PLAIN_SAT_UNSAT), "flat"),
    "C4_L2": ("FlatRelationEncoder", dict(max_goal_level=2, target_sources=["goal", "subgoal"],
                                           goal_derivations=GOAL_PLAIN_SAT_UNSAT), "flat"),
    "C5A": ("FlatRelationEncoder", dict(target_sources=["history"]), "flat"),
    "C5B": ("FlatRelationEncoder", {}, "flat"),
    "C7A": ("FlatHorizonEncoder", dict(transition_mode="full", root_policy="include", ignore_actions=False),
            "horizon"),
    "C7B": ("FlatHorizonEncoder", dict(transition_mode="delta", root_policy="exclude", ignore_actions=True,
                                        ignore_zero_arity_relations=False, goal_derivations=GOAL_ADDED), "horizon"),
    "C7C": ("FlatHorizonEncoder", dict(transition_mode="delta", root_policy="encode_only", ignore_actions=False,
                                        ignore_zero_arity_relations=False, enable_parent_relation=True,
                                        goal_derivations=GOAL_ADDED), "horizon"),
    "C7D": ("FlatHorizonEncoder", dict(transition_mode="action", root_policy="exclude", ignore_actions=False),
            "horizon"),
    "C7E": ("FlatRootedHorizonEncoder", {}, "horizon"),
    "C7F": ("FlatRootedHorizonEncoder", dict(enable_sibling_relation=True, enable_cousin_relation=True), "horizon"),
    "C8A": ("FlatTransitionEncoder", {}, "transition"),
    "C8B": ("FlatTransitionEffectsEncoder", {}, "transition"),
    "C9A": ("HGraphEncoder", {}, "hgraph"),
    "C9B": ("HGraphEncoder", dict(add_nullary_predicates=True), "hgraph"),
    "C9C": ("HGraphEncoder", dict(ignore_actions=False, target_sources=["action"]), "hgraph"),
}


# ---------------------------------------------------------------------------------------------------------------------
# pymimir side: tasks, states, literals


def action_name(action):
    """``(schema o1 ... on)`` over all objects of the ground action, as mymyr names it (``(switch)`` if none)."""
    objects = " ".join(o.get_name() for o in action.get_objects())
    return f"({action.get_action().get_name()} {objects})" if objects else f"({action.get_action().get_name()})"


def literal_name(literal):
    atom = str(literal.get_atom())
    return atom if literal.get_polarity() else f"(not {atom})"


class Task:
    def __init__(self, entry):
        import pymimir

        self.entry = entry
        domain_path, problem_path = entry["files"]
        self.domain = pymimir.Domain(str(ROOT / domain_path))
        self.problem = pymimir.Problem(self.domain, str(ROOT / problem_path), mode="lifted")
        self.initial = self.problem.get_initial_state()
        objects = list(self.domain.get_constants()) + list(self.problem.get_objects())
        self.objects = {obj.get_name(): obj for obj in objects}
        self.universe = []
        self.predicates = {p.get_name(): p for p in self.domain.get_predicates()}
        self.goal = list(self.problem.get_goal_condition())

    def applicable(self, state):
        actions = list(state.generate_applicable_actions())
        return sorted(actions, key=action_name)

    def step(self, state, action):
        result = action.apply(state)
        return result[0] if isinstance(result, tuple) else result

    def atoms(self, state, kind):
        pick = {"static": "is_static", "fluent": "is_fluent", "derived": "is_derived"}[kind]
        return sorted({str(a) for a in state.get_atoms() if getattr(a, pick)()})

    def literal(self, text):
        """The ground literal for ``(p a b)`` or ``(not (p a b))``."""
        positive = not text.startswith("(not ")
        body = text if positive else text[5:-1]
        parts = body[1:-1].split()
        atom = self.problem.new_ground_atom(self.predicates[parts[0]], [self.objects[o] for o in parts[1:]])
        return self.problem.new_ground_literal(atom, positive)

    def literals(self, texts):
        return [self.literal(t) for t in texts]


def random_walk(task, rng, length):
    """A self-avoiding random walk of at most ``length`` steps from the initial state."""
    state, path = task.initial, []
    visited = {state_key(state)}
    for _ in range(length):
        options = [
            (a, n) for a in task.applicable(state) for n in [task.step(state, a)] if state_key(n) not in visited
        ]
        if not options:
            break
        action, state = options[rng.randrange(len(options))]
        visited.add(state_key(state))
        path.append(action_name(action))
    return state, path


def build_states(task):
    """The initial state and one self-avoiding walk per length, each ending in a state not chosen before."""
    rng = random.Random(f"{SEED}/{task.entry['id']}")
    states = [dict(id="s0", path=[], obj=task.initial)]
    chosen = {state_key(task.initial)}
    for index, length in enumerate(WALK_LENGTHS, start=1):
        for _ in range(32):
            state, path = random_walk(task, rng, length)
            if state_key(state) not in chosen:
                break
        chosen.add(state_key(state))
        states.append(dict(id=f"s{index}", path=path, obj=state))
    task.universe = sorted({a for s in states for a in task.atoms(s["obj"], "fluent") if " " in a})
    return states


def state_literals(task, state):
    """Literals to draw subgoals and history from: the goal literals, the state's non-nullary fluent atoms and one
    atom that holds in another recorded state but not in this one (as a negative literal)."""
    goal = [literal_name(l) for l in task.goal]
    fluent = [a for a in task.atoms(state, "fluent") if " " in a]
    absent = next((f"(not {a})" for a in task.universe if a not in set(task.atoms(state, "fluent"))), None)
    return goal, fluent, absent


def pick(items, count, rng):
    items = list(items)
    if len(items) <= count:
        return items
    return sorted(rng.sample(items, count))


def sampled_actions(task, state, rng):
    actions = task.applicable(state)
    return [actions[i] for i in sorted(rng.sample(range(len(actions)), min(MAX_ACTIONS, len(actions))))]


def payloads(task, state_record, config_id, kind):
    """The optional inputs for (config, state): both the literal strings and the pymimir objects."""
    state = state_record["obj"]
    rng = random.Random(f"{SEED}/{task.entry['id']}/{config_id}/{state_record['id']}")
    goal, fluent, absent = state_literals(task, state)
    texts, objs = {}, {}
    if config_id == "C3":
        actions = sampled_actions(task, state, rng)
        texts["actions"] = [action_name(a) for a in actions]
        objs["actions"] = actions
        if int(state_record["id"][1:]) % 2 == 1:
            texts["goals"] = pick(fluent, 3, rng) + ([absent] if absent else [])
    elif config_id in ("C4_L1", "C4_L2"):
        first = pick(goal, 1, rng) + pick([f for f in fluent if f not in goal], 1, rng)
        layers = [first]
        if config_id == "C4_L2":
            second = pick([g for g in goal if g not in first], 1, rng) + ([absent] if absent else [])
            layers.append(second)
        texts["subgoal_layers"] = layers
    elif config_id in ("C5A", "C5B"):
        steps = [
            [-3, pick(fluent, 2, rng)],
            [-2, pick(goal, 1, rng) + pick(fluent, 1, rng)],
            [-1, ([absent] if absent else []) + pick(fluent, 2, rng)],
        ]
        if config_id == "C5A":
            steps = steps[1:]
        else:
            texts["history_max_steps"] = 2
        texts["history_subgoals"] = steps
    elif kind == "hgraph" and config_id == "C9C":
        actions = sampled_actions(task, state, rng)
        texts["actions"] = [action_name(a) for a in actions]
        objs["actions"] = actions
    return texts, objs


def horizon_states(task, states):
    """The initial state and the first later state with at least three applicable actions (else the third)."""
    later = [s for s in states[1:] if len(task.applicable(s["obj"])) >= 3]
    return [states[0], later[0] if later else states[2]]


def horizon_dag(task, state_record, rng):
    """A root with up to MAX_CHILDREN successors and one grandchild under each of the first two."""
    root, root_path = state_record["obj"], state_record["path"]
    nodes = [dict(path=root_path, obj=root)]
    edges = []
    seen = {state_key(root)}
    actions = task.applicable(root)
    for index in sorted(rng.sample(range(len(actions)), min(MAX_CHILDREN, len(actions)))):
        child = task.step(root, actions[index])
        key = state_key(child)
        if key in seen:
            continue
        seen.add(key)
        nodes.append(dict(path=root_path + [action_name(actions[index])], obj=child))
        edges.append([0, len(nodes) - 1, actions[index]])
    for parent_index in range(1, min(len(nodes), 3)):
        parent = nodes[parent_index]
        for action in task.applicable(parent["obj"]):
            grand = task.step(parent["obj"], action)
            key = state_key(grand)
            if key in seen:
                continue
            seen.add(key)
            nodes.append(dict(path=parent["path"] + [action_name(action)], obj=grand))
            edges.append([parent_index, len(nodes) - 1, action])
            break
    return nodes, edges


def state_key(state):
    return tuple(sorted(str(a) for a in state.get_atoms() if a.is_fluent() or a.is_derived()))


# ---------------------------------------------------------------------------------------------------------------------
# mifrost side


class CapsuleArray:
    def __init__(self, capsule):
        self.capsule = capsule

    def __dlpack__(self, **kwargs):
        return self.capsule

    def __dlpack_device__(self):
        return 1, 0


def to_numpy(payload):
    if hasattr(payload, "__dlpack__"):
        return np.from_dlpack(payload)
    return np.from_dlpack(CapsuleArray(payload))


def resolve_options(mifrost, options):
    resolved = dict(options)
    if "goal_derivations" in resolved:
        resolved["goal_derivations"] = {getattr(mifrost.GoalDerivation, name) for name in resolved["goal_derivations"]}
    return resolved


def make_encoder(mifrost, domain, config_id):
    name, options, _ = CONFIGS[config_id]
    return getattr(mifrost, name)(domain, **resolve_options(mifrost, options))


def catalog_of(encoder, kind):
    if kind == "hgraph":
        return {}
    roles = encoder.relation_slot_roles
    offsets = encoder.relation_slot_role_offsets
    role_names = list(encoder.slot_role_names)
    slots = [[int(r) for r in roles[offsets[i] : offsets[i + 1]]] for i in range(len(encoder.relation_names))]
    return dict(
        names=list(encoder.relation_names),
        arities=[int(a) for a in encoder.relation_arities],
        logical_arities=[int(a) for a in encoder.relation_logical_arities],
        sources=list(encoder.relation_sources),
        slots=slots,
        slot_role_names=role_names,
    )


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [jsonable(v) for v in value]
        return sorted(items, key=json.dumps) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class Pool:
    """Distinct tuple lists, keyed by content."""

    def __init__(self):
        self.items = {}

    def add(self, tuples):
        text = json.dumps(tuples, separators=(",", ":"))
        key = hashlib.sha1(text.encode()).hexdigest()[:10]
        self.items[key] = tuples
        return key


def graph_arrays(tensors):
    """The ``__graph__`` arrays that the decoded blocks do not already carry."""
    skip = ("relation_", "target_", "history_", "entity_role_ids")
    return {
        key[len("__graph__/") :]: value.reshape(-1).tolist()
        for key, value in sorted(tensors.items())
        if key.startswith("__graph__/") and not key[len("__graph__/") :].startswith(skip)
    }


def entity_block(names, tensors):
    """Names by role and the role of each entity as runs ``[role, count]`` in entity-id order."""
    roles = [ROLE_NAMES[int(r)] for r in tensors["__graph__/entity_role_ids"].reshape(-1)]
    by_role, runs = {}, []
    for name, role in zip(names, roles):
        by_role.setdefault(role, []).append(name)
        if runs and runs[-1][0] == role:
            runs[-1][1] += 1
        else:
            runs.append([role, 1])
    return dict(by_role=by_role, runs=runs)


def target_block(tensors, graph_attrs, entity_names):
    if "__graph__/target_positions" not in tensors:
        return None

    def ints(key):
        return tensors[key].reshape(-1).tolist() if key in tensors else []

    names = list(graph_attrs.get("target_names", []))
    if any(n.startswith("State(") for n in names):
        names = [entity_names[e] for e in ints("__graph__/target_positions")]
    return dict(
        groups=list(graph_attrs.get("target_groups", [])),
        names=names,
        rows=[list(r) for r in zip(ints("__graph__/target_positions"), ints("__graph__/target_group_ids"),
                                    ints("__graph__/target_candidate_ids"))],
        entities=[list(r) for r in zip(ints("__graph__/target_entity_indices"),
                                        ints("__graph__/target_entity_group_ids"))],
    )


def history_block(tensors, names):
    if "__graph__/history_entity_indices" not in tensors:
        return []
    indices = tensors["__graph__/history_entity_indices"].reshape(-1).tolist()
    dts = tensors["__graph__/history_entity_dt"].reshape(-1).tolist()
    return [[i, dt] for i, dt in zip(indices, dts)]


def decode_flat(encoder, encoding, pool):
    data = encoding.as_dict()
    tensors = {k: to_numpy(v) for k, v in data["tensors"].items()}
    names = list(data["node_names"]["entity"])
    counts = tensors["__graph__/relation_counts"].reshape(-1)
    args = tensors["__graph__/relation_args"].reshape(-1)
    relations = {}
    offset = 0
    for name, arity, count in zip(encoder.relation_names, encoder.relation_arities, counts):
        arity, count = int(arity), int(count)
        if count:
            rows = args[offset : offset + count * arity].reshape(count, arity).tolist()
            tuples = [[names[i] for i in row] for row in rows]
            emit, ordered = pool.add(tuples), pool.add(sorted(tuples))
            relations[name] = dict(emit=emit) if emit == ordered else dict(emit=emit, sorted=ordered)
        offset += count * arity
    assert offset == len(args), "relation_args longer than the relation counts account for"
    graph_attrs = data.get("graph_attrs") or {}
    out = dict(entities=entity_block(names, tensors), relations=relations,
               history=history_block(tensors, names), graph_arrays=graph_arrays(tensors))
    targets = target_block(tensors, graph_attrs, names)
    if targets is not None:
        out["targets"] = targets
    return out


def decode_hgraph(encoder, encoding, pool):
    data = encoding.as_dict()
    tensors = {k: to_numpy(v) for k, v in data["tensors"].items()}
    node_names = {k: list(v) for k, v in data["node_names"].items()}
    nodes = {}
    for node_type in data["schema"]["node_types"]:
        ptr = tensors.get(f"{node_type}/ptr")
        count = int(ptr.reshape(-1)[-1]) if ptr is not None else 0
        names = node_names.get(node_type, [])
        if count or names:
            nodes[node_type] = names if names else [None] * count
    edges = {}
    for key in sorted(tensors):
        if not key.endswith("/edge_index_0"):
            continue
        edge_type = key[: -len("/edge_index_0")]
        src_type, _, dst_type = edge_type.split("|")
        src = tensors[key].reshape(-1).tolist()
        dst = tensors[edge_type + "/edge_index_1"].reshape(-1).tolist()
        if not src:
            continue
        edges[edge_type] = [[node_names[src_type][i], node_names[dst_type][j]] for i, j in zip(src, dst)]
    graph_attrs = data.get("graph_attrs") or {}
    out = dict(hgraph=dict(nodes=nodes, edges=edges), graph_arrays=graph_arrays(tensors))
    entity_names = node_names.get("_symbol_", [])
    targets = None
    if "__graph__/target_positions" in tensors and entity_names:
        targets = target_block(tensors, graph_attrs, entity_names)
    if targets is not None:
        out["targets"] = targets
    out["_schema"] = dict(node_types=list(data["schema"]["node_types"]), edge_types=jsonable(data["schema"]["edge_types"]))
    return out


def encode_record(task, encoder, kind, state_record, texts, objs, extra, pool):
    state = state_record["obj"]
    kwargs = {}
    if "goals" in texts:
        kwargs["goals"] = task.literals(texts["goals"])
    if "actions" in texts:
        kwargs["actions"] = objs["actions"]
    if "subgoal_layers" in texts:
        kwargs["subgoal_layers"] = [task.literals(layer) for layer in texts["subgoal_layers"]]
    if "history_subgoals" in texts:
        kwargs["history_subgoals"] = [(dt, task.literals(lits)) for dt, lits in texts["history_subgoals"]]
    if "history_max_steps" in texts:
        kwargs["history_max_steps"] = texts["history_max_steps"]
    if kind == "horizon":
        import rustworkx as rx

        dag = rx.PyDiGraph()
        for node in extra["nodes"]:
            dag.add_node(node["obj"])
        for parent, child, action in extra["edges"]:
            dag.add_edge(parent, child, action)
        encoding = encoder.encode(extra["nodes"][0]["obj"], dag=dag, **kwargs)
    elif kind == "transition":
        encoding = encoder.encode(state, successor=extra["successor"], **kwargs)
    else:
        encoding = encoder.encode(state, **kwargs)
    if kind == "hgraph":
        return decode_hgraph(encoder, encoding, pool)
    return decode_flat(encoder, encoding, pool)


def export_task(mifrost, entry):
    task = Task(entry)
    states = build_states(task)
    pool = Pool()
    configs = {}
    catalogs = Pool()
    encodings = []
    for config_id in entry["configs"]:
        name, options, kind = CONFIGS[config_id]
        encoder = make_encoder(mifrost, task.domain, config_id)
        record = dict(encoder=name, options=jsonable(options))
        if kind == "hgraph":
            record["catalog"] = None
        else:
            record["catalog"] = catalogs.add(catalog_of(encoder, kind))
        configs[config_id] = record
        selected = states
        if kind in ("horizon", "transition"):
            selected = horizon_states(task, states)
        if kind == "hgraph":
            selected = [s for s in states if s["id"] in HGRAPH_STATES]
        for state_record in selected:
            texts, objs = payloads(task, state_record, config_id, kind)
            inputs = dict(texts)
            extra = None
            rng = random.Random(f"{SEED}/{entry['id']}/{config_id}/{state_record['id']}/dag")
            if kind == "horizon":
                nodes, edges = horizon_dag(task, state_record, rng)
                extra = dict(nodes=nodes, edges=edges)
                inputs["dag"] = dict(nodes=[n["path"] for n in nodes],
                                     edges=[[p, c, action_name(a)] for p, c, a in edges])
            elif kind == "transition":
                nodes, edges = horizon_dag(task, state_record, rng)
                if not edges:
                    continue
                parent, child, action = edges[0]
                extra = dict(successor=nodes[child]["obj"])
                inputs["successor"] = dict(path=nodes[child]["path"], action=action_name(action))
            try:
                outputs = encode_record(task, encoder, kind, state_record, texts, objs, extra, pool)
                schema = outputs.pop("_schema", None)
                if schema is not None and configs[config_id]["catalog"] is None:
                    configs[config_id]["catalog"] = catalogs.add(schema)
            except Exception as error:  # noqa: BLE001 - recorded as a finding
                if RAISE:
                    raise
                outputs = dict(error=f"{type(error).__name__}: {error}")
            encodings.append(dict(config=config_id, state=state_record["id"], inputs=inputs, outputs=outputs))
    state_records = [
        dict(id=s["id"], path=s["path"], fluent=task.atoms(s["obj"], "fluent"), derived=task.atoms(s["obj"], "derived"))
        for s in states
    ]
    return dict(
        format=FORMAT,
        provenance=dict(
            producer="mifrost",
            mifrost_version=importlib.metadata.version("mifrost"),
            mifrost_source_checkout=MIFROST_SOURCE_CHECKOUT,
            pymimir_version=importlib.metadata.version("pymimir"),
            python=platform.python_version(),
            seed=SEED,
        ),
        task=dict(id=entry["id"], slug=entry["slug"], covers=entry["covers"], domain=entry["files"][0],
                  problem=entry["files"][1], objects=sorted(task.objects)),
        static_atoms=task.atoms(task.initial, "static"),
        states=state_records,
        catalogs=catalogs.items,
        configs=configs,
        encodings=encodings,
        tuples=pool.items,
    )


# ---------------------------------------------------------------------------------------------------------------------
# output


def write_fixture(path, fixture):
    """One top-level key per section; one record per line in the long lists, keys sorted, no trailing spaces."""

    def dump(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    lines = ["{"]
    keys = list(fixture)
    for index, key in enumerate(keys):
        value = fixture[key]
        comma = "," if index + 1 < len(keys) else ""
        if key == "encodings":
            lines.append(f'"{key}":[')
            lines.extend(dump(item) + ("," if i + 1 < len(value) else "") for i, item in enumerate(value))
            lines.append(f"]{comma}")
        elif key in ("tuples", "catalogs"):
            lines.append(f'"{key}":{{')
            items = sorted(value.items())
            lines.extend(f'"{k}":{dump(v)}' + ("," if i + 1 < len(items) else "") for i, (k, v) in enumerate(items))
            lines.append(f"}}{comma}")
        else:
            lines.append(f'"{key}":{dump(value)}{comma}')
    lines.append("}")
    path.write_text("\n".join(lines) + "\n")


def introspect(mifrost):
    import inspect

    print("mifrost", importlib.metadata.version("mifrost"), "pymimir", importlib.metadata.version("pymimir"))
    print("GoalDerivation:", [m for m in dir(mifrost.GoalDerivation) if not m.startswith("_")])
    for name in sorted({c[0] for c in CONFIGS.values()}):
        parameters = list(inspect.signature(getattr(mifrost, name).__init__).parameters)
        print(name, parameters[2:])
    print("findings:", json.dumps(FINDINGS, indent=1))


def print_small_goals_fixture():
    """Print the signed-goal fixture of mifrost's neutral engine (no planning backend)."""
    from mifrost import _neutral_core as core

    category = core.SemanticPredicateCategory
    predicates = [
        core.SemanticPredicateSpec(category.fluent, "flag", 1),
        core.SemanticPredicateSpec(getattr(category, "static"), "fixed", 1),
        core.SemanticPredicateSpec(category.derived, "available", 1),
        core.SemanticPredicateSpec(category.fluent, "ready", 0),
    ]
    config = core.FlatRelationEncoderConfig(
        support_literals=True,
        ignore_zero_arity_relations=False,
        goal_derivations={core.GoalDerivation.satisfied, core.GoalDerivation.unsatisfied},
    )
    engine = core.SemanticFlatRelationEncoderEngine(predicates, [], config)
    value = core.SemanticFlatRelationInput()
    value.objects = ["a", "b"]
    value.state_facts = [core.SemanticAtom(predicate, [0]) for predicate in range(3)]
    value.goals = [
        core.SemanticLiteral(core.SemanticAtom(predicate, objects), positive)
        for predicate, objects, positive in [(0, [1], True), (0, [0], False), (1, [0], True),
                                              (1, [1], False), (2, [1], True), (2, [0], False), (3, [], True)]
    ]
    batch = engine.encode(value)
    tensors = {
        key: to_numpy(payload)
        for key, payload in batch.as_dict()["tensors"].items()
        if key in {"__graph__/relation_args", "__graph__/relation_counts"}
    }
    counts = tensors["__graph__/relation_counts"][0]
    packed = tensors["__graph__/relation_args"]
    output = {}
    offset = 0
    for name, arity, count in zip(engine.relation_names, batch.graph_attrs["relation_arities"], counts):
        count, arity = int(count), int(arity)
        if count:
            output[name] = packed[offset : offset + count * arity].reshape(count, arity).tolist()
        offset += count * arity
    print(json.dumps(output, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", help="comma-separated task ids, e.g. F1,F5")
    parser.add_argument("--raise-errors", action="store_true", help="stop at the first mifrost error")
    parser.add_argument("--small-goals", action="store_true", help="print the small signed-goal fixture and exit")
    parser.add_argument("--introspect", action="store_true", help="print the encoder options of the installed mifrost")
    arguments = parser.parse_args()
    global RAISE
    RAISE = arguments.raise_errors
    import mifrost

    if importlib.metadata.version("mifrost") != MIFROST_VERSION:
        sys.exit(f"mifrost {MIFROST_VERSION} is required, found {importlib.metadata.version('mifrost')}")
    if importlib.metadata.version("pymimir") != PYMIMIR_VERSION:
        sys.exit(f"pymimir {PYMIMIR_VERSION} is required, found {importlib.metadata.version('pymimir')}")
    if arguments.small_goals:
        print_small_goals_fixture()
        return
    if arguments.introspect:
        introspect(mifrost)
        return
    wanted = set(arguments.only.split(",")) if arguments.only else None
    arguments.out.mkdir(parents=True, exist_ok=True)
    for entry in TASKS:
        if wanted and entry["id"] not in wanted:
            continue
        fixture = export_task(mifrost, entry)
        target = arguments.out / f"f{int(entry['id'][1:]):02d}_{entry['slug']}.json"
        write_fixture(target, fixture)
        print(target.name, target.stat().st_size)


if __name__ == "__main__":
    main()
