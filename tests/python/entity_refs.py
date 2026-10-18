"""Reads the goal and action entities of an encoded graph through their ``entity_ref`` and checks their names.

The ref of a goal entity is the number of its tuple among the graph's tuples in graph-major order (relation after
relation, in catalog order); the ref of an action entity is the row of its action among the graph's rows of the
expansion given as ``actions``. Nothing here reads the target slot of a tuple to find an entity's name, so a name
assigned to the wrong entity is caught.
"""

GOAL_ROLE, ACTION_ROLE = 2, 4
ARGUMENT_SLOT, GOAL_TARGET_SLOT = 0, 4


def graph_tuples(encoder, batch, graph):
    """The tuples of one graph in graph-major order: a list of (relation id, entity ids of the tuple), the ids being
    batch-global."""
    arrays = batch.as_dict()
    graphs, relations = arrays["num_graphs"], arrays["num_relations"]
    counts = arrays["relation_counts"].reshape(graphs, relations)
    by_relation = encoder.options["layout"] == "relation"
    out = []
    for relation in range(relations):
        segment = relation * graphs + graph if by_relation else graph * relations + relation
        first = int(arrays["relation_arg_offsets"][segment])
        arity = encoder.relation_arities[relation]
        for number in range(int(counts[graph, relation])):
            out.append((relation, arrays["relation_args"][first + number * arity : first + (number + 1) * arity]))
    return out


def goal_literal_name(encoder, batch, graph, ref, objects):
    """The name of the goal entity whose ref is `ref`: the literal of the graph's tuple number `ref`, with its
    arguments' objects named by `objects` (the objects of the graph's instance)."""
    relation, entities = graph_tuples(encoder, batch, graph)[ref]
    sign_and_predicate = encoder.relation_names[relation]
    assert sign_and_predicate[0] == "[" and sign_and_predicate.endswith("[g]"), sign_and_predicate
    sign, predicate = sign_and_predicate[1], sign_and_predicate[3:-3]
    roles = encoder.relation_slot_roles[relation]
    refs = batch.entity_ref
    arguments = [objects[int(refs[int(e)])] for e, role in zip(entities, roles) if role == ARGUMENT_SLOT]
    return f"[{sign}]({' '.join([predicate, *arguments])})[g]"


def check_entity_names(encoder, batch, graph, objects, actions=None, state_row=0):
    """Checks the name of every goal and action entity of `graph` against its ref, and that the goal entity is the
    target of the tuple its ref names. `actions` is the expansion the graph's state was given with and `state_row` the
    state's row in it. Returns the names that were checked, by role."""
    names = encoder.entity_names(batch, graph)
    begin, end = (int(x) for x in batch.entity_offsets[graph : graph + 2])
    roles, refs = batch.entity_role[begin:end].tolist(), batch.entity_ref[begin:end].tolist()
    assert len(names) == end - begin
    checked = {"goal": [], "action": []}
    tuples = graph_tuples(encoder, batch, graph)
    for local, (role, ref) in enumerate(zip(roles, refs)):
        if role == GOAL_ROLE:
            relation, entities = tuples[ref]
            target = encoder.relation_slot_roles[relation].index(GOAL_TARGET_SLOT)
            assert int(entities[target]) == begin + local, (graph, local, ref)
            assert names[local] == goal_literal_name(encoder, batch, graph, ref, objects), (graph, local)
            checked["goal"].append(names[local])
        elif role == ACTION_ROLE:
            assert actions is not None
            assert names[local] == str(actions.action(int(actions.offsets[state_row]) + ref)), (graph, local)
            checked["action"].append(names[local])
    return checked
