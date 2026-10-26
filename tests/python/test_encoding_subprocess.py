"""Corpus encodings run in child processes so a native fault is reported for its task."""

import itertools
import json
from pathlib import Path

import pytest

import mymyr
from mymyr import encoding, rl
from conftest import ROOT
from python_children import run_code


TASKS = sorted((ROOT / "tests/data/numeric_tasks").rglob("*.txt")) + sorted(
    path for path in (ROOT / "tests/data/pddl").rglob("*.pddl") if path.name != "domain.pddl"
)


def encode_task(path):
    """Encode the initial state with both graph families, layouts and optional literal layers."""
    if path.suffix == ".txt":
        task = mymyr.Task.from_text(path)
    else:
        task = mymyr.Task.from_pddl(path.with_name("domain.pddl"), path)
    table = rl.as_table(task)
    encoded = rejected = 0
    for outputs, layout, level, subgoals, history in itertools.product(
        (("relations",), ("relations", "simple")), ("graph", "relation"),
        (None, 3), (None, "empty", "layer"), (None, "empty", "step"),
    ):
        if level is None and subgoals == "layer":
            continue
        options = {"layout": layout, "outputs": outputs}
        if level is not None:
            options["max_goal_level"] = level
        if history is not None:
            options["history"] = True
        print(f"{options} subgoals={subgoals} history={history}", flush=True)
        encoder = encoding.FlatRelationEncoder(table, **options)
        inputs = {}
        if subgoals is not None:
            inputs["subgoals"] = encoding.subgoal_masks(encoder, [[] if subgoals == "empty" else [[]]])
        if history is not None:
            inputs["history"] = encoding.history_masks(encoder, [[] if history == "empty" else [(-1, [])]])
        if subgoals == "layer" and not any(value and value > 0 for value in encoder.relation_levels):
            with pytest.raises(ValueError, match="^mymyr: encoding: subgoal layers are given, "
                                                 "but the schema has no subgoal relations$"):
                encoder.encode([task.initial_state], threads=1, **inputs)
            rejected += 1
            continue
        batch = encoder.encode([task.initial_state], threads=1, **inputs)
        assert batch.num_graphs == 1
        assert batch.num_relations == encoder.num_relations
        encoded += 1
    return {"encodes": encoded, "rejected": rejected}


@pytest.mark.parametrize("path", TASKS, ids=lambda path: str(path.relative_to(ROOT / "tests/data")))
def test_task_encodes_in_subprocess(path, request):
    if path.suffix == ".pddl" and not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    code = (
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from test_encoding_subprocess import encode_task\n"
        "from pathlib import Path\n"
        "import json\n"
        "print(json.dumps(encode_task(Path(sys.argv[1]))))\n"
    )
    child = run_code(code, str(path), capture_output=True, check=False)
    assert child.returncode == 0, f"{path}: exit {child.returncode}\n{child.stdout}\n{child.stderr}"
    result = json.loads(child.stdout.splitlines()[-1])
    assert result in ({"encodes": 60, "rejected": 0}, {"encodes": 48, "rejected": 12})
    request.node.user_properties.extend(result.items())


@pytest.mark.parametrize("storage", ["i32", "f64"])
@pytest.mark.parametrize("include_numeric", [False, True])
def test_numeric_derived_facts_follow_each_row_in_subprocess(tmp_path, storage, include_numeric):
    path = tmp_path / "numeric-derived.txt"
    path.write_text("""O 1
P 1
D 0 positive
N 1
F 0 value
SI 0
FI 0
NI 1
0 0
G 1
0 1
NG 0
M none
A 1
set 0
L 0
NC 0
E 1
0
L 0
NC 0
F 0
NE 1
assign 0 1 c 1
NA -
X 1
0 0
H
L 0
NC 1
> 1 f 0 1 c 0
""")
    code = """
import numpy as np
from mymyr import encoding, rl
task = mymyr.Task.from_text(sys.argv[1], numeric_storage=sys.argv[2])
table = rl.as_table(task)
assert task.words == 0 and table.words == 1
include_numeric = sys.argv[3] == 'True'
for outputs in (('relations',), ('relations', 'simple')):
    for layout in ('graph', 'relation'):
        encoder = encoding.FlatRelationEncoder(table, outputs=outputs, layout=layout,
                                               max_goal_level=3, history=True, include_numeric=include_numeric,
                                               goal_derivations=('plain', 'satisfied', 'unsatisfied'))
        first = encoder.encode([task.initial_state], threads=2)
        assert encoder.describe(first, 0).relations.get('positive', ()) == ()
        initial = task.initial_state
        positive = task.successor_states(initial)[0]
        states = [positive, initial, positive, initial]
        expected = [1, 0, 1, 0]
        for use_layers in (False, True):
            inputs = {}
            if use_layers:
                inputs['subgoals'] = encoding.subgoal_masks(encoder, [[['(not (positive))']]] * len(states))
                inputs['history'] = encoding.history_masks(encoder, [[(-1, ['(positive)'])]] * len(states))
            for use_arrays in (False, True):
                rows = np.zeros((len(states), table.words + 2 + table.numeric_words), dtype=np.uint64)
                for g, state in enumerate(states):
                    rows[g, table.words + 2:] = state.numeric_words
                batch = encoder.encode(rows if use_arrays else states, threads=2, **inputs)
                for g, want in enumerate(expected):
                    relations = encoder.describe(batch, g).relations
                    assert len(relations.get('positive', ())) == want, (g, relations)
                    assert len(relations.get('[+]positive[g][sat]', ())) == want, (g, relations)
                    assert len(relations.get('[+]positive[g][unsat]', ())) == 1 - want, (g, relations)
                    if use_layers:
                        assert len(relations.get('[-]positive[sg][sat]', ())) == 1 - want, (g, relations)
                        assert len(relations.get('[-]positive[sg][unsat]', ())) == want, (g, relations)
        stream = encoder.stream()
        for state in states:
            stream.append([state])
        batch = stream.flush(threads=2)
        assert [len(encoder.describe(batch, g).relations.get('positive', ())) for g in range(4)] == expected
"""
    child = run_code(code, str(path), storage, str(include_numeric), capture_output=True, check=False)
    assert child.returncode == 0, f"exit {child.returncode}\n{child.stdout}\n{child.stderr}"
