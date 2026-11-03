"""Learned evaluators in the searches: the StateBatch an evaluator receives (rows, tasks, encodings), its values,
batched expansions, scored layer orderings and the SharedEvaluator that coalesces the batches of many searches."""

import math
import pathlib

import numpy as np
import pytest

import mymyr
from mymyr import encoding, rl, search
from mymyr.search import Status
from conftest import text_task

ROOT = pathlib.Path(__file__).resolve().parents[2]
GRIPPER = ROOT / "tests/data/benchmarks/small/gripper/domain.pddl"


@pytest.fixture(scope="module")
def task():
    return text_task("logistics00__probLOGISTICS-6-1")


def gripper_tasks(sizes=(4, 10, 20)):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    return [mymyr.Task.from_pddl(GRIPPER, ROOT / f"tests/data/table_instances/gripper/gripper-{n}.pddl")
            for n in sizes]


def plan_of(r):
    return [str(a) for a in r.plan]


class Recording:
    """h_max through evaluate_batch, recording every batch."""

    def __init__(self, task, kind="max"):
        self.h = search.Heuristic(task, kind)
        self.batches = []

    def evaluate_batch(self, batch):
        self.batches.append(batch)
        return np.array([self.h(s) for s in batch.states])


# ------------------------------------------------------------------------------------------------ StateBatch


def test_batch_rows_are_the_packed_states(task):
    obj = Recording(task)
    r = search.astar(task, heuristic=obj)
    assert r.status == Status.SOLVED
    assert r.cost == search.astar(task, heuristic="max").cost
    batch = max(obj.batches, key=len)
    assert len(batch) > 1
    assert batch.rows.dtype == np.uint64 and batch.rows.shape == (len(batch), batch.words + batch.numeric_words)
    assert not batch.rows.flags.writeable
    assert batch.words == task.words and batch.numeric_words == 0
    assert batch.task_ids.dtype == np.int32 and not batch.task_ids.any()
    assert batch.tasks == (task,) and batch.tasks[0] is task
    assert batch.encoded is None
    for row, state in zip(batch.rows, batch.states):
        words = state.words
        assert np.array_equal(row[: len(words)], words) and not row[len(words):].any()
    # the batch owns its arrays: they outlive the call
    assert sum(len(b) for b in obj.batches) == r.evaluations


def test_numeric_rows_carry_the_values():
    numeric = mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt"))
    obj = Recording(numeric, "goal_count")
    r = search.gbfs(numeric, heuristic=obj)
    assert (r.status, r.cost, plan_of(r)) == (lambda n: (n.status, n.cost, plan_of(n)))(
        search.gbfs(numeric, heuristic="goal_count", queue="heap"))
    batch = obj.batches[-1]
    assert batch.numeric_words == numeric.numeric_words > 0
    for row, state in zip(batch.rows, batch.states):
        assert np.array_equal(row[batch.words:].view(np.float64), np.asarray(state.numeric_values(), dtype=np.float64)) \
            or numeric.numeric_storage != "f64"


def test_values_as_arrays(task):
    h = search.Heuristic(task, "max")
    expected = search.astar(task, heuristic="max").cost

    def returning(make):
        class Arr:
            def evaluate_batch(self, batch):
                return make([h(s) for s in batch.states])

        return Arr()

    for make in (lambda v: np.array(v, dtype=np.float32), lambda v: np.array(v, dtype=np.int64),
                 lambda v: np.array(v)[:, None], lambda v: np.array(v)[::-1].copy()[::-1], tuple):
        assert search.astar(task, heuristic=returning(make)).cost == expected
    with pytest.raises(ValueError, match=r"array of shape \(\d+, 2\) for \d+ states"):
        search.astar(task, heuristic=returning(lambda v: np.zeros((len(v), 2))))
    with pytest.raises(TypeError, match="unsupported dtype"):
        search.astar(task, heuristic=returning(lambda v: np.zeros(len(v), dtype=np.complex64)))
    with pytest.raises(ValueError, match="a Python heuristic returned -2.0"):
        search.astar(task, heuristic=returning(lambda v: -2 * np.ones(len(v))))
    with pytest.raises(ValueError, match="a Python heuristic returned nan"):
        search.astar(task, heuristic=returning(lambda v: np.full(len(v), np.nan)))
    inf = search.gbfs(task, heuristic=returning(lambda v: np.where(np.arange(len(v)) % 2, np.inf, 1.0)))
    assert inf.dead_ends > 0


def test_torch_values():
    torch = pytest.importorskip("torch")
    task = text_task("gripper__prob05")
    h = search.Heuristic(task, "max")

    class Torch:
        def evaluate_batch(self, batch):
            return torch.tensor([h(s) for s in batch.states], dtype=torch.float32)

    assert search.astar(task, heuristic=Torch()).cost == search.astar(task, heuristic="max").cost


def test_encoder_encodes_the_rows(task):
    class Encoded:
        def __init__(self):
            self.encoder = encoding.FlatRelationEncoder(task)
            self.checked = 0

        def evaluate_batch(self, batch):
            expected = self.encoder.encode(batch.states)
            got = batch.encoded
            assert isinstance(got, encoding.FlatBatch)
            for name, array in expected.as_dict().items():
                assert np.array_equal(np.asarray(array), np.asarray(got.as_dict()[name])), name
            self.checked += len(batch)
            return np.diff(np.asarray(got.relation_offsets)).reshape(len(batch), -1).sum(axis=1) * 0.0

    obj = Encoded()
    r = search.gbfs(task, heuristic=obj, max_expanded=50)
    assert r.status == Status.OUT_OF_STATES and obj.checked == r.evaluations > 0

    class Options(Encoded):
        encode_options = {"threads": 2, "framework": "dlpack"}

        def evaluate_batch(self, batch):
            assert type(batch.encoded.relation_offsets).__name__ == "DLArray"
            return np.zeros(len(batch))

    assert search.gbfs(task, heuristic=Options(), max_expanded=20).status == Status.OUT_OF_STATES


def test_encoder_over_a_table():
    tasks = gripper_tasks()
    table = rl.TaskTable(tasks)

    class Over:
        encoder = encoding.FlatRelationEncoder(table)

        def __init__(self, i):
            self.i = i

        def evaluate_batch(self, batch):
            assert (batch.task_ids == self.i).all() and batch.tasks[self.i] is tasks[self.i]
            assert batch.words == table.words
            return np.zeros(len(batch))

    for i, t in enumerate(tasks):
        assert search.gbfs(t, heuristic=Over(i), max_expanded=20).evaluations > 0
    other = text_task("logistics00__probLOGISTICS-6-1")
    with pytest.raises(ValueError, match="not an instance of the table of the evaluator's encoder"):
        search.gbfs(other, heuristic=Over(0))


def test_evaluator_shape_errors(task):
    class NoBatch:
        encoder = None

        def evaluate(self, s):
            return 0.0

    assert search.gbfs(task, heuristic=NoBatch(), max_expanded=5).evaluations > 0

    class EncoderWithoutBatch:
        encoder = encoding.FlatRelationEncoder(task)

        def evaluate(self, s):
            return 0.0

    with pytest.raises(TypeError, match="encoder needs evaluate_batch"):
        search.gbfs(task, heuristic=EncoderWithoutBatch())

    class BadOptions:
        encoder = encoding.FlatRelationEncoder(task)
        encode_options = ["threads"]

        def evaluate_batch(self, batch):
            return np.zeros(len(batch))

    with pytest.raises(TypeError, match="encode_options is a dict"):
        search.gbfs(task, heuristic=BadOptions())
    with pytest.raises(TypeError):
        search.gbfs(task, heuristic=42)


# ------------------------------------------------------------------------------------------------ batched expansions


@pytest.mark.parametrize("algorithm", ["astar", "gbfs"])
def test_expansion_batch_gives_larger_batches(task, algorithm):
    run = getattr(search, algorithm)
    sizes = {}
    for k in (1, 8, 64):
        obj = Recording(task)
        r = run(task, heuristic=obj, expansion_batch=k)
        assert r.status == Status.SOLVED
        sizes[k] = max(len(b) for b in obj.batches)
        if algorithm == "astar":
            assert r.cost == search.astar(task, heuristic="max").cost  # admissible, reopening: optimal
    assert sizes[1] < sizes[8] < sizes[64]


def test_expansion_batch_budgets_and_refusals(task):
    r = search.gbfs(task, heuristic=Recording(task), expansion_batch=16, max_expanded=40)
    assert r.status == Status.OUT_OF_STATES and r.stats.expanded == 40
    with pytest.raises(ValueError, match="expansion_batch > 1 needs an eager search"):
        search.astar(task, heuristic="max", lazy=True, expansion_batch=4)
    with pytest.raises(ValueError):
        search.gbfs(task, expansion_batch=0)
    r = search.gbfs(task, heuristic="lmcount", expansion_batch=4)
    assert r.status == Status.FAILED and "path-dependent" in r.message


# ------------------------------------------------------------------------------------------------ scored layers


class GoalScore:
    """Unsatisfied goal atoms of each state: GoalCount's ranking as a learned score."""

    def __init__(self, task):
        self.h = search.Heuristic(task, "goal_count")
        self.calls = 0

    def evaluate_batch(self, batch):
        self.calls += 1
        return np.array([self.h(s) for s in batch.states])


@pytest.mark.parametrize("novelty", ["all_tested", "survivors_only"])
@pytest.mark.parametrize("threads", [1, 4])
def test_scored_layers_rank_like_goal_count(novelty, threads):
    blocks = text_task("blocks__probBLOCKS-8-0")
    for run in (lambda **kw: search.iw(blocks, max_width=1, **kw), lambda **kw: search.brfs(blocks, **kw)):
        common = dict(beam_width=8, beam_novelty=novelty, threads=threads, randomize_ties=True, seed=3)
        a = run(layer_order="goal_count", **common)
        scorer = GoalScore(blocks)
        b = run(layer_order=scorer, **common)
        assert scorer.calls > 0
        assert plan_of(a) == plan_of(b)
        assert a.stats.expanded == b.stats.expanded if hasattr(a, "stats") else a.expanded == b.expanded


def test_scored_layers_values_and_errors():
    blocks = text_task("blocks__probBLOCKS-8-0")
    h = search.Heuristic(blocks, "goal_count")
    a = search.iw(blocks, max_width=1, layer_order="goal_count", prefer_more_satisfied_goals=False, beam_width=4)
    # negative scores are fine: lower first
    b = search.iw(blocks, max_width=1, layer_order=lambda s: -h(s), beam_width=4)
    assert plan_of(a) == plan_of(b)
    with pytest.raises(ValueError, match="layer score is nan"):
        search.iw(blocks, max_width=1, layer_order=lambda s: math.nan, beam_width=4)
    with pytest.raises(ZeroDivisionError):
        search.brfs(blocks, layer_order=lambda s: 1 / 0)
    with pytest.raises(ValueError, match="relaxed_survivors_only|RelaxedSurvivorsOnly"):
        search.brfs(blocks, layer_order=lambda s: 0.0, beam_width=4, beam_novelty="relaxed_survivors_only")
    with pytest.raises(ValueError, match="layer_order must be"):
        search.iw(blocks, layer_order="best")


def test_scored_layers_record_trees():
    blocks = text_task("blocks__probBLOCKS-8-0")
    h = search.Heuristic(blocks, "goal_count")
    a = search.iw(blocks, max_width=1, layer_order="goal_count", beam_width=6, tree="nodes")
    b = search.iw(blocks, max_width=1, layer_order=lambda s: h(s), beam_width=6, tree="nodes")
    assert plan_of(a) == plan_of(b)
    ta, tb = a.passes[-1].tree, b.passes[-1].tree
    assert ta.num_nodes == tb.num_nodes > 0
    assert np.array_equal(np.asarray(ta.rows), np.asarray(tb.rows))
    assert np.array_equal(np.asarray(ta.parent), np.asarray(tb.parent))
