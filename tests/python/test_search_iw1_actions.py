"""Width-one action filtering preserves admitted trees and expansion/progress ordering."""

import numpy as np
import pytest

import mymyr
from mymyr import search
from conftest import ROOT, text_task


class Capture(search.Observer):
    def __init__(self, generate=False):
        super().__init__()
        self.generate = generate
        self.pending = False
        self.expansions = []
        self.admitted = []
        self.passes = []

    def on_expand(self, node, state):
        assert not self.pending
        self.pending = True
        self.expansions.append((node, state))

    def on_progress(self, stats):
        assert self.pending
        self.pending = False
        self.expansions[-1] += (stats.states,)
        return True

    def on_generate(self, parent, action, child, state, is_new):
        assert not self.pending
        if self.generate and is_new:
            self.admitted.append((parent, str(action), child, state))

    def on_pass(self, width, stats):
        assert not self.pending
        self.passes.append((self.expansions, self.admitted))
        self.expansions, self.admitted = [], []


@pytest.fixture(params=['spanner', 'logistics00__probLOGISTICS-6-1', 'rovers__p02'])
def task(request):
    if request.param == 'spanner':
        if not hasattr(mymyr, 'Domain'):
            pytest.skip('built without the loki front end')
        p = ROOT / 'tests/data/ipc/spanner-ipc/test'
        return mymyr.Task.from_pddl(p / 'domain.pddl', p / 'p01-easy.pddl')
    return text_task(request.param)


def node_key(result):
    return [(p.width, p.status, None if p.tree is None else
             (tuple(p.tree.states()[:p.tree.num_nodes]), tuple(p.tree.parent), tuple(p.tree.depth),
              tuple(map(str, p.tree.labels()[:p.tree.num_nodes])), tuple(p.tree.node_flags))) for p in result.passes]


def run(task, kind, **kw):
    if kind == 'abstracted':
        return search.abstracted_iw(task, keep_depth_one_novel=True, **kw)
    if kind == 'pass':
        return search.iw_pass(task, 1, **kw)
    return search.iw(task, max_width=1, optimize_iw1=kind == 'optimized', **kw)


@pytest.mark.parametrize('kind', ['pass', 'ladder', 'optimized', 'abstracted'])
@pytest.mark.parametrize('observed', [False, True])
def test_tree_equality(task, kind, observed):
    for state in [task.initial_state] + task.successor_states(task.initial_state)[:2]:
        kw = dict(start=state, tree='nodes', goal=lambda s: False, canonical_order=True)
        baseline_obs = Capture(True) if observed else None
        baseline = run(task, kind, observer=baseline_obs, progress_interval=1, **kw)
        for incremental, precheck in [(True, False), (False, True), (True, True)]:
            obs = Capture(True) if observed else None
            result = run(task, kind, observer=obs, progress_interval=1,
                         incremental_first_applicability=incremental, precheck_add_effects=precheck, **kw)
            assert node_key(result) == node_key(baseline)
            assert result.stats.generated <= baseline.stats.generated
            if observed:
                assert obs.passes == baseline_obs.passes


@pytest.mark.parametrize('kind', ['pass', 'ladder', 'optimized', 'abstracted'])
def test_transition_tree_and_blocked_states(task, kind):
    blocked = task.successor_states(task.initial_state)[:1]
    kw = dict(tree='transitions', blocked_states=blocked, goal=lambda s: False, progress_interval=1)
    baseline = run(task, kind, **kw)
    result = run(task, kind, incremental_first_applicability=True, precheck_add_effects=True, **kw)
    assert node_key(result) == node_key(baseline)
    for p in result.passes:
        if p.tree is not None:
            t = p.tree
            assert t.offsets[-1] == t.num_transitions
            for parent, child in zip(np.repeat(np.arange(t.num_nodes), np.diff(t.offsets)), t.t_child):
                if child >= 0:
                    assert t.parent[child] == parent or p.width == 0


@pytest.mark.parametrize('option', ['incremental_first_applicability', 'precheck_add_effects'])
def test_unsupported_options(option):
    task = text_task('gripper__prob05')
    ce = text_task('miconic-simpleadl__s10-2')
    for fn, kw in [(search.iw, dict(max_width=1)), (search.iw_pass, dict(width=1)),
                   (search.abstracted_iw, dict(width=1))]:
        with pytest.raises(ValueError, match='conditional effects'):
            fn(ce, **kw, **{option: True})
        wider = {'max_width': 2} if fn == search.iw else {'width': 2}
        with pytest.raises(ValueError, match='width 1'):
            fn(task, **wider, **{option: True})
    # The automatic fast precheck falls back on conditional-effect tasks.
    baseline = search.abstracted_iw(ce, tree='nodes', goal=lambda s: False)
    observed = search.abstracted_iw(ce, tree='nodes', goal=lambda s: False, observer=Capture(), progress_interval=1)
    assert node_key(baseline) == node_key(observed)
    graph = search.approximate_fact_landmarks(task)
    with pytest.raises(ValueError, match='landmark'):
        search.abstracted_iw(task, landmarks=graph, **{option: True})
    with pytest.raises(ValueError, match='transition ordering'):
        search.iw(task, max_width=1, transition_ordering=search.LandmarkTransitionOrdering(graph), **{option: True})


def test_derived_and_numeric_applicability():
    for task in [text_task('openstacks-opt08-adl__p03'),
                 mymyr.Task.from_text(str(ROOT / 'tests/data/numeric_tasks/cs-counters.txt'))]:
        baseline = search.iw(task, max_width=1, tree='nodes', goal=lambda s: False)
        result = search.iw(task, max_width=1, tree='nodes', goal=lambda s: False,
                           incremental_first_applicability=True, precheck_add_effects=True)
        assert node_key(result) == node_key(baseline)
