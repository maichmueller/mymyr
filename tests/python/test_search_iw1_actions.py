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
    return search.iw(task, max_width=1, optimize_iw1=kind == 'optimized',
                     width_zero='root_only' if kind == 'root_only' else 'expand_depth_one', **kw)


@pytest.mark.parametrize('kind', ['pass', 'ladder', 'root_only', 'optimized', 'abstracted'])
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


def test_incremental_layer_restrictions():
    task = text_task('gripper__prob05')
    for fn, width in [(search.iw, dict(max_width=1)), (search.abstracted_iw, dict(width=1))]:
        for kw in [dict(layer_order='reverse'), dict(threads=0), dict(threads=2),
                   dict(max_next_layer_states=3), dict(layer_order='in_order', beam_width=2)]:
            with pytest.raises(ValueError, match='queued layers and one thread'):
                fn(task, **width, incremental_first_applicability=True, **kw)


def test_negative_preconditions_constants_and_repeated_parameters():
    if not hasattr(mymyr, 'Domain'):
        pytest.skip('built without the loki front end')
    domain = mymyr.Domain.from_string('''
      (define (domain locks)
        (:requirements :strips :typing :negative-preconditions)
        (:types item) (:constants anchor - item)
        (:predicates (locked ?x - item) (ready ?x - item)
                     (link ?x ?y - item) (seen ?x - item))
        (:action unlock :parameters (?x - item) :precondition (locked ?x)
          :effect (and (not (locked ?x)) (ready ?x) (link ?x ?x)))
        (:action finish :parameters (?x - item)
          :precondition (and (not (locked ?x)) (link ?x ?x) (link anchor anchor))
          :effect (seen ?x))
        (:action copy :parameters (?x ?y - item)
          :precondition (and (ready ?x) (link ?x ?y) (link anchor anchor))
          :effect (seen ?y)))''')
    task = mymyr.Task(domain.instantiate_string('''
      (define (problem locks-3) (:domain locks) (:objects a b - item)
        (:init (locked a) (locked b) (link anchor anchor)) (:goal (seen b)))'''), atoms='lazy')
    for kind in ('pass', 'ladder', 'optimized', 'abstracted'):
        for canonical, witness in [(True, False), (False, False), (True, True)]:
            kw = dict(tree='nodes', goal=lambda s: False, canonical_order=canonical, witness_pruning=witness)
            baseline = run(task, kind, **kw)
            for incremental, precheck in [(True, False), (False, True), (True, True)]:
                result = run(task, kind, incremental_first_applicability=incremental,
                             precheck_add_effects=precheck, **kw)
                assert node_key(result) == node_key(baseline)
    for option in ('incremental_first_applicability', 'precheck_add_effects'):
        # A blocked first successor must leave its action eligible in another state.
        blocked = task.successor_states(task.initial_state)[:1]
        kw = dict(tree='nodes', goal=lambda s: False, blocked_states=blocked)
        assert node_key(run(task, 'pass', **kw)) == node_key(run(task, 'pass', **kw, **{option: True}))


@pytest.mark.parametrize('kind', ['ladder', 'abstracted'])
@pytest.mark.parametrize('layers', [dict(layer_order='reverse'),
                                    dict(layer_order='in_order', max_next_layer_states=5),
                                    dict(layer_order='goal_count', beam_width=5, randomize_ties=True, seed=7),
                                    dict(layer_order='in_order', beam_width=5, beam_novelty='survivors_only')])
def test_precheck_ordered_layers(kind, layers):
    task = text_task('gripper__prob05')
    kw = dict(tree='transitions', observer=Capture(), progress_interval=1, goal=lambda s: False, **layers)
    baseline = run(task, kind, **kw)
    kw['observer'] = Capture()
    result = run(task, kind, precheck_add_effects=True, **kw)
    assert node_key(result) == node_key(baseline)


@pytest.mark.parametrize('kind', ['optimized', 'abstracted'])
def test_cancel_after_fully_filtered_expansion(tmp_path, kind):
    count = 1100
    path = tmp_path / 'fanout.txt'
    path.write_text(f'O {count}\nP 1\nF 1 p\nSI 0\nFI 0\nG 0\nA 1\nmark 1\nL 0\nE 1\n0\nL 0\nF 1\n0 1 0\nX 0\n')
    task = mymyr.Task.from_text(str(path))
    token = search.CancelToken()

    class StopAtLast:
        def on_expand(self, node, state):
            if node == count:
                token.request()

    result = run(task, kind, goal=lambda s: False, observer=StopAtLast(), cancel=token,
                 incremental_first_applicability=True)
    assert token.requested
    assert result.status == search.Status.CANCELLED
