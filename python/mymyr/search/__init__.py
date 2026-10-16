"""The search family, heuristics and landmarks (search/*.hpp, landmarks/*.hpp and
reachability/*.hpp in the C++ core).

    r = mymyr.search.iw(task, max_width=2)              # IW(k) ladder
    r = mymyr.search.siw(task, max_width=2)             # serialized IW
    r = mymyr.search.astar_iw(task, heuristic="max", width=2, features="classical")
    r = mymyr.search.astar(task, heuristic="max")       # A* (eager; lazy=True for lazy)
    r = mymyr.search.astar(task, heuristic="ff", weight=2)   # weighted A*: f = g + 2 h
    r = mymyr.search.gbfs(task, heuristic="ff", lazy=True)
    r = mymyr.search.ehc(task)                          # enforced hill-climbing, h_FF with helpful actions
    r = mymyr.search.anytime_astar(task, max_seconds=60)    # restarting weighted A*, weights 5, 3, 2, 1
    r = mymyr.search.beam(task, beam_width=100)
    r = mymyr.search.brfs(task, threads=8)              # breadth-first search to a shortest plan, deterministic ids
    r.status, r.plan, r.cost, r.goal_state              # Status.SOLVED, [Action], float, State
    text = mymyr.search.format_plan(task, r.plan)       # IPC plan text; parse_plan(task, text) reads it back

    h = mymyr.search.Heuristic(task, "ff")              # h(state) -> float (+inf: dead end)
    r = mymyr.search.astar(task, heuristic="lmcut")     # LM-cut: admissible
    r = mymyr.search.gbfs(task, heuristic="lmcount", lazy=True)   # LAMA's landmark count, path-dependent
    r = mymyr.search.astar(task, heuristic=lambda s: my_estimate(s))   # a heuristic written in Python

The landmark count values a state by the path to it: a search keeps a status per node (the landmarks reached on the
path; a state reached on several paths keeps those all of them reached). ``Heuristic`` gives the same steps for a
search written in Python: ``status = h.initial_status(start)``, ``h.progress(status, state, action, successor)``,
``h.merge(status, other)`` and ``h(state, status)``; ``h(state)`` takes the state as the start of a path, with the
status ``h.initial_status(state)`` infers from it (as a search with ``start=`` does).

``astar(weight=w)`` is weighted A* (f = g + w h; with an admissible heuristic the plan costs at most w times the
optimum) and ``astar(cost_bound=c)`` searches for a plan cheaper than ``c`` only. ``ehc`` is enforced hill-climbing:
breadth-first lookaheads for a state with a smaller h, with the heuristic's preferred operators first
(``helpful_actions``) and greedy best-first search when hill-climbing fails (``fallback``); ``result.ehc`` holds its
counters. ``anytime_astar`` restarts weighted A* with decreasing ``weights``, each iteration bounded by the best plan
so far; ``result.solutions`` lists the improved plans in the order found (the observer's ``on_solution`` reports them
as they are found), ``result.optimal`` says whether the search proved the plan optimal.

``astar_iw`` runs weighted A* with minimum-g novelty pruning. It requires unit-cost actions and no numeric
fluents. ``features='classical'`` supports widths 1..5; ``'abstracted'`` and ``'base_abstracted'`` support 1..3.
``landmarks=`` restricts novelty to concrete landmark coordinates, and abstracted modes preserve goal and landmark
atom identities by default. ``weight`` is finite and nonnegative. A successor must lower a tuple label, and a
popped state must still own one at its g; root goals may bypass novelty. The result extends ``BestFirstResult``
with ``novelty`` statistics. Novelty pruning does not guarantee optimality.

A heuristic written in Python (``heuristic=`` of astar, gbfs, beam, ehc and anytime_astar) is a callable
``state -> float`` or an object (:class:`mymyr._typing.HeuristicObject`) with ``evaluate(state)``; ``math.inf`` marks a
dead end. Optionally the object has ``evaluate_batch(states)`` returning one float per state (a list or an array; eager
A* and eager GBFS then evaluate the new successors of an expansion in one call, beam search those of a layer) and
``preferred_actions()``, the preferred actions of the state of the last ``evaluate`` call (the lazy searches' preferred
list, EHC's helpful actions). It is called on the searching thread like the other callbacks below. Numeric tasks run
in A*, GBFS, beam search, EHC and the anytime search; h_max, h_add and h_FF ignore their numeric conditions and effects
(the delete relaxation), and g is the metric value.

IW, SIW, iw_pass, brfs and the IW family variants expand layer by layer in a chosen order (layer ordering
strategies): ``layer_order`` is ``"queue"`` (plain breadth-first, the default), ``"in_order"``, ``"reverse"``,
``"randomized"`` (with ``seed``) or ``"goal_count"`` (most satisfied goal literals first;
``prefer_more_satisfied_goals=False``: fewest first), and ``max_next_layer_states`` stops expanding a layer once the
next one holds that many states:

    r = mymyr.search.brfs(task, layer_order="goal_count", max_next_layer_states=100)

With ``beam_width`` an ordered search keeps only the best ``beam_width`` states of every next layer (equal scores in
generation order, or randomly by ``seed`` with ``randomize_ties=True``); ``beam_novelty`` decides whether the
successors the beam drops still mark the novelty table (``"all_tested"``) or only the kept ones do
(``"survivors_only"``):

    r = mymyr.search.iw(task, max_width=2, layer_order="goal_count", beam_width=32, beam_novelty="survivors_only")

``threads`` runs the beam's layer step (successor generation, scoring, read-only novelty tests) on that many threads
with the single-threaded result. ``beam_novelty="relaxed_survivors_only"`` (goal_count only) splits each layer's
transitions into ``min(threads, ceil(n / beam_chunk))`` parts and keeps the best ``beam_width`` of every part before
merging, so its result depends on the thread count (deterministic for a thread count and seed):

    r = mymyr.search.iw(task, max_width=2, layer_order="goal_count", beam_width=32, threads=8)
    r = mymyr.search.brfs(task, layer_order="goal_count", beam_width=32,
                          beam_novelty="relaxed_survivors_only", threads=8)

The IW family variants:

    r = mymyr.search.liw(task, max_width=1, landmarks="lifted")       # LIW(k): landmark-restricted novelty
    r = mymyr.search.abstracted_iw(task, width=2)                     # AIW / BAIW (base_abstracted=True)
    r = mymyr.search.projective_iw(task)                              # alias of abstracted IW(1)
    r = mymyr.search.rollout_iw(task, ordering=ActionOrdering.DIRECT_GOAL_ACHIEVER_FIRST)
    b = mymyr.search.find_rollouts_parallel(task, seeds=range(64), max_width=1, report_co_occurrence=True)
    rows = mymyr.search.intersect_co_occurrence(b)                   # {Atom: [Atom]} over the batch
    p = mymyr.search.atomic_goal_portfolio(task, goal=[["(on a b)"]], threads=1)
    p.plan, p.certified_optimal

LIW, abstracted IW and projective IW return an :class:`IwResult`; ``landmarks`` is a :class:`FactLandmarkGraph`,
``"approximate"`` or ``"lifted"`` (computed with default options), or None. Randomized orders use a portable
SplitMix64: the same seed gives the same run on every platform, and a rollout's result does not depend on the thread
count. Numeric tasks raise ValueError in the IW family variants (IW and SIW support them).

Landmarks and relaxed reachability:

    g = mymyr.search.approximate_fact_landmarks(task)       # grounded, with the achiever index
    g = mymyr.search.lifted_fact_landmarks(task)            # no grounding; g.lifted: lifted landmarks
    g.landmarks, g.disjunctive, g.orderings, g.num_unachieved(state)
    rr = mymyr.search.RelaxedReachability(task)             # delete-relaxed fixpoint without grounding
    rr.is_reachable("(on a b)"), rr.goal_reachable_without(["(clear a)"]), rr.restricted([...]).atoms("on")
    order = mymyr.search.LandmarkTransitionOrdering(g)      # iw(task, max_width=1, transition_ordering=order)

Every search runs with the thread state detached: other Python threads keep running, and any number of searches may
run at once on one task (free-threaded CPython). Common keyword arguments:

- budgets ``max_states``, ``max_expanded``, ``max_depth``, ``max_seconds`` (unset: unlimited; brfs takes
  ``max_states`` and ``max_seconds``). ``max_states`` counts the states a search stores, the start state included
  (IW: the nodes of a pass); the search stops with OUT_OF_STATES at the new state that fills them, inside its
  parent's expansion, so it never stores more. The deadline and the cancellation are checked inside expansions too;
- ``cancel``: a :class:`CancelToken`; ``token.request()`` from any thread stops the search (status CANCELLED);
- ``goal``: ``None`` (the task's goal), a callable ``state -> bool``, a :class:`mymyr.GroundCondition`, or a sequence
  of goals, each a GroundCondition or a sequence of ground literals and atoms (anything ``task.ground_condition``
  takes: static, fluent and derived literals of either polarity, e.g. ``[["(on a b)", "(not (clear c))"]]``); any of
  them counts. ``task.ground_condition(literals, constraints=["(>= (fuel t) 10)"])`` adds numeric constraints;
- ``blocked_states``: States the search never enters;
- ``observer``: a :class:`Observer` subclass that overrides the events it needs: ``on_start(state)``,
  ``on_expand(id, state)``, ``on_generate(parent, action, child, state, is_new)``, ``on_prune(parent, action, state)``,
  ``on_pass(width, stats)``, ``on_solution(plan, cost)``, ``on_progress(stats) -> bool`` (False stops the search),
  ``on_end(status, stats)``, and for the IW family variants ``on_transition(parent, action, child, state, outcome)``
  (a :class:`TransitionOutcome`). Events it does not override are never called. Any object with some of these
  methods works too:

    class Counter(mymyr.search.Observer):
        def __init__(self):
            self.expanded = 0

        def on_expand(self, id, state):
            self.expanded += 1

    r = mymyr.search.astar(task, observer=(c := Counter()))   # c.expanded == r.stats.expanded

Callbacks (observer methods, a goal callable) run on the searching thread with the thread state attached. The first
exception a callback raises requests the search's CancelToken and is re-raised when the search returns. The parallel
searches (brfs with ``threads > 1``, find_rollouts_parallel, atomic_goal_portfolio) run their workers on their own
threads only when the observer and a callable goal have a ``make_worker(k)`` method returning worker k's own object
(the make_worker protocol; brfs takes no callable goal with ``threads > 1``); otherwise they run on the calling thread
alone.
"""

from mymyr._core._search import (
    ActionOrdering,
    AnytimeIteration,
    AnytimeResult,
    AnytimeSolution,
    BestFirstResult,
    AStarIwResult,
    AStarIwNoveltyStatistics,
    BrfsResult,
    CancelToken,
    CompleteFactLandmarks,
    EhcResult,
    EhcStatistics,
    FactLandmarkGraph,
    FirstAchievers,
    Heuristic,
    HeuristicStatistics,
    IwPass,
    IwResult,
    LandingState,
    LandmarkTransitionOrdering,
    LandmarkTransitionScore,
    LiftedLandmark,
    MergedLandingStates,
    ParallelRolloutsResult,
    PlanError,
    PlanValidation,
    PortfolioResult,
    ReachabilityDisambiguation,
    ReachabilityStatistics,
    ReachabilityTable,
    RelaxedReachability,
    RolloutIwResult,
    RolloutIwStatistics,
    RolloutResult,
    SiwResult,
    SiwSubproblem,
    Statistics,
    Status,
    TransitionOutcome,
    WitnessQuery,
    WitnessVerdict,
    abstracted_iw,
    anytime_astar,
    approximate_fact_landmarks,
    astar,
    astar_iw,
    atomic_goal_portfolio,
    beam,
    brfs,
    ehc,
    find_rollouts_parallel,
    format_plan,
    gbfs,
    intersect_co_occurrence,
    iw,
    iw_pass,
    lifted_fact_landmarks,
    liw,
    merge_landing_states,
    parse_plan,
    projective_iw,
    rollout_iw,
    siw,
    validate_plan,
    verify_pi_plus_fact_landmarks,
)
from mymyr.search._observer import Observer

__all__ = [
    "ActionOrdering",
    "AnytimeIteration",
    "AnytimeResult",
    "AnytimeSolution",
    "BestFirstResult",
    "AStarIwResult",
    "AStarIwNoveltyStatistics",
    "BrfsResult",
    "CancelToken",
    "CompleteFactLandmarks",
    "EhcResult",
    "EhcStatistics",
    "FactLandmarkGraph",
    "FirstAchievers",
    "Heuristic",
    "HeuristicStatistics",
    "IwPass",
    "IwResult",
    "LandingState",
    "LandmarkTransitionOrdering",
    "LandmarkTransitionScore",
    "LiftedLandmark",
    "MergedLandingStates",
    "Observer",
    "ParallelRolloutsResult",
    "PlanError",
    "PlanValidation",
    "PortfolioResult",
    "ReachabilityDisambiguation",
    "ReachabilityStatistics",
    "ReachabilityTable",
    "RelaxedReachability",
    "RolloutIwResult",
    "RolloutIwStatistics",
    "RolloutResult",
    "SiwResult",
    "SiwSubproblem",
    "Statistics",
    "Status",
    "TransitionOutcome",
    "WitnessQuery",
    "WitnessVerdict",
    "abstracted_iw",
    "anytime_astar",
    "approximate_fact_landmarks",
    "astar",
    "astar_iw",
    "atomic_goal_portfolio",
    "beam",
    "brfs",
    "ehc",
    "find_rollouts_parallel",
    "format_plan",
    "gbfs",
    "intersect_co_occurrence",
    "iw",
    "iw_pass",
    "lifted_fact_landmarks",
    "liw",
    "merge_landing_states",
    "parse_plan",
    "projective_iw",
    "rollout_iw",
    "siw",
    "validate_plan",
    "verify_pi_plus_fact_landmarks",
]
