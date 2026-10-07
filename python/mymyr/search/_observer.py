"""mymyr.search.Observer: the base class of search observers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from mymyr._core import Action, State
    from mymyr._core._search import Statistics, Status, TransitionOutcome

__all__ = ["Observer"]

_F = TypeVar("_F", bound=Callable[..., object])


def _default_event(fn: _F) -> _F:
    # The searches skip methods that carry this attribute: an event a subclass does not override is never called.
    fn._mymyr_default_event = True  # type: ignore[attr-defined]
    return fn


class Observer:
    """Base class of search observers: pass an instance as ``observer=`` to a search and override the events of
    interest. The searches never call an event a subclass does not override (it costs nothing), and an observer that
    overrides none is not installed at all. ``id``, ``parent`` and ``child`` are the search's own state ids.

    Which events a search sends:

    - every search: ``on_start`` once, ``on_expand`` for every counted expansion, ``on_generate`` for every successor
      generated, ``on_progress`` every ``progress_interval`` expansions, ``on_solution`` when a plan was found, and
      ``on_end`` once (so the number of ``on_expand`` and ``on_generate`` calls equals the result's ``expanded`` and
      ``generated`` statistics);
    - ``on_prune`` for a successor that is not entered (not novel, blocked, a dead end), after its ``on_generate``;
    - ``on_pass``: after every IW(k) pass (iw, siw, the IW family variants: the arity and the pass's statistics) and
      after every layer of ``brfs`` (its depth and the layer's statistics);
    - ``on_transition``: every transition of an expanded state with a :class:`TransitionOutcome` (the IW family
      variants, astar_iw).

    Threads. Every event runs with the thread state attached, on the thread that called the search, except in the
    parallel searches (``brfs`` with ``threads > 1``, ``find_rollouts_parallel``, ``atomic_goal_portfolio``): there
    the search first calls :meth:`make_worker` once per worker on the calling thread, and worker ``k`` sends its hot
    events (``on_expand``, ``on_generate``, ``on_prune``, ``on_transition``, ``on_progress``) to the observer it got,
    from that worker's thread; the lifecycle events (``on_start``, ``on_pass``, ``on_solution``, ``on_end``) stay on
    this, the root observer, on the calling thread. Worker observers run at the same time on free-threaded CPython,
    so give each its own state and combine it after the search (or in ``on_end``). Without ``make_worker`` (or when
    it returns None) a parallel search runs on the calling thread alone.

    Errors. The first exception an event raises stops the search and is re-raised when the search returns.

    Any object with some of these methods works as an observer as well; the base class documents and types them.
    """

    @_default_event
    def on_start(self, state: State) -> None:
        """The search starts from ``state``."""

    @_default_event
    def on_expand(self, id: int, state: State) -> None:
        """State ``id`` is expanded."""

    @_default_event
    def on_generate(self, parent: int, action: Action, child: int | None, state: State, is_new: bool) -> None:
        """``action`` leads from ``parent`` to ``state``. ``child`` is its id (None when it got none: pruned, or in
        multi-threaded brfs, which numbers states when a layer ends); ``is_new``: the state was entered by this
        transition."""

    @_default_event
    def on_prune(self, parent: int, action: Action, state: State) -> None:
        """The successor ``state`` of ``parent`` is not entered."""

    @_default_event
    def on_transition(
        self, parent: int, action: Action, child: int | None, state: State, outcome: TransitionOutcome
    ) -> None:
        """What became of a successor (the IW family variants, astar_iw); ``child`` is None when it got no id."""

    @_default_event
    def on_pass(self, arity: int, stats: Statistics) -> None:
        """An IW pass of this arity, or the brfs layer of this depth, ended; ``stats`` are its own counts."""

    @_default_event
    def on_solution(self, plan: list[Action], cost: float) -> None:
        """A plan was found."""

    @_default_event
    def on_progress(self, stats: Statistics) -> bool:
        """Every ``progress_interval`` expansions with the statistics so far; return False to stop the search (status
        CANCELLED)."""
        return True

    @_default_event
    def on_end(self, status: Status, stats: Statistics) -> None:
        """The search ended with ``status`` and the total ``stats``."""

    @_default_event
    def make_worker(self, worker: int) -> Observer | None:
        """The observer of worker ``worker`` of a parallel search, or None to run that search on the calling thread."""
        return None
