"""Structural types named by the stubs of mymyr._core (python/src/typing.hpp). Only type checkers use them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from mymyr import State

__all__ = ["HeuristicObject", "SearchObserver", "SupportsCudaStream", "SupportsDLPack"]


class SupportsDLPack(Protocol):
    """An array that exports itself through DLPack: NumPy, torch and JAX arrays, mymyr.DLArray.

    State words are such arrays: [N, W] 64-bit or [N, 2W] 32-bit integers (one state: [W] or [2W])."""

    def __dlpack__(self) -> Any: ...


class SupportsCudaStream(Protocol):
    """A CUDA stream object: its ``cuda_stream`` is the cudaStream_t (torch.cuda.Stream)."""

    @property
    def cuda_stream(self) -> int: ...


class SearchObserver(Protocol):
    """A search observer (mymyr.search): any object. The searches call whichever of these methods it has, so none is
    required (and every object satisfies this protocol):

    on_start(state), on_expand(id, state), on_generate(parent, action, child, state, is_new), on_prune(parent, action,
    state), on_transition(parent, action, child, state, outcome) (the IW family variants), on_pass(arity, stats),
    on_solution(plan, cost), on_progress(stats) -> bool (False stops the search), on_end(status, stats), and
    make_worker(k) -> SearchObserver | None (the make_worker protocol: worker k's own observer in the parallel
    searches)."""


class HeuristicObject(Protocol):
    """A heuristic written in Python for mymyr.search.astar, gbfs and beam (``heuristic=``).

    evaluate(state) returns h of the state (``math.inf`` marks a dead end). Optional methods the searches use when
    present: evaluate_batch(states) -> sequence of floats, one per state (then eager A* and eager GBFS hand over the new
    successors of each expansion in one call, beam search those of a layer; an object may have evaluate_batch alone),
    and preferred_actions() -> iterable of actions, the preferred operators of the state of the last evaluate call
    (the lazy searches' preferred list). A plain callable state -> float is accepted as well."""

    def evaluate(self, state: State) -> float: ...

