"""Structural types named by the stubs of mymyr._core (python/src/typing.hpp), and the Python classes they name
(imported for type checkers only: stubgen resolves mymyr._typing without the packages that define them). Only type
checkers use them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from mymyr import State
    from mymyr.encoding import DeviceFlatBatch as DeviceFlatBatch
    from mymyr.encoding import DeviceFlatStream as DeviceFlatStream
    from mymyr.encoding import GoalMasks as GoalMasks
    from mymyr.encoding import GraphDescription as GraphDescription
    from mymyr.encoding import LayerMasks as LayerMasks
    from mymyr.rl import InstanceSampler as InstanceSampler

__all__ = ["HeuristicObject", "SupportsCudaStream", "SupportsDLPack", "SupportsHorizonView", "DeviceActionRows"]


class SupportsDLPack(Protocol):
    """An array that exports itself through DLPack: NumPy, torch and JAX arrays, mymyr.DLArray.

    State words are such arrays: [N, W] 64-bit or [N, 2W] 32-bit integers (one state: [W] or [2W])."""

    def __dlpack__(self) -> Any: ...


class SupportsCudaStream(Protocol):
    """A CUDA stream object: its ``cuda_stream`` is the cudaStream_t (torch.cuda.Stream)."""

    @property
    def cuda_stream(self) -> int: ...


class HeuristicObject(Protocol):
    """A heuristic written in Python for mymyr.search.astar, gbfs and beam (``heuristic=``).

    evaluate(state) returns h of the state (``math.inf`` marks a dead end). Optional methods the searches use when
    present: evaluate_batch(states) -> sequence of floats, one per state (then eager A* and eager GBFS hand over the new
    successors of each expansion in one call, beam search those of a layer; an object may have evaluate_batch alone),
    and preferred_actions() -> iterable of actions, the preferred operators of the state of the last evaluate call
    (the lazy searches' preferred list). A plain callable state -> float is accepted as well."""

    def evaluate(self, state: State) -> float: ...


class SupportsHorizonView(Protocol):
    """The input of FlatHorizonEncoder.encode: ``mymyr.encoding.HorizonView`` or any object with these attributes (the
    edge arrays and ``task_ids`` may be None where the schema does not read them)."""

    @property
    def nodes(self) -> Any: ...
    @property
    def node_offsets(self) -> Any: ...
    @property
    def edge_offsets(self) -> Any: ...
    @property
    def edge_src(self) -> Any: ...
    @property
    def edge_dst(self) -> Any: ...
    @property
    def edge_schema(self) -> Any: ...
    @property
    def edge_binding(self) -> Any: ...
    @property
    def task_ids(self) -> Any: ...


class DeviceActionRows(Protocol):
    """Expansion labels on one CUDA device. Offsets delimit the canonical rows of each parent state."""

    @property
    def schema(self) -> SupportsDLPack: ...
    @property
    def binding(self) -> SupportsDLPack: ...
    @property
    def offsets(self) -> SupportsDLPack: ...
    @property
    def parent(self) -> SupportsDLPack | None: ...
