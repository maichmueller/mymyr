"""Encoding rows in CUDA memory: ``FlatRelationEncoder.encode`` on device rows and the torch op behind it.

Rows (and the per-row task ids, derived words and goal masks) that are CUDA tensors, or any DLPack or
``__cuda_array_interface__`` array on a CUDA device, are encoded by the device kernels into torch tensors on the same
device. The arrays are allocated on torch's current stream and the kernels run on it, in stream order, like torch's own
operations: nothing needs a synchronization before or after the call. ``DeviceFlatBatch`` is the result; it has the
arrays, names and accessors of the host ``FlatBatch``.

Synchronization: an encoding without ``capacities`` must size its arrays by the batch, so it counts the batch, copies the
sizes to the host and waits for them once, then allocates and fills the arrays. An encoding with ``capacities`` allocates
every array at its capacity, writes the true sizes into a device tensor (``DeviceFlatBatch.device_sizes``) and never
waits: it is what ``torch.compile`` and CUDA graphs run.

Capacity calls allocate scratch through torch's caching allocator on the current stream. During capture the graph's
private pool owns it through the last replay. Warm the encoder on the capture stream to upload and acquire its context;
no scratch reservation is needed. Exact calls reuse a workspace per stream; ``reserve_device`` can size it in advance
and ``release_device`` frees it.

The subgoal layers and history steps of the rows (``subgoals=``, ``history=``) are CUDA tensors of the same forms as on
the host: masks [N, L, W] (or [1, L, W] for layers that every row shares) of 64-bit words, ``dt`` int32 [N, H] and the
valid step ``count`` int32 [N]; device streams and device collation carry neither. What the device does not encode: the
Simple graph with numeric relations or of a horizon. Shared statics (``statics='shared'``) keep one static block per
instance of the batch, as on the host. The device evaluates axioms into scratch when a graph needs derived atoms and
``derived=`` is absent; supplied derived words take precedence.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any, Optional

import numpy as np

from mymyr._core._encoding import FlatBatch

__all__ = ["DeviceFlatBatch"]

# The arrays of the Simple graph; a Simple-only batch stores them in place of the relation arrays.
_SIMPLE_ARRAYS = ("node_offsets", "node_label", "edge_offsets", "edge_index", "edge_label")
# The shared static blocks of the instances of a batch (statics='shared').
_STATIC_ARRAYS = ("static_task_ids", "static_offsets", "static_arg_offsets", "static_args", "static_relations")
# The arrays of a batch in the order that the native side takes them, then the sizes.
_ARRAYS = (
    "task_ids",
    "num_objects",
    "entity_offsets",
    "entity_role",
    "entity_ref",
    "relation_counts",
    "relation_offsets",
    "relation_arg_offsets",
    "relation_args",
    "goal_complete",
    "target_offsets",
    "target_entity",
    "target_group",
    "history_offsets",
    "history_entity",
    "history_dt",
    *_SIMPLE_ARRAYS,
    "numeric_values",
    *_STATIC_ARRAYS,
)
_OUTPUTS = (*_ARRAYS, "static_sizes", "sizes")
# The private relation storage of a Simple-only fill that follows the outputs in the native pointer list.
_RELATION_SCRATCH = ("_relation_offsets", "_relation_arg_offsets", "_relation_args")
# The native output fields: the outputs, the private relation storage and the edge_index row stride.
_POINTER_FIELDS = len(_OUTPUTS) + len(_RELATION_SCRATCH) + 1
# The row record (_Rows.fields) the native side decodes by position: the graph count; the states (pointer, row stride,
# words), task ids and derived words (pointer, row stride); the goal masks (8), subgoal layers (9) and history steps
# (11, with dt and count); the expansion rows (offsets, schema, binding, parent, capacity, label width, status,
# canonical index); the
# horizon view (node offsets, edge offsets, src, dst, schema, binding, nodes, edges, label width, scratch, scratch
# words, status); the constant literals of the goals and of the subgoal layers (rows, row stride, capacity, width).
_ROW_STATES = 1
_ROW_DERIVED = 5
_ROW_ACTIONS = 35
_ROW_HORIZON = 43
_ROW_HORIZON_NODES = _ROW_HORIZON + 6
_ROW_CONSTANTS = 55
_ROW_FIELDS = 63
_GOAL_LANES = {"goal", "goal_derivation", "goal_satisfaction"}
_SIZE_FIELDS = ("entities", "tuples", "args", "targets", "history", "overflow")
_OVERFLOW_CAPACITY, _OVERFLOW_INDEX, _OVERFLOW_INPUT = 1, 2, 4
_INVALID = "a row is invalid"


def _error(message: str) -> ValueError:
    return ValueError(f"mymyr: encoding: {message}")


def _torch() -> Any:
    try:
        import torch
    except ImportError:
        raise _error("encoding on the device needs torch (the arrays are torch tensors)") from None
    return torch


def _native() -> Any:
    try:
        from mymyr._core._encoding import DeviceEncoder
    except ImportError:
        raise _error("this mymyr was built without the CUDA backend: encode host arrays") from None
    return DeviceEncoder


# ------------------------------------------------------------------------------------------------ the schema's arrays


class _Layout:
    """Which arrays a batch of an encoder stores, and their dtypes."""

    __slots__ = ("counts", "entity_arrays", "goal_complete", "history", "index", "relations", "targets",
                 "numeric", "numeric_dtype", "numeric_relations", "emits_relations", "simple", "shared",
                 "static_relations", "argumentless_relations")

    def __init__(self, encoder: Any):
        options = encoder.options
        numeric_relations = encoder.relation_lanes.count("numeric")
        # the refusals of the device encoder (DeviceEncoder::check_schema), also where the build has no CUDA backend
        if numeric_relations and "simple" in options["outputs"]:
            raise _error("the device does not encode Simple with numeric relations")
        if options["horizon_mode"] is not None and "simple" in options["outputs"]:
            raise _error("the device does not encode Simple horizons")
        self.numeric_relations = numeric_relations
        self.numeric = self.numeric_relations != 0
        self.numeric_dtype = options["numeric_dtype"]
        self.shared = options["statics"] == "shared"
        self.static_relations = sum(lane == "static" for lane in encoder.relation_lanes)
        self.index = options["index_dtype"]
        self.relations = encoder.num_relations
        # a cell of a relation without argument slots holds at most one tuple: they bound the Simple nodes
        self.argumentless_relations = list(encoder.relation_arities).count(0)
        self.entity_arrays = not encoder.implicit_entities
        self.emits_relations = "relations" in options["outputs"]
        self.simple = "simple" in options["outputs"]
        self.counts = self.emits_relations and bool(options["emit_counts"])
        self.targets = bool(options["target_sources"]) or options["horizon_mode"] is not None
        self.history = bool(options["history"])
        self.goal_complete = any(lane in _GOAL_LANES for lane in encoder.relation_lanes)

    def present(self, name: str) -> bool:
        if name in ("entity_role", "entity_ref"):
            return self.entity_arrays
        if name == "numeric_values":
            return self.numeric
        if name == "relation_counts":
            return self.counts
        if name.startswith("relation_"):
            return self.emits_relations
        if name in _SIMPLE_ARRAYS:
            return self.simple
        if name == "goal_complete":
            return self.goal_complete
        if name.startswith("static_"):
            return self.shared
        if name.startswith("target_"):
            return self.targets
        if name.startswith("history_"):
            return self.history
        return True

    def specs(
        self, torch: Any, graphs: int, entities: int, args: int, targets: int, history: int = 0, *,
        tuples: Optional[int] = None, numeric: int = 0, static_instances: int = 0, static_args: int = 0,
    ) -> dict[str, tuple[tuple[int, ...], Any]]:
        """The shape and dtype of every output, absent arrays included (as empty arrays)."""
        index = torch.int32 if self.index == "int32" else torch.int64
        g, r = graphs, self.relations
        shapes = {
            "task_ids": ((g,), torch.int32),
            "num_objects": ((g,), index),
            "entity_offsets": ((g + 1,), index),
            "entity_role": ((entities,), torch.uint8),
            "entity_ref": ((entities,), index),
            "relation_counts": ((g, r), index),
            "relation_offsets": ((g * r + 1,), index),
            "relation_arg_offsets": ((g * r + 1,), index),
            "relation_args": ((args,), index),
            "goal_complete": ((g,), torch.uint8),
            "target_offsets": ((g + 1,), index),
            "target_entity": ((targets,), index),
            "target_group": ((targets,), torch.uint8),
            "history_offsets": ((g + 1,), index),
            "history_entity": ((history,), index),
            "history_dt": ((history,), torch.int32),
            "node_offsets": ((g + 1,), index),
            "node_label": ((entities + (args + g * self.argumentless_relations if tuples is None else tuples),),
                           torch.int32),
            "edge_offsets": ((g + 1,), index),
            "edge_index": ((2, 2 * args), index),
            "edge_label": ((2 * args,), torch.int32),
            "numeric_values": ((numeric,), torch.float64 if self.numeric_dtype == "float64" else torch.float32),
            "static_task_ids": ((static_instances,), torch.int32),
            "static_offsets": ((static_instances * self.static_relations + 1,), index),
            "static_arg_offsets": ((static_instances * self.static_relations + 1,), index),
            "static_args": ((static_args,), index),
            "static_relations": ((self.static_relations,), index),
            "static_sizes": ((3,), torch.int64),
        }
        specs = {name: (shape if self.present(name) else (0,), dtype) for name, (shape, dtype) in shapes.items()}
        specs["sizes"] = ((len(_SIZE_FIELDS) + int(self.numeric),), torch.int64)
        return specs


_layouts: "weakref.WeakKeyDictionary[Any, _Layout]" = weakref.WeakKeyDictionary()
_devices: "weakref.WeakKeyDictionary[Any, dict[int, Any]]" = weakref.WeakKeyDictionary()
_lock = threading.Lock()


def layout_of(encoder: Any) -> _Layout:
    with _lock:
        layout = _layouts.get(encoder)
        if layout is None:
            layout = _layouts[encoder] = _Layout(encoder)
        return layout


def _device_encoder(encoder: Any, index: int) -> Any:
    """The encoder's native device side on CUDA device ``index``, made (and its context uploaded) on first use."""
    with _lock:
        per_device = _devices.get(encoder)
        if per_device is None:
            per_device = _devices[encoder] = {}
        device = per_device.get(index)
        if device is None:
            device = per_device[index] = _native()(encoder, index)
        return device


# ------------------------------------------------------------------------------------------------ the arguments


def _cuda_tensor(torch: Any, x: Any, name: str, keep: list) -> Any:
    """``x`` as a torch tensor on a CUDA device: a tensor as it is, another DLPack or CUDA-array-interface producer
    wrapped without a copy (and kept alive until the kernels that read it are done)."""
    if isinstance(x, torch.Tensor):
        t = x
    elif hasattr(x, "__dlpack__"):
        t = torch.from_dlpack(x)
        keep.append(t)
    elif hasattr(x, "__cuda_array_interface__"):
        stream = x.__cuda_array_interface__.get("stream")
        if stream == 0:
            raise _error(f"'{name}' has CUDA array interface stream 0: use 1 for the legacy default stream or "
                         "2 for the per-thread default stream")
        t = torch.as_tensor(x)
        keep.append(t)
        if stream is not None and t.device.type == "cuda":
            if stream == 1:
                wait = torch.cuda.default_stream(t.device)
            else:
                wait = torch.cuda.ExternalStream(stream, t.device)
            event = torch.cuda.Event()
            event.record(wait)
            torch.cuda.current_stream(t.device).wait_event(event)
    else:
        raise _error(f"'{name}' must be a CUDA tensor or an array that speaks DLPack or __cuda_array_interface__, not "
                     f"{type(x).__name__}")
    if t.device.type != "cuda":
        raise _error(f"'{name}' is on {t.device}: the rows of one call are all CUDA arrays on one device or all host "
                     f"arrays")
    return t


class _ConstantMasks(tuple):
    """CUDA masks (a tuple of tensors) with the constant literals of their rows (``constants``)."""

    constants: Any = None


def _with_constants(masks: Optional[tuple], given: Any) -> Optional[tuple]:
    """``masks`` (a tuple, or None) with the constant literals of ``given`` (a GoalMasks or LayerMasks) if it has
    any."""
    constants = getattr(given, "constants", None)
    if masks is None or constants is None:
        return masks
    out = _ConstantMasks(masks)
    out.constants = constants
    return out


def _cuda_tuple(torch: Any, items: Any, name: str, keep: list) -> Optional[tuple]:
    """The arrays of ``items`` (a tuple of masks, with None for an absent entry) as CUDA tensors, with the constant
    literals of a GoalMasks or LayerMasks."""
    if items is None:
        return None
    if isinstance(items, str) or not hasattr(items, "__len__"):
        raise _error(f"{name} must be a tuple of arrays")
    return _with_constants(
        tuple(None if x is None else _cuda_tensor(torch, x, f"{name}[{k}]", keep) for k, x in enumerate(items)), items)


def _words(torch: Any, t: Any, name: str) -> Any:
    """``t`` as 64-bit words [rows, width] (a 1-d array is one row): 64-bit integers, or 32-bit halves that are
    reinterpreted as the little-endian halves of words."""
    if t.dtype == torch.int64:
        w = t
    elif t.dtype == torch.uint64:
        w = t.view(torch.int64)
    elif t.dtype in (torch.int32, torch.uint32):
        if t.ndim == 0 or t.shape[-1] % 2 or (t.numel() and t.stride(-1) != 1):
            raise _error(f"'{name}' holds 32-bit halves: it needs an even number of contiguous columns")
        try:
            w = t.view(torch.int64)
        except RuntimeError as error:
            raise _error(f"'{name}' holds 32-bit halves that cannot be read as 64-bit words ({error})") from None
    else:
        raise _error(f"'{name}' must hold 64-bit state words (or 32-bit halves), not {t.dtype}")
    if w.ndim == 1:
        w = w.unsqueeze(0)
    if w.ndim != 2:
        raise _error(f"'{name}' must be an array of state words [N, W], not {w.ndim}-dimensional")
    if w.numel() and w.shape[1] > 1 and w.stride(1) != 1:
        raise _error(f"'{name}' must have contiguous rows")
    if w.numel() and w.data_ptr() % 8:
        raise _error(f"'{name}' must be 8-byte aligned")
    if w.shape[0] > 1 and w.stride(0) < w.shape[1]:
        raise _error(f"'{name}' has rows that overlap")
    return w


def _stride(w: Any) -> int:
    return w.stride(0) if w.shape[0] > 1 else w.shape[1]


def _same_strides(pos: Any, neg: Any, pos_name: str, neg_name: str) -> None:
    """The two masks of a pair (of the same shape) are read with one row stride: they need the same one."""
    if pos.shape[0] > 1 and pos.numel() and pos.stride(0) != neg.stride(0):
        raise _error(f"'{neg_name}' has a row stride of {neg.stride(0)} words, '{pos_name}' of {pos.stride(0)}: the two "
                     f"masks need the same one (make both contiguous)")


def _layer_words(torch: Any, t: Any, name: str, graphs: int, may_share: bool) -> Any:
    """``t`` as 64-bit masks [rows, layers, width]: 64-bit integers, or 32-bit halves reinterpreted as little-endian
    halves of words. A row's layers are contiguous; one row per state, or a single row that all share if ``may_share``."""
    if t.ndim != 3:
        raise _error(f"'{name}' must be an array of masks [N, layers, words], not {t.ndim}-dimensional")
    if t.dtype == torch.int64:
        w = t
    elif t.dtype == torch.uint64:
        w = t.view(torch.int64)
    elif t.dtype in (torch.int32, torch.uint32):
        if t.shape[-1] % 2 or (t.numel() and t.stride(-1) != 1):
            raise _error(f"'{name}' holds 32-bit halves: it needs an even number of contiguous columns")
        try:
            w = t.view(torch.int64)
        except RuntimeError as error:
            raise _error(f"'{name}' holds 32-bit halves that cannot be read as 64-bit words ({error})") from None
    else:
        raise _error(f"'{name}' must hold 64-bit mask words (or 32-bit halves), not {t.dtype}")
    rows, layers, width = w.shape
    if rows != graphs and not (may_share and rows == 1):
        raise _error(f"'{name}' has {rows} rows: one per state ({graphs})" + (" or a single shared row" if may_share else ""))
    if w.numel() and (width > 1 and w.stride(2) != 1 or layers > 1 and w.stride(1) != width):
        raise _error(f"'{name}' must have contiguous layers")
    if w.numel() and w.data_ptr() % 8:
        raise _error(f"'{name}' must be 8-byte aligned")
    if w.numel() and rows > 1 and w.stride(0) < layers * width:
        raise _error(f"'{name}' has rows that overlap")
    return w


class _Rows:
    """The checked rows of a call: the device pointers and strides the native side takes."""

    __slots__ = ("device", "fields", "graphs", "history_layers", "tensors")

    def __init__(self, torch: Any, states: Any, task_ids: Any, derived: Any, goals: Any, subgoals: Any = None,
                 history: Any = None):
        if states.device.type != "cuda":
            raise _error(f"'states' is on {states.device}: the rows must be CUDA tensors")
        w = _words(torch, states, "states")
        self.device = w.device
        graphs = w.shape[0]
        self.graphs = graphs
        tensors = [w]
        ids_ptr = 0
        if task_ids is not None:
            ids = task_ids
            if ids.ndim == 0:
                ids = ids.reshape(1)
            if ids.dtype != torch.int32 or ids.ndim != 1 or ids.shape[0] != graphs or not ids.is_contiguous():
                raise _error(f"'task_ids' must be a contiguous int32 tensor of {graphs} entries, one per row, not a "
                             f"{ids.dtype} tensor of shape {list(ids.shape)} (convert with .to(torch.int32))")
            self._same_device(ids, "task_ids")
            ids_ptr = ids.data_ptr()
            tensors.append(ids)
        derived_ptr = derived_stride = 0
        if derived is not None:
            d = _words(torch, derived, "derived")
            if d.shape[0] != graphs:
                raise _error(f"'derived' has {d.shape[0]} rows, the states {graphs}")
            self._same_device(d, "derived")
            derived_ptr, derived_stride = d.data_ptr(), _stride(d)
            tensors.append(d)
        goal = [0] * 8  # goal_pos, goal_neg, stride, words, derived_pos, derived_neg, stride, words
        if goals is not None:
            if isinstance(goals, str) or not hasattr(goals, "__len__") or len(goals) not in (2, 4):
                raise _error("goals must be (goal_pos, goal_neg) or (goal_pos, goal_neg, goal_derived_pos, "
                             "goal_derived_neg)")
            names = ("goals[0] (goal_pos)", "goals[1] (goal_neg)", "goals[2] (goal_derived_pos)", "goals[3] (goal_derived_neg)")
            masks = []
            for k, mask in enumerate(goals):
                m = _words(torch, mask, names[k])
                if m.shape[0] != 1 and m.shape[0] != graphs:
                    raise _error(f"{names[k]} has {m.shape[0]} rows: one per state ({graphs}) or a single shared row")
                if k % 2 == 1 and (m.shape != masks[-1].shape):
                    raise _error(f"{names[k]} has another shape than {names[k - 1]}")
                self._same_device(m, names[k])
                masks.append(m)
                tensors.append(m)
            for pair in range(len(masks) // 2):
                pos, neg = masks[2 * pair], masks[2 * pair + 1]
                _same_strides(pos, neg, names[2 * pair], names[2 * pair + 1])
                shared = pos.shape[0] == 1 and graphs != 1
                at = 4 * pair
                goal[at:at + 4] = [pos.data_ptr(), neg.data_ptr(), 0 if shared else _stride(pos), pos.shape[1]]
        sub = self._subgoals(torch, subgoals, graphs, tensors)
        hist, self.history_layers = self._history(torch, history, graphs, tensors)
        constants = [*self._constants(torch, getattr(goals, "constants", None), "goals.constants", graphs, tensors),
                     *self._constants(torch, getattr(subgoals, "constants", None), "subgoals.constants", graphs,
                                      tensors)]
        self.tensors = tensors
        self.fields = [graphs, w.data_ptr() if w.numel() else 0, _stride(w), w.shape[1], ids_ptr, derived_ptr,
                       derived_stride, *goal[:4], *goal[4:], *sub, *hist]
        self.fields += [0] * (_ROW_CONSTANTS - _ROW_ACTIONS)  # no expansion rows, no horizon view
        self.fields += constants
        if len(self.fields) != _ROW_FIELDS:
            raise AssertionError("the row fields have a fixed length")

    def _same_device(self, t: Any, name: str) -> None:
        if t.device != self.device:
            raise _error(f"'{name}' is on {t.device}, the states on {self.device}")

    def _pair(self, torch: Any, items: Any, at: int, names: tuple, graphs: int, may_share: bool, tensors: list) -> tuple:
        """The masks ``items[at]`` and ``items[at + 1]`` as (pos, neg, row stride) (stride 0: shared by all rows)."""
        pair = []
        for k in (at, at + 1):
            if items[k] is None:
                raise _error(f"'{names[k]}' must be an array, not None")
            m = _layer_words(torch, items[k], names[k], graphs, may_share)
            self._same_device(m, names[k])
            tensors.append(m)
            pair.append(m)
        pos, neg = pair
        if pos.shape != neg.shape:
            raise _error(f"'{names[at + 1]}' has another shape than '{names[at]}'")
        _same_strides(pos, neg, names[at], names[at + 1])
        shared = pos.shape[0] == 1 and graphs != 1
        stride = 0 if shared else (pos.stride(0) if pos.shape[0] > 1 else pos.shape[1] * pos.shape[2])
        return pos, neg, stride

    def _layers(self, torch: Any, items: Any, names: tuple, derived_at: int, graphs: int, may_share: bool,
                tensors: list) -> list[int]:
        """The fields of a set of layers: pos, neg, stride, layers, words, then the derived pointers, stride and
        words (0 without derived masks)."""
        pos, neg, stride = self._pair(torch, items, 0, names, graphs, may_share, tensors)
        fields = [pos.data_ptr() if pos.numel() else 0, neg.data_ptr() if neg.numel() else 0, stride, pos.shape[1],
                  pos.shape[2], 0, 0, 0, 0]
        if len(items) > derived_at:
            dpos, dneg, dstride = self._pair(torch, items, derived_at, names, graphs, may_share, tensors)
            if dpos.shape[1] != pos.shape[1]:
                raise _error(f"'{names[derived_at]}' has {dpos.shape[1]} layers, '{names[0]}' has {pos.shape[1]}")
            fields[5:] = [dpos.data_ptr() if dpos.numel() else 0, dneg.data_ptr() if dneg.numel() else 0, dstride,
                          dpos.shape[2]]
        return fields

    def _constants(self, torch: Any, constants: Any, name: str, graphs: int, tensors: list) -> list[int]:
        """The fields of constant literals (rows, row stride, capacity, width; zeros without): an int32 array
        ``[N, K, C]`` with one row per state or a single shared row, on the rows' device (a host array is copied to
        it)."""
        if constants is None:
            return [0] * 4
        if isinstance(constants, torch.Tensor):
            c = constants
        elif hasattr(constants, "__cuda_array_interface__") or (
                hasattr(constants, "__dlpack_device__") and constants.__dlpack_device__()[0] == 2):
            c = torch.from_dlpack(constants)
        else:
            c = torch.as_tensor(np.asarray(constants))
        if c.device != self.device:
            if c.device.type != "cpu":
                raise _error(f"'{name}' is on {c.device}, the states on {self.device}")
            if torch.cuda.is_current_stream_capturing():
                raise _error(f"'{name}' is a host array: give CUDA constant literals under capture")
            c = c.to(self.device)
        if c.dtype != torch.int32 or c.ndim != 3:
            raise _error(f"'{name}' must be an int32 array [N, K, C], not a {c.dtype} array of shape {list(c.shape)}")
        rows, capacity, width = c.shape
        if rows != graphs and rows != 1:
            raise _error(f"'{name}' has {rows} rows: one per state ({graphs}) or a single shared row")
        if capacity == 0 or c.numel() == 0:
            return [0] * 4
        if not c[0].is_contiguous():
            c = c.contiguous()
        tensors.append(c)
        shared = rows == 1 and graphs != 1
        return [c.data_ptr(), 0 if shared else (c.stride(0) if rows > 1 else capacity * width), capacity, width]

    def _subgoals(self, torch: Any, subgoals: Any, graphs: int, tensors: list) -> list[int]:
        if subgoals is None:
            return [0] * 9
        names = ("subgoals[0] (pos)", "subgoals[1] (neg)", "subgoals[2] (derived_pos)", "subgoals[3] (derived_neg)")
        if isinstance(subgoals, str) or not hasattr(subgoals, "__len__") or len(subgoals) not in (2, 4):
            raise _error("subgoals must be (pos, neg) or (pos, neg, derived_pos, derived_neg)")
        return self._layers(torch, subgoals, names, 2, graphs, True, tensors)

    def _history(self, torch: Any, history: Any, graphs: int, tensors: list) -> tuple[list[int], int]:
        """The fields of the history (its masks, the pointers of dt and count) and the steps of its longest row."""
        if history is None:
            return [0] * 11, 0
        names = ("history[0] (pos)", "history[1] (neg)", "history[2] (dt)", "history[3] (count)",
                 "history[4] (derived_pos)", "history[5] (derived_neg)")
        if isinstance(history, str) or not hasattr(history, "__len__") or len(history) not in (3, 4, 6):
            raise _error("history must be (pos, neg, dt), (pos, neg, dt, count) or (pos, neg, dt, count, derived_pos, "
                         "derived_neg)")
        masks = self._layers(torch, (history[0], history[1], *history[4:]), (names[0], names[1], *names[4:]), 2, graphs,
                             False, tensors)
        steps = masks[3]
        ptrs = []
        for k, shape in ((2, (graphs, steps)), (3, (graphs,))):
            x = history[k] if k < len(history) else None
            if x is None:
                if k == 2 and graphs * steps != 0:
                    raise _error(f"'{names[2]}' must be an array, not None")
                ptrs.append(0)
                continue
            t = x
            if t.dtype != torch.int32 or tuple(t.shape) != shape or not t.is_contiguous():
                raise _error(f"'{names[k]}' must be a contiguous int32 tensor of shape {list(shape)}, not a {t.dtype} "
                             f"tensor of shape {list(t.shape)} (convert with .to(torch.int32))")
            self._same_device(t, names[k])
            tensors.append(t)
            ptrs.append(t.data_ptr() if t.numel() else 0)
        return [*masks, *ptrs], steps


def _capacities(value: Any, layout: _Layout) -> tuple[int, int, int, int, int, int, int, int]:
    """The (graphs, entities, args, targets, history, numeric, static_instances, static_args) capacities of a
    ``capacities`` dict."""
    if not isinstance(value, dict):
        raise _error(f"capacities must be a dict, not {type(value).__name__}")
    keys = {"entities", "graphs", "args", "targets", "history", "numeric", "static_instances", "static_args"}
    unknown = set(value) - keys
    if unknown:
        raise _error(f"capacities has the unknown key '{sorted(unknown, key=str)[0]}'; the keys are 'entities', "
                     f"'graphs', 'args', 'targets', 'history', 'numeric', 'static_instances' and 'static_args'")
    if not {"entities", "graphs", "args"} <= set(value):
        raise _error("capacities needs the keys 'entities', 'graphs' and 'args'")
    if "targets" not in value and layout.targets:
        raise _error("the schema has targets: capacities needs the key 'targets' as well")
    if "history" not in value and layout.history:
        raise _error("the schema has history: capacities needs the key 'history' as well")
    if "numeric" not in value and layout.numeric:
        raise _error("the schema has numeric values: capacities needs the key 'numeric' as well")
    if layout.shared and not {"static_instances", "static_args"} <= set(value):
        raise _error("shared statics need capacities 'static_instances' and 'static_args'")
    if not layout.shared and (value.get("static_instances", 0) or value.get("static_args", 0)):
        raise _error("repeated statics take no static capacities")
    out = []
    for key in ("graphs", "entities", "args", "targets", "history", "numeric", "static_instances", "static_args"):
        n = value.get(key, 0)
        if isinstance(n, bool) or not isinstance(n, int):
            if hasattr(n, "__index__") and not isinstance(n, bool):
                n = n.__index__()
            else:
                raise _error(f"capacities['{key}'] must be an int, not {type(n).__name__}")
        if n < 0:
            raise _error(f"capacities['{key}'] must be at least 0, got {n}")
        out.append(n)
    return tuple(out)


# ------------------------------------------------------------------------------------------------ pending inputs

_pending: list = []
_pending_lock = threading.Lock()


def _retain(torch: Any, keep: list, device: Any) -> None:
    """Keeps the wrappers of foreign arrays alive until the work enqueued so far has completed (their owners free the
    memory when the wrappers go). During capture, torch owns temporary allocations through the graph, and callers
    keep external input owners alive across every replay."""
    if torch.cuda.is_current_stream_capturing():
        return
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(device))
    with _pending_lock:
        _pending.append((event, keep))


def _drain(torch: Any) -> None:
    """Drops the wrappers whose work has completed (not during a capture: querying an event would invalidate it)."""
    with _pending_lock:
        if _pending and not torch.cuda.is_current_stream_capturing():
            _pending[:] = [item for item in _pending if not item[0].query()]


# ------------------------------------------------------------------------------------------------ encoding


def _pointers(tensors: dict[str, Any]) -> list[int]:
    names = (*_OUTPUTS, *_RELATION_SCRATCH)
    pointers = [tensors[name].data_ptr() if tensors.get(name) is not None and tensors[name].numel() else 0
                for name in names]
    edge = tensors.get("edge_index")
    pointers.append(edge.shape[1] if edge is not None and edge.ndim == 2 else 0)
    return pointers


def _outputs(torch: Any, specs: dict, device: Any) -> dict[str, Any]:
    """Independent uninitialized outputs; the fill kernel writes the used entries and the tails it zeroes."""
    return {name: torch.empty(shape, dtype=dtype, device=device) for name, (shape, dtype) in specs.items()}


def _relation_scratch(torch: Any, out: dict, layout: _Layout, graphs: int, args: int, device: Any) -> None:
    """Private relation storage used while filling a Simple-only graph, retained through the launch."""
    if layout.emits_relations:
        return
    index = torch.int32 if layout.index == "int32" else torch.int64
    cells = graphs * layout.relations + 1
    for name, entries in zip(_RELATION_SCRATCH, (cells, cells, args), strict=True):
        out[name] = torch.empty(entries, dtype=index, device=device)


def _scratch(torch: Any, native: Any, stream: int, graphs: int, device: Any, derived: bool) -> Any:
    """Per-call scratch, owned by the graph's private allocator pool during capture; `derived`: the rows' derived words
    are given, so nothing is evaluated."""
    return torch.empty(native.scratch_bytes(stream, graphs, derived), dtype=torch.uint8, device=device)


def encode_arrays(
    encoder: Any,
    states: Any,
    task_ids: Any,
    derived: Any,
    goals: Any,
    capacities: Optional[tuple[int, int, int, int, int, int, int, int]],
    keep: Optional[list] = None,
    subgoals: Any = None,
    history: Any = None,
    actions: Any = None,
    horizon: Any = None,
) -> dict[str, Any]:
    """The arrays of checked tensors, by name (absent arrays empty), and int64 sizes: entities, tuples, args, targets,
    history, overflow flags, and the numeric count when stored. Sizes are None without capacities."""
    torch = _torch()
    layout = layout_of(encoder)
    if horizon is None:
        rows = _Rows(torch, states, task_ids, derived, goals, subgoals, history)
        if actions is not None:
            from mymyr.encoding._device_horizon import bind_actions
            bind_actions(encoder, rows, actions)
            keep = [*(keep or []), actions, *rows.tensors]
    else:
        from mymyr.encoding._device_horizon import horizon_rows
        rows = horizon_rows(encoder, horizon, derived, goals, subgoals)
    try:
        return _encode_rows(encoder, rows, capacities, keep)
    except ValueError as error:
        state = None if actions is None or _INVALID not in str(error) else _misordered(torch, encoder, actions,
                                                                                         task_ids)
        if state is None:
            raise
        raise _error(f"the action rows of state {state} are not in canonical order: strictly ascending in (schema, "
                     "binding), and in index (from 0) for an expansion with an index") from None


def _misordered(torch: Any, encoder: Any, actions: Any, task_ids: Any) -> Optional[int]:
    """The first state whose expansion rows (device arrays) are not in canonical order, or None: the diagnosis of an
    exact call that found an invalid row (it waits for the device). Rows that are not valid labels (a schema or an
    object the instance does not have, a parent that is not their state) are the cause first, as on the host: then
    None."""
    offsets = actions.offsets.to(torch.int64)
    total = int(offsets[-1])
    counts = (offsets[1:] - offsets[:-1]).clamp(min=0)
    if total < 1 or int(counts.sum()) != total or total > actions.schema.shape[0]:
        return None
    parent = torch.repeat_interleave(torch.arange(counts.shape[0], device=offsets.device), counts)
    if not _valid_labels(torch, encoder, actions, task_ids, parent, total):
        return None
    key = torch.cat((actions.schema[:total, None], actions.binding[:total]), dim=1).to(torch.int64)
    step = key[1:] - key[:-1]
    changed = step != 0
    first = changed.to(torch.int8).argmax(dim=1, keepdim=True)
    ascending = changed.any(dim=1) & (step.gather(1, first).squeeze(1) > 0)
    same = parent[1:] == parent[:-1]
    bad = torch.zeros(total, dtype=torch.bool, device=offsets.device)
    bad[1:] = same & ~ascending
    index = getattr(actions, "index", None)
    if index is not None:
        index = index[:total].to(torch.int64)
        bad[1:] |= same & (index[1:] <= index[:-1])
        bad |= index < 0
    at = torch.nonzero(bad)
    return int(parent[at[0, 0]]) if at.shape[0] else None


def _valid_labels(torch: Any, encoder: Any, actions: Any, task_ids: Any, parent: Any, total: int) -> bool:
    """Whether the first ``total`` rows are labels of their states' instances: schemas of the table, parents that are
    their states, and objects of the instance in the schema's arguments."""
    from mymyr.encoding._device_actions import table_of

    table = table_of(encoder)
    device = parent.device
    schema = actions.schema[:total].to(torch.int64)
    if bool(((schema < 0) | (schema >= table.num_schemas)).any()):
        return False
    given = getattr(actions, "parent", None)
    if given is not None and not bool((given[:total].to(torch.int64) == parent).all()):
        return False
    if task_ids is None:
        instance = torch.zeros_like(parent)
    else:
        ids = torch.as_tensor(task_ids, device=device).to(torch.int64)
        instance = ids[parent]
        if bool(((instance < 0) | (instance >= len(table))).any()):
            return False
    objects = torch.as_tensor(list(table.num_objects), dtype=torch.int64, device=device)[instance]
    arity = torch.as_tensor(list(table.schema_arities), dtype=torch.int64, device=device)[schema]
    binding = actions.binding[:total].to(torch.int64)
    used = torch.arange(binding.shape[1], device=device) < arity[:, None]
    return not bool((used & ((binding < 0) | (binding >= objects[:, None]))).any())


def _encode_rows(encoder: Any, rows: Any, capacities: Any, keep: Any = None, *, graph_count: Any = None) -> dict[str, Any]:
    torch = _torch()
    _drain(torch)
    layout = layout_of(encoder)
    device = rows.device
    native = _device_encoder(encoder, device.index)
    from mymyr.rl.torch._ops import _stream

    stream = _stream(device)
    horizon = rows.fields[_ROW_HORIZON] != 0
    scratch_rows = max(rows.graphs, rows.fields[_ROW_HORIZON_NODES]) if horizon else rows.graphs
    out: dict[str, Any] = {}
    if capacities is not None:
        graphs, entities, args, targets, steps, numeric, static_instances, static_args = capacities
        if rows.graphs > graphs:
            raise _error("the batch has more graphs than the output capacity")
        specs = layout.specs(torch, rows.graphs, entities, args, targets, steps, numeric=numeric,
                             static_instances=static_instances, static_args=static_args)
        out = _outputs(torch, specs, device)
        out["_entity_capacity"] = entities
        _relation_scratch(torch, out, layout, rows.graphs, args, device)
        scratch = _scratch(torch, native, stream, scratch_rows, device, rows.fields[_ROW_DERIVED] != 0)
        native.encode(stream, rows.fields, _pointers(out),
                      [out["task_ids"].shape[0], entities, args, targets, steps, numeric, static_instances, static_args],
                      scratch.data_ptr(), scratch.numel(), 0 if graph_count is None else graph_count.data_ptr())
    else:

        def allocate(
            entities: int, tuples: int, args: int, targets: int, steps: int, numeric: int, flags: int,
            static_instances: int, static_tuples: int, static_args: int,
        ) -> list[int]:
            if flags & _OVERFLOW_INPUT:
                raise _error(_INVALID + ": a task id outside the table, a state row, goal mask, subgoal layer or "
                             "history step narrower than its instance, invalid expansion labels, horizon offsets or "
                             "endpoints, a history count outside [0, steps], a bit that its instance has no atom "
                             "for, or a constant literal that it cannot encode")
            if flags & _OVERFLOW_INDEX:
                raise OverflowError("mymyr: encoding: the batch needs more indices than its index type holds "
                                    f"(entities {entities}, tuples {tuples}, args {args}, targets {targets}, "
                                    f"history {steps}): use index_dtype='int64'")
            specs = layout.specs(torch, rows.graphs, entities, args, targets, steps, tuples=tuples, numeric=numeric,
                                 static_instances=static_instances, static_args=static_args)
            out["_entity_capacity"] = entities
            for name, (shape, dtype) in specs.items():
                if name != "sizes":
                    out[name] = torch.empty(shape, dtype=dtype, device=device)
            _relation_scratch(torch, out, layout, rows.graphs, args, device)
            return [out["task_ids"].shape[0], *_pointers(out)]

        native.encode_exact(stream, rows.fields, allocate)
        out["sizes"] = None
    _retain(torch, [*rows.tensors, *(keep or [])], device)
    return out


def encode(
    encoder: Any,
    states: Any,
    task_ids: Any,
    derived: Any,
    goals: Any,
    actions: Any,
    subgoals: Any,
    history: Any,
    framework: Any,
    threads: Any,
    pool: Any,
    capacities: Any,
) -> "DeviceFlatBatch":
    """``FlatRelationEncoder.encode`` for arguments that are CUDA arrays."""
    torch = _torch()
    if pool is not None:
        raise _error("pool is a pool of host threads: it has no meaning for rows on the device")
    if framework is not None and not (isinstance(framework, str) and framework in ("torch", "pytorch")):
        raise _error(f"device rows are encoded into torch tensors: framework must be 'torch' or None, got {framework!r}")
    layout = layout_of(encoder)
    caps = _capacities(capacities, layout) if capacities is not None else None
    _drain(torch)
    keep: list = []
    if not (hasattr(states, "__dlpack__") or hasattr(states, "__cuda_array_interface__")):
        raise _error(f"'states' must be a CUDA array when other arguments are: got {type(states).__name__}")
    s = _cuda_tensor(torch, states, "states", keep)
    ids = None if task_ids is None else _cuda_tensor(torch, task_ids, "task_ids", keep)
    d = None if derived is None else _cuda_tensor(torch, derived, "derived", keep)
    g = None
    if goals is not None:
        if isinstance(goals, str) or not hasattr(goals, "__len__"):
            raise _error("goals must be (goal_pos, goal_neg) or (goal_pos, goal_neg, goal_derived_pos, goal_derived_neg)")
        g = _with_constants(tuple(_cuda_tensor(torch, mask, f"goals[{k}]", keep) for k, mask in enumerate(goals)),
                            goals)
    sub = _cuda_tuple(torch, subgoals, "subgoals", keep)
    hist = _cuda_tuple(torch, history, "history", keep)
    arrays = encode_arrays(encoder, s, ids, d, g, caps, keep, sub, hist, actions=actions)
    return DeviceFlatBatch(encoder, arrays, exact=caps is None)


# ------------------------------------------------------------------------------------------------ workspaces


def _device_index(torch: Any, device: Any) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, torch.device):
        if device.type != "cuda":
            raise _error(f"device must be a CUDA device, not {device}")
        return torch.cuda.current_device() if device.index is None else device.index
    if isinstance(device, int) and not isinstance(device, bool) and device >= 0:
        return device
    raise _error(f"device must be a CUDA device index or a torch.device, got {device!r}")


def _stream_pointer(torch: Any, stream: Any, index: int) -> int:
    if stream is None:
        from mymyr.rl.torch._ops import _stream

        return _stream(torch.device("cuda", index))
    handle = getattr(stream, "cuda_stream", stream)
    if isinstance(handle, bool) or not isinstance(handle, int):
        raise _error(f"stream must be a torch.cuda.Stream or an int cudaStream_t, got {stream!r}")
    return handle


def reserve(encoder: Any, graphs: Any, device: Any, stream: Any) -> None:
    torch = _torch()
    if isinstance(graphs, bool) or not isinstance(graphs, int) or graphs < 0:
        raise _error(f"graphs must be an int of at least 0, got {graphs!r}")
    index = _device_index(torch, device)
    _device_encoder(encoder, index).reserve(_stream_pointer(torch, stream, index), graphs)


def release(encoder: Any, device: Any, stream: Any) -> None:
    torch = _torch()
    index = _device_index(torch, device)
    with _lock:
        native = _devices.get(encoder, {}).get(index)
    if native is not None:
        native.release(None if stream is None else _stream_pointer(torch, stream, index))


def workspace_state(encoder: Any, device: Any = None, stream: Any = None) -> dict[str, int]:
    """The graphs the workspace of a stream holds, its scratch bytes, the reservations made on it and the host waits its
    calls made (a diagnostic of the workspace reuse and of the one-wait rule)."""
    torch = _torch()
    index = _device_index(torch, device)
    graphs, scratch, reservations, waits = _device_encoder(encoder, index).state(_stream_pointer(torch, stream, index))
    return {"graphs": graphs, "scratch_bytes": scratch, "reservations": reservations, "host_waits": waits}


# ------------------------------------------------------------------------------------------------ the batch


class _Host:
    """One graph's arrays on the host, with graph-local offsets for the naming accessors."""

    def __init__(self, batch: "DeviceFlatBatch", graph: int):
        import numpy as np

        if not 0 <= graph < batch.num_graphs:
            raise IndexError(f"mymyr: encoding: graph {graph} is out of range for a batch of {batch.num_graphs} graphs")
        self.schema = batch.schema
        self.num_graphs = 1
        self.num_relations = batch.num_relations
        self.implicit_entities = batch.implicit_entities
        self.sizes = batch.sizes
        if self.sizes is not None and self.sizes["overflow"]:
            return
        torch = _torch()
        self.task_ids = batch.task_ids[graph:graph + 1].cpu().numpy()
        self.num_objects = batch.num_objects[graph:graph + 1].cpu().numpy()
        begin, end = batch.entity_offsets[graph:graph + 2].tolist()
        self.entity_offsets = np.array([0, end - begin], dtype=np.int64)
        if not self.implicit_entities:
            self.entity_role = batch.entity_role[begin:end].cpu().numpy()
            self.entity_ref = batch.entity_ref[begin:end].cpu().numpy()
        r = self.num_relations
        if batch._options["layout"] == "relation":
            segments = torch.arange(r, device=batch.device) * batch.num_graphs + graph
            bounds = torch.stack((batch.relation_arg_offsets[segments],
                                  batch.relation_arg_offsets[segments + 1]), dim=1).tolist()
            args = torch.cat([batch.relation_args[a:b] for a, b in bounds]) if r else batch.relation_args[:0]
            lengths = [b - a for a, b in bounds]
            tuples = (batch.relation_offsets[segments + 1] - batch.relation_offsets[segments]).tolist()
        else:
            offsets = batch.relation_arg_offsets[graph * r:(graph + 1) * r + 1].tolist()
            args = batch.relation_args[offsets[0]:offsets[-1]]
            lengths = np.diff(offsets)
            tuples = np.diff(batch.relation_offsets[graph * r:(graph + 1) * r + 1].tolist())
        # graph-local offsets: the tuples of a relation without argument slots have no entries in relation_args
        self.relation_offsets = np.concatenate(([0], np.cumsum(tuples, dtype=np.int64)))
        self.relation_arg_offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
        self.relation_args = args.cpu().numpy() - begin
        if batch._layout.shared:
            ids = batch.static_task_ids
            blocks = ids.numel() if self.sizes is None else self.sizes["static_instances"]
            block = int(torch.searchsorted(ids[:blocks], int(self.task_ids[0])).item())
            count = batch._layout.static_relations
            offsets = batch.static_arg_offsets[block * count:(block + 1) * count + 1].cpu().numpy()
            tuples = batch.static_offsets[block * count:(block + 1) * count + 1].cpu().numpy()
            self.static_task_ids = self.task_ids.copy()
            self.static_relations = batch.static_relations.cpu().numpy()
            self.static_offsets = tuples - tuples[0]
            self.static_arg_offsets = offsets - offsets[0]
            self.static_args = batch.static_args[int(offsets[0]):int(offsets[-1])].cpu().numpy()
            if self.sizes is not None:
                self.sizes = {**self.sizes, "static_instances": 1, "static_args": len(self.static_args)}
        if batch._layout.history:
            first, last = batch.history_offsets[graph:graph + 2].tolist()
            self.history_offsets = np.array([0, last - first], dtype=np.int64)
            self.history_entity = batch.history_entity[first:last].cpu().numpy() - begin
            self.history_dt = batch.history_dt[first:last].cpu().numpy()


def _signed(fingerprint: int) -> int:
    return fingerprint - (1 << 64) if fingerprint >= 1 << 63 else fingerprint


class DeviceFlatBatch:
    """The graphs of a batch of states encoded on the device: what a ``FlatBatch`` is, with the arrays in CUDA memory as
    torch tensors. ``FlatRelationEncoder.encode`` returns it for rows on the device. The array names, shapes and dtypes are
    those of ``FlatBatch``; an array that the schema does not store raises ``AttributeError``. Encoded with ``capacities``,
    the variable-size arrays have the capacities' sizes and the true sizes are in ``device_sizes`` (a tensor: reading it
    on the host waits for the device); ``sizes`` reads it. ``entity_names`` and ``describe`` of the encoder copy only the requested graph
    to the host (and wait for the device)."""

    def __init__(self, encoder: Any, arrays: dict[str, Any], exact: bool):
        self._encoder = encoder
        self._device_graphs = None
        self._schema = encoder.schema
        self._fingerprint = encoder.schema_fingerprint
        self._table_fingerprint = encoder.table_fingerprint
        self._options = encoder.options
        layout = layout_of(encoder)
        self._layout = layout
        self._arrays = {name: arrays[name] for name in _ARRAYS if layout.present(name)}
        self._sizes = None if exact else arrays["sizes"]
        self._static_sizes = arrays["static_sizes"]
        self._graphs = self._arrays["task_ids"].shape[0]
        self._entity_capacity = arrays.get("_entity_capacity")

    @property
    def nbytes_by_array(self) -> dict[str, int]:
        """Output array names and payload bytes, including unused capacity and device counters. Reads only host
        metadata: no device wait, tensor export or device allocation. Excludes inputs, schema and scratch."""
        arrays = dict(self._arrays)
        for name, value in (("device_sizes", self._sizes), ("device_static_sizes", self.device_static_sizes),
                            ("device_graphs", self._device_graphs)):
            if value is not None:
                arrays[name] = value
        return {name: int(value.nbytes) for name, value in arrays.items()}

    @property
    def nbytes(self) -> int:
        """The sum of ``nbytes_by_array``; includes unused capacity, excludes allocator overhead. Does not wait."""
        return sum(self.nbytes_by_array.values())

    @property
    def device_static_sizes(self) -> Optional[Any]:
        """The device counts [instances, tuples, args] of the shared static blocks, or None."""
        return self._static_sizes if self._layout.shared else None

    @property
    def device(self) -> Any:
        """The ``torch.device`` of the arrays."""
        return self._arrays["task_ids"].device

    @property
    def num_graphs(self) -> int:
        """The number of graphs (states) in the batch."""
        return self._graphs if self._device_graphs is None else int(self._device_graphs.item())

    @property
    def device_graphs(self) -> Any:
        """A device count for a dynamically sized stream batch; None for an ordinary row batch."""
        return self._device_graphs

    @property
    def num_relations(self) -> int:
        """The number of relations of the schema the batch was encoded with."""
        return self._schema.num_relations

    @property
    def schema(self) -> Any:
        """The EncodingSchema the batch was encoded for."""
        return self._schema

    @property
    def implicit_entities(self) -> bool:
        """True if the batch has no entity_role and entity_ref (see ``FlatBatch.implicit_entities``)."""
        return self._schema.implicit_entities

    @property
    def device_sizes(self) -> Optional[Any]:
        """After an encoding with capacities: int64 [6] with entities, tuples, args, targets, history steps and overflow
        flags (1: a capacity, 2: the index type, 4: an invalid row). Numeric batches append the numeric value count
        (int64 [7]). Reading the tensor on the host waits; None for exact calls."""
        return self._sizes

    @property
    def sizes(self) -> Optional[dict[str, Any]]:
        """After an encoding with capacities: the true entities, tuples, args and targets of the batch (and history steps, for
        a schema with history), and whether it overflowed
        (``overflow``: the batch is all zero) because of the capacities (``capacity_overflow``), the index type
        (``index_overflow``) or an invalid row (``invalid_input``: a task id outside the table, a row or goal mask
        narrower than its instance, an atom bit its instance has no slot for); None otherwise. It reads
        ``device_sizes`` on the host, which waits for the device."""
        if self._sizes is None:
            return None
        fields = (*_SIZE_FIELDS, "numeric") if self._layout.numeric else _SIZE_FIELDS
        words = self._sizes
        if self._layout.shared:
            # one transfer for both counts
            fields = (*fields, "static_instances", "static_tuples", "static_args")
            words = _torch().cat((words, self._static_sizes))
        values = dict(zip(fields, words.tolist(), strict=True))
        if not self._layout.history:
            del values["history"]  # a schema without history has the keys of a batch without it
        flags = values["overflow"]
        values["overflow"] = flags != 0
        values["capacity_overflow"] = bool(flags & _OVERFLOW_CAPACITY)
        values["index_overflow"] = bool(flags & _OVERFLOW_INDEX)
        values["invalid_input"] = bool(flags & _OVERFLOW_INPUT)
        return values

    def as_dict(self, framework: Optional[str] = None) -> dict[str, Any]:
        """The scalars (version, num_graphs, num_relations, layout, index_dtype, schema_fingerprint, table_fingerprint,
        statics) and the arrays of the batch. framework: ``"torch"`` or ``"dlpack"`` (default; the tensors, which speak DLPack) or ``"numpy"``
        (copies to the host and waits for the device)."""
        scalars = {
            "version": 1,
            "num_graphs": self.num_graphs,
            "num_relations": self.num_relations,
            "layout": 1 if self._options["layout"] == "relation" else 0,
            "index_dtype": 1 if self._options["index_dtype"] == "int64" else 0,
            "schema_fingerprint": _signed(self._fingerprint),
            "table_fingerprint": _signed(self._table_fingerprint),
            "statics": int(self._layout.shared),
        }
        if self._layout.numeric:
            scalars["numeric_dtype"] = int(self._layout.numeric_dtype == "float64")
            scalars["numeric_relations"] = self._layout.numeric_relations
        if self._layout.simple:
            scalars["simple_node_labels"] = len(self._schema.simple_label_names)
            scalars["simple_edge_labels"] = len(self._schema.simple_edge_label_names)
        arrays = self._arrays
        if self._device_graphs is not None:
            g = scalars["num_graphs"]
            sizes = self.sizes
            arrays = dict(arrays)
            extents = {"task_ids": g, "num_objects": g, "entity_offsets": g + 1, "goal_complete": g,
                       "target_offsets": g + 1, "relation_offsets": g * self.num_relations + 1,
                       "relation_arg_offsets": g * self.num_relations + 1, "entity_role": sizes["entities"],
                       "entity_ref": sizes["entities"], "relation_args": sizes["args"],
                       "target_entity": sizes["targets"], "target_group": sizes["targets"],
                       "history_offsets": g + 1, "history_entity": sizes.get("history", 0),
                       "history_dt": sizes.get("history", 0),
                       "numeric_values": sizes.get("numeric", 0),
                       "node_offsets": g + 1, "node_label": sizes["entities"] + sizes["tuples"],
                       "edge_offsets": g + 1, "edge_label": 2 * sizes["args"]}
            if self._layout.shared:
                blocks = sizes["static_instances"] * self._layout.static_relations + 1
                extents.update(static_task_ids=sizes["static_instances"], static_offsets=blocks,
                               static_arg_offsets=blocks, static_args=sizes["static_args"],
                               static_relations=self._layout.static_relations)
            arrays = {key: (value[:g] if key == "relation_counts" else
                            value[:, :2 * sizes["args"]] if key == "edge_index" else value[:extents[key]])
                      for key, value in arrays.items()}
        name = "torch" if framework is None else framework
        if name in ("torch", "pytorch", "dlpack", "array_api"):
            return {**scalars, **arrays}
        if name in ("numpy", "np"):
            torch = _torch()
            torch.cuda.current_stream(self.device).synchronize()
            return {**scalars, **{key: t.cpu().numpy() for key, t in arrays.items()}}
        raise _error(f"a device batch exports to 'torch', 'dlpack' or 'numpy', not {framework!r}")

    def __getattr__(self, name: str) -> Any:
        if name in _ARRAYS:
            raise AttributeError(f"mymyr: encoding: this batch has no {name}")
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @classmethod
    def from_arrays(cls, encoder: Any, arrays: Any, exact: bool = False, *,
                    entity_capacity: Optional[int] = None) -> "DeviceFlatBatch":
        """The batch of the outputs of ``torch.ops.mymyr.encode_flat``: a tuple of its 29 tensors (the arrays in the
        order of ``mymyr.encoding.DeviceFlatBatch.array_names``, sizes last), for ``encoder``.
        Supply ``entity_capacity`` for per-entity metadata when the entity arrays are implicit: their omitted tensors
        do not retain the entity extent. This is a Python integer and does not read device counters.
        """
        if len(arrays) != len(_OUTPUTS):
            raise _error(f"expected the {len(_OUTPUTS)} outputs of encode_flat, got {len(arrays)}")
        columns = dict(zip(_OUTPUTS, arrays, strict=True))
        columns["_entity_capacity"] = entity_capacity
        return cls(encoder, columns, exact)

    array_names = _OUTPUTS

    @staticmethod
    def _template(encoder: Any) -> tuple:
        """What every capacity batch of ``encoder`` shares, for :meth:`_of`."""
        layout = layout_of(encoder)
        present = tuple((k, name) for k, name in enumerate(_ARRAYS) if layout.present(name))
        return (encoder, encoder.schema, encoder.schema_fingerprint, encoder.table_fingerprint, encoder.options, layout,
                present)

    @classmethod
    def _of(cls, template: tuple, outputs: Any, entity_capacity: Optional[int] = None) -> "DeviceFlatBatch":
        """The capacity batch of the outputs of ``encode_flat`` (in ``array_names`` order) for a :meth:`_template`,
        made with attribute writes only (so that torch.compile traces it without a break)."""
        batch = object.__new__(cls)
        batch._encoder, batch._schema, batch._fingerprint, batch._table_fingerprint, batch._options, batch._layout = (
            template[:6]
        )
        batch._device_graphs = None
        batch._arrays = {name: outputs[k] for k, name in template[6]}
        batch._static_sizes = outputs[len(_ARRAYS)]
        batch._sizes = outputs[len(_ARRAYS) + 1]
        batch._graphs = outputs[0].shape[0]
        batch._entity_capacity = entity_capacity
        return batch


def _array_property(name: str) -> property:
    def get(self: DeviceFlatBatch) -> Any:
        try:
            return self._arrays[name]
        except KeyError:
            raise AttributeError(f"mymyr: encoding: this batch has no {name}") from None

    return property(get, doc=getattr(FlatBatch, name).__doc__)


for _name in _ARRAYS:
    setattr(DeviceFlatBatch, _name, _array_property(_name))
del _name


class DeviceFlatStream:
    """A fixed-capacity ring of CUDA rows, task ids and optional goal masks and derived words.

    The ring is bound to the current torch stream at construction. Append and capacity flush may be captured;
    keep the stream, inputs, returned batches and this ring alive through the last replay. Append copies all rows
    or sets the sticky device overflow flag without changing the ring. ``clear`` clears that flag. The mutable
    stream remains host-only. With ``constant_literals``, each row holds up to that many constant literals of its goal:
    appended goals may carry them (``GoalMasks.constants``, at most that many per row; under capture a CUDA array).
    """

    def __init__(self, encoder: Any, capacity: int, *, device: Any = None, goals: bool = False,
                 derived: bool = False, derived_goals: bool = False, constant_literals: int = 0):
        torch = _torch()
        from mymyr import rl
        from mymyr.rl.torch._ops import _stream

        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise _error("a device stream needs a positive integer capacity")
        if derived_goals and not goals:
            raise _error("derived goal masks need fluent goal masks")
        if isinstance(constant_literals, bool) or not isinstance(constant_literals, int) or constant_literals < 0:
            raise _error(f"constant_literals must be an int of at least 0, got {constant_literals!r}")
        if constant_literals and not goals:
            raise _error("constant goal literals need goal masks")
        self.encoder, self.capacity = encoder, capacity
        self.device = torch.device("cuda", _device_index(torch, device))
        self._stream = torch.cuda.current_stream(self.device)
        self._stream_ptr = _stream(self.device)
        if torch.cuda.is_current_stream_capturing():
            raise _error("construct the device stream outside capture")
        table = rl.as_table(encoder.table)
        _native().check_stream_schema(encoder)
        self._native = _device_encoder(encoder, self.device.index)
        # Acquiring the immutable context before capture is independent of the scratch allocation.
        self._native.scratch_bytes(self._stream_ptr, capacity)
        self.device_status = torch.zeros(3, dtype=torch.int64, device=self.device)
        self._states = torch.empty((capacity, table.words + table.numeric_words), dtype=torch.int64, device=self.device)
        self._ids = torch.empty(capacity, dtype=torch.int32, device=self.device)
        self._derived = torch.empty((capacity, table.derived_words), dtype=torch.int64, device=self.device) if derived else None
        self._goals = None
        if goals:
            self._goals = tuple(torch.empty((capacity, table.words), dtype=torch.int64, device=self.device) for _ in range(2))
            if derived_goals:
                self._goals += tuple(torch.empty((capacity, table.derived_words), dtype=torch.int64, device=self.device) for _ in range(2))
            if constant_literals:
                from mymyr._core._encoding import _constant_width

                self._goals = _ConstantMasks(self._goals)
                self._goals.constants = torch.full((capacity, constant_literals, _constant_width(encoder)), -1,
                                                   dtype=torch.int32, device=self.device)
        self._rows = _Rows(torch, self._states, self._ids, self._derived, self._goals)

    def _check_stream(self) -> None:
        from mymyr.rl.torch._ops import _stream
        if _stream(self.device) != self._stream_ptr:
            raise _error("use the device stream on the torch stream that constructed it")

    @property
    def device_overflow(self) -> Any:
        """A device int64 scalar view; nonzero means an append did not fit. Reading it on the host waits."""
        return self.device_status[2]

    def __len__(self) -> int:
        """The number of live rows; waits for the device and cannot be read during capture."""
        self._check_stream()
        return int(self.device_status[1].item())

    def append(self, states: Any, task_ids: Any = None, *, derived: Any = None, goals: Any = None) -> None:
        """Copy CUDA rows into the ring. Widths and optional inputs match the columns chosen at construction."""
        torch = _torch()
        self._check_stream()
        keep: list = []
        states = _cuda_tensor(torch, states, "states", keep)
        ids = None if task_ids is None else _cuda_tensor(torch, task_ids, "task_ids", keep)
        d = None if derived is None else _cuda_tensor(torch, derived, "derived", keep)
        g = _cuda_tuple(torch, goals, "goals", keep)
        rows = _Rows(torch, states, ids, d, g)
        if rows.device != self.device:
            raise _error("appended rows and the device ring must be on the same device")
        if not rows.graphs:
            return
        self._native.append(self._stream_ptr, rows.fields, self._rows.fields, self.device_status.data_ptr())
        for tensor in rows.tensors:
            tensor.record_stream(self._stream)
        if keep:
            _retain(torch, keep, self.device)

    def clear(self) -> None:
        """Empty the ring and clear its sticky overflow flag, in stream order."""
        self._check_stream()
        self._native.clear(self._stream_ptr, self.device_status.data_ptr(), self.capacity)

    def flush(self, *, capacities: dict[str, int]) -> DeviceFlatBatch:
        """Encode all live rows once and empty the ring, without a host wait.

        Graph capacity covers the entire ring. Arrays are allocated at capacity; ``device_graphs`` gives the live
        graph count. ``num_graphs`` and exporting a compact dictionary read counts on the host and wait. Direct
        tensor properties retain their capacities: per-graph entries past the live graphs are zero, variable entries
        past the sizes are left unwritten.
        """
        torch = _torch()
        self._check_stream()
        layout = layout_of(self.encoder)
        caps = _capacities(capacities, layout)
        if caps[0] < self.capacity:
            raise _error("flush graph capacity must cover the entire device ring")
        result, scratch = _flush_rows(torch, self._native, self._stream_ptr, self._rows, self.device_status,
                                      self.encoder, self.capacity, caps, self.device, self._derived is not None)
        result._keepalive = (self, scratch)
        return result


def _flush_rows(torch: Any, native: Any, stream: int, rows: "_Rows", status: Any, encoder: Any, capacity: int,
                caps: tuple, device: Any, derived: bool, zero_unused: bool = False) -> tuple[DeviceFlatBatch, Any]:
    """Encodes the live rows of a ring (``rows``, with its int64 [3] ``status``: first row, live count, dropped rows)
    into a batch of the graph capacity ``capacity`` and the capacities ``caps`` (of ``_capacities``), with the private
    relation storage of a Simple-only fill, and empties the ring, on ``stream`` without a host wait. ``derived``: the
    rows carry their derived words. ``zero_unused`` also zeroes the variable arrays past their sizes. Returns the batch
    (its ``device_graphs`` the live count) and the scratch, which must live as long as the batch."""
    layout = layout_of(encoder)
    # a ring's schema has no history steps: the history capacity plays no part
    out = _outputs(torch, layout.specs(torch, capacity, *caps[1:5], numeric=caps[5], static_instances=caps[6],
                                       static_args=caps[7]), device)
    _relation_scratch(torch, out, layout, capacity, caps[2], device)
    graphs = torch.empty(1, dtype=torch.int64, device=device)
    scratch = _scratch(torch, native, stream, capacity, device, derived)
    native.flush(stream, rows.fields, status.data_ptr(), _pointers(out), [capacity, *caps[1:]], graphs.data_ptr(),
                 scratch.data_ptr(), scratch.numel(), zero_unused=zero_unused)
    batch = DeviceFlatBatch(encoder, out, exact=False)
    batch._entity_capacity = caps[1]
    batch._device_graphs = graphs
    return batch, scratch


class DeviceCollator:
    """A prepared list of CUDA batches for repeated or captured rollout collation.

    Construct outside capture on the stream that will encode the batches. The batch tensors keep their addresses;
    values may be updated in stream order before each encode, with graph counts and array shapes fixed.
    Capacity calls allocate output and scratch through
    torch, including its CUDA graph pool. Keep this object and the returned batch alive through the final replay.
    """

    def __init__(self, batches: Any):
        batches = list(batches)
        if not batches or not all(isinstance(batch, DeviceFlatBatch) for batch in batches):
            raise _error("a DeviceCollator needs at least one device batch")
        torch = _torch()
        first = batches[0]
        for batch in batches:
            if batch.device != first.device:
                raise _error("collate received mixed CUDA devices")
            for option in ("layout", "index_dtype", "statics"):
                if batch._options.get(option, "repeat") != first._options.get(option, "repeat"):
                    raise _error(f"collate received mixed {option}")
            if batch._layout.numeric_relations != first._layout.numeric_relations:
                raise _error("collate received mixed numeric_relations")
            if batch._layout.numeric and batch._layout.numeric_dtype != first._layout.numeric_dtype:
                raise _error("collate received mixed numeric_dtype")
            if batch._fingerprint != first._fingerprint:
                raise _error("collate received mixed schemas")
            if batch._table_fingerprint != first._table_fingerprint:
                raise _error("collate received batches of different tables: their task ids name different instances")
        if torch.cuda.is_current_stream_capturing():
            raise _error("prepare a DeviceCollator outside capture for captured collation")
        from mymyr._core._encoding import DeviceCollator as NativeCollator
        from mymyr.rl.torch._ops import _stream

        encoder = first._encoder
        # Keep capacity storage: empty live views have null data pointers.
        arrays = [batch._arrays.copy() for batch in batches]
        inputs = [_pointers({**a, "sizes": batch._sizes, "static_sizes": batch._static_sizes})
                  for a, batch in zip(arrays, batches, strict=True)]
        counts = [batch.num_graphs for batch in batches]
        encoders = [batch._encoder for batch in batches]
        shapes = [[count, batch.static_task_ids.numel() if batch._layout.shared else 0,
                   batch.static_args.numel() if batch._layout.shared else 0]
                  for count, batch in zip(counts, batches, strict=True)]
        native = NativeCollator(encoders, first.device.index, _stream(first.device), inputs, shapes)
        self._prepared = (torch, first, encoder, native, arrays, batches, counts)
        self._stream = _stream(first.device)

    def encode(self, *, capacities: Any = None) -> DeviceFlatBatch:
        """Concatenate the prepared graphs; a capacity call never waits and is capturable."""
        from mymyr.rl.torch._ops import _stream
        torch, first, encoder, native, arrays, batches, counts = self._prepared
        if _stream(first.device) != self._stream:
            raise _error("use the collator on the torch stream that constructed it")
        scratch = torch.empty(native.scratch_bytes, dtype=torch.uint8, device=first.device)
        layout = layout_of(encoder)
        graphs = sum(counts)
        out: dict[str, Any] = {}
        caps = None if capacities is None else _capacities(capacities, layout)
        if caps is None:
            def allocate(entities: int, tuples: int, args: int, targets: int, numeric: int, flags: int,
                         static_instances: int, static_tuples: int, static_args: int) -> list[int]:
                if flags & _OVERFLOW_INDEX:
                    raise OverflowError("mymyr: encoding: collated totals exceed the index type")
                if flags:
                    raise _error("cannot collate an overflowed batch")
                specs = layout.specs(torch, graphs, entities, args, targets, tuples=tuples, numeric=numeric,
                                     static_instances=static_instances, static_args=static_args)
                out["_entity_capacity"] = entities
                for name, (shape, dtype) in specs.items():
                    if name != "sizes":
                        out[name] = torch.empty(shape, dtype=dtype, device=first.device)
                return _pointers(out)
            native.encode_exact(scratch.data_ptr(), scratch.numel(), allocate)
            out["sizes"] = None
        else:
            if caps[0] < graphs:
                raise _error("collated graphs exceed the output capacity")
            # collated batches have no history steps: the history capacity plays no part
            out = _outputs(torch, layout.specs(torch, graphs, *caps[1:5], numeric=caps[5], static_instances=caps[6],
                                               static_args=caps[7]), first.device)
            out["_entity_capacity"] = caps[1]
            native.encode(_pointers(out), [graphs, *caps[1:]], scratch.data_ptr(), scratch.numel())
        result = DeviceFlatBatch(encoder, out, exact=caps is None)
        result._keepalive = (native, scratch, batches, arrays)
        return result


def collate_device(batches: list[DeviceFlatBatch], capacities: Any = None) -> DeviceFlatBatch:
    """Concatenate device batches with one count/scan/fill, rebuilding offsets on the device."""
    return DeviceCollator(batches).encode(capacities=capacities)
