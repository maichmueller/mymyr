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

The device evaluates axioms into scratch when a graph needs derived atoms and ``derived=`` is absent. Supplied
derived words take precedence. Action relations (``actions=``) are not supported on the device.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any, Optional

from mymyr._core._encoding import FlatBatch

__all__ = ["DeviceFlatBatch"]

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
)
_OUTPUTS = (*_ARRAYS, "sizes")
_GOAL_LANES = {"goal", "goal_derivation", "goal_satisfaction"}
_SIZE_FIELDS = ("entities", "tuples", "args", "targets", "overflow")
_OVERFLOW_CAPACITY, _OVERFLOW_INDEX, _OVERFLOW_INPUT = 1, 2, 4


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

    __slots__ = ("counts", "entity_arrays", "goal_complete", "index", "relations", "targets")

    def __init__(self, encoder: Any):
        options = encoder.options
        self.index = options["index_dtype"]
        self.relations = encoder.num_relations
        self.entity_arrays = not encoder.implicit_entities
        self.counts = bool(options["emit_counts"])
        self.targets = bool(options["target_sources"])
        self.goal_complete = any(lane in _GOAL_LANES for lane in encoder.relation_lanes)

    def present(self, name: str) -> bool:
        if name in ("entity_role", "entity_ref"):
            return self.entity_arrays
        if name == "relation_counts":
            return self.counts
        if name == "goal_complete":
            return self.goal_complete
        if name.startswith("target_"):
            return self.targets
        return True

    def specs(self, torch: Any, graphs: int, entities: int, args: int, targets: int) -> dict[str, tuple[tuple[int, ...], Any]]:
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
        }
        specs = {name: (shape if self.present(name) else (0,), dtype) for name, (shape, dtype) in shapes.items()}
        specs["sizes"] = ((len(_SIZE_FIELDS),), torch.int64)
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


class _Rows:
    """The checked rows of a call: the device pointers and strides the native side takes."""

    __slots__ = ("device", "fields", "graphs", "tensors")

    def __init__(self, torch: Any, states: Any, task_ids: Any, derived: Any, goals: Any):
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
                shared = pos.shape[0] == 1 and graphs != 1
                at = 4 * pair
                goal[at:at + 4] = [pos.data_ptr(), neg.data_ptr(), 0 if shared else _stride(pos), pos.shape[1]]
        self.tensors = tensors
        self.fields = [graphs, w.data_ptr() if w.numel() else 0, _stride(w), w.shape[1], ids_ptr, derived_ptr,
                       derived_stride, *goal[:4], *goal[4:]]
        if len(self.fields) != 15:
            raise AssertionError("the row fields have a fixed length")

    def _same_device(self, t: Any, name: str) -> None:
        if t.device != self.device:
            raise _error(f"'{name}' is on {t.device}, the states on {self.device}")


def _capacities(value: Any, layout: _Layout) -> tuple[int, int, int, int]:
    """The (graphs, entities, args, targets) capacities of a ``capacities`` dict."""
    if not isinstance(value, dict):
        raise _error(f"capacities must be a dict, not {type(value).__name__}")
    keys = {"entities", "graphs", "args", "targets"}
    unknown = set(value) - keys
    if unknown:
        raise _error(f"capacities has the unknown key '{sorted(unknown, key=str)[0]}'; the keys are 'entities', "
                     f"'graphs', 'args' and 'targets'")
    if not {"entities", "graphs", "args"} <= set(value):
        raise _error("capacities needs the keys 'entities', 'graphs' and 'args'")
    if "targets" not in value and layout.targets:
        raise _error("the schema has targets: capacities needs the key 'targets' as well")
    out = []
    for key in ("graphs", "entities", "args", "targets"):
        n = value.get(key, 0)
        if isinstance(n, bool) or not isinstance(n, int):
            if hasattr(n, "__index__") and not isinstance(n, bool):
                n = n.__index__()
            else:
                raise _error(f"capacities['{key}'] must be an int, not {type(n).__name__}")
        if n < 0:
            raise _error(f"capacities['{key}'] must be at least 0, got {n}")
        out.append(n)
    return out[0], out[1], out[2], out[3]


# ------------------------------------------------------------------------------------------------ pending inputs

_pending: list = []
_pending_lock = threading.Lock()


def _retain(torch: Any, keep: list, device: Any) -> None:
    """Keeps the wrappers of foreign arrays alive until the work enqueued so far has completed (their owners free the
    memory when the wrappers go)."""
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
    return [tensors[name].data_ptr() if tensors.get(name) is not None and tensors[name].numel() else 0
            for name in _OUTPUTS]


def _outputs(torch: Any, specs: dict, device: Any) -> dict[str, Any]:
    """Independent uninitialized outputs; the fill kernel writes the used entries and zeros the tails."""
    return {name: torch.empty(shape, dtype=dtype, device=device) for name, (shape, dtype) in specs.items()}


def _scratch(torch: Any, native: Any, stream: int, graphs: int, device: Any) -> Any:
    """Per-call scratch, owned by the graph's private allocator pool during capture."""
    return torch.empty(native.scratch_bytes(stream, graphs), dtype=torch.uint8, device=device)


def encode_arrays(
    encoder: Any,
    states: Any,
    task_ids: Any,
    derived: Any,
    goals: Any,
    capacities: Optional[tuple[int, int, int, int]],
    keep: Optional[list] = None,
) -> dict[str, Any]:
    """The arrays of the encoding of checked tensors, by name (absent arrays empty), with ``sizes`` [5] int64 (entities,
    tuples, args, targets, overflow flags; None without capacities)."""
    torch = _torch()
    layout = layout_of(encoder)
    rows = _Rows(torch, states, task_ids, derived, goals)
    device = rows.device
    native = _device_encoder(encoder, device.index)
    from mymyr.rl.torch._ops import _stream

    stream = _stream(states)
    out: dict[str, Any] = {}
    if capacities is not None:
        graphs, entities, args, targets = capacities
        if rows.graphs > graphs:
            raise _error("the batch has more graphs than the output capacity")
        specs = layout.specs(torch, rows.graphs, entities, args, targets)
        out = _outputs(torch, specs, device)
        scratch = _scratch(torch, native, stream, rows.graphs, device)
        native.encode(stream, rows.fields, _pointers(out),
                      [out["task_ids"].shape[0], entities, args, targets], scratch.data_ptr(), scratch.numel())
    else:

        def allocate(entities: int, tuples: int, args: int, targets: int, flags: int) -> list[int]:
            if flags & _OVERFLOW_INPUT:
                raise _error("a row is invalid: a task id outside the table, a state row or goal mask narrower than "
                             "its instance, or a bit that its instance has no atom for")
            if flags & _OVERFLOW_INDEX:
                raise OverflowError("mymyr: encoding: the batch needs more indices than its index type holds "
                                    f"(entities {entities}, tuples {tuples}, args {args}, targets {targets}): "
                                    "use index_dtype='int64'")
            specs = layout.specs(torch, rows.graphs, entities, args, targets)
            for name, (shape, dtype) in specs.items():
                if name != "sizes":
                    out[name] = torch.empty(shape, dtype=dtype, device=device)
            return [out["task_ids"].shape[0], *_pointers(out)]

        native.encode_exact(stream, rows.fields, allocate)
        out["sizes"] = None
    if keep:
        _retain(torch, keep, device)
    return out


def encode(
    encoder: Any,
    states: Any,
    task_ids: Any,
    derived: Any,
    goals: Any,
    actions: Any,
    framework: Any,
    threads: Any,
    pool: Any,
    capacities: Any,
) -> "DeviceFlatBatch":
    """``FlatRelationEncoder.encode`` for arguments that are CUDA arrays."""
    torch = _torch()
    if actions is not None:
        raise _error("actions are not encoded on the device: encode the rows on the host with actions, or without actions=")
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
        g = tuple(_cuda_tensor(torch, mask, f"goals[{k}]", keep) for k, mask in enumerate(goals))
    arrays = encode_arrays(encoder, s, ids, d, g, caps, keep)
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
        else:
            offsets = batch.relation_arg_offsets[graph * r:(graph + 1) * r + 1].tolist()
            args = batch.relation_args[offsets[0]:offsets[-1]]
            lengths = np.diff(offsets)
        self.relation_arg_offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
        self.relation_args = args.cpu().numpy() - begin


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
        self._options = encoder.options
        layout = layout_of(encoder)
        self._arrays = {name: arrays[name] for name in _ARRAYS if layout.present(name)}
        self._sizes = None if exact else arrays["sizes"]
        self._graphs = self._arrays["task_ids"].shape[0]

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
        """After an encoding with capacities: a device tensor int64 [5] with the true entities, tuples, args and targets
        of the batch and the overflow flags (1: a capacity, 2: the index type, 4: an invalid row), with no host wait;
        None otherwise."""
        return self._sizes

    @property
    def sizes(self) -> Optional[dict[str, Any]]:
        """After an encoding with capacities: the true entities, tuples, args and targets of the batch, and whether it overflowed
        (``overflow``: the batch is all zero) because of the capacities (``capacity_overflow``), the index type
        (``index_overflow``) or an invalid row (``invalid_input``: a task id outside the table, a row or goal mask
        narrower than its instance, an atom bit its instance has no slot for); None otherwise. It reads
        ``device_sizes`` on the host, which waits for the device."""
        if self._sizes is None:
            return None
        values = dict(zip(_SIZE_FIELDS, self._sizes.tolist(), strict=True))
        flags = values["overflow"]
        values["overflow"] = flags != 0
        values["capacity_overflow"] = bool(flags & _OVERFLOW_CAPACITY)
        values["index_overflow"] = bool(flags & _OVERFLOW_INDEX)
        values["invalid_input"] = bool(flags & _OVERFLOW_INPUT)
        return values

    def as_dict(self, framework: Optional[str] = None) -> dict[str, Any]:
        """The scalars (version, num_graphs, num_relations, layout, index_dtype, schema_fingerprint, statics) and the arrays
        of the batch. framework: ``"torch"`` or ``"dlpack"`` (default; the tensors, which speak DLPack) or ``"numpy"``
        (copies to the host and waits for the device)."""
        scalars = {
            "version": 1,
            "num_graphs": self.num_graphs,
            "num_relations": self.num_relations,
            "layout": 1 if self._options["layout"] == "relation" else 0,
            "index_dtype": 1 if self._options["index_dtype"] == "int64" else 0,
            "schema_fingerprint": _signed(self._fingerprint),
            "statics": 0,
        }
        arrays = self._arrays
        if self._device_graphs is not None:
            g = scalars["num_graphs"]
            sizes = self.sizes
            arrays = dict(arrays)
            extents = {"task_ids": g, "num_objects": g, "entity_offsets": g + 1, "goal_complete": g,
                       "target_offsets": g + 1, "relation_offsets": g * self.num_relations + 1,
                       "relation_arg_offsets": g * self.num_relations + 1, "entity_role": sizes["entities"],
                       "entity_ref": sizes["entities"], "relation_args": sizes["args"],
                       "target_entity": sizes["targets"], "target_group": sizes["targets"]}
            arrays = {key: value[:g] if key == "relation_counts" else value[:extents[key]] for key, value in arrays.items()}
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
    def from_arrays(cls, encoder: Any, arrays: Any, exact: bool = False) -> "DeviceFlatBatch":
        """The batch of the outputs of ``torch.ops.mymyr.encode_flat``: a tuple of its 14 tensors (the arrays in the
        order of ``mymyr.encoding.DeviceFlatBatch.array_names``, sizes last), for ``encoder``."""
        if len(arrays) != len(_OUTPUTS):
            raise _error(f"expected the {len(_OUTPUTS)} outputs of encode_flat, got {len(arrays)}")
        return cls(encoder, dict(zip(_OUTPUTS, arrays, strict=True)), exact)

    array_names = _OUTPUTS


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
    stream remains host-only.
    """

    def __init__(self, encoder: Any, capacity: int, *, device: Any = None, goals: bool = False,
                 derived: bool = False, derived_goals: bool = False):
        torch = _torch()
        from mymyr import rl
        from mymyr.rl.torch._ops import _stream

        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise _error("a device stream needs a positive integer capacity")
        if derived_goals and not goals:
            raise _error("derived goal masks need fluent goal masks")
        self.encoder, self.capacity = encoder, capacity
        self.device = torch.device("cuda", _device_index(torch, device))
        self._stream = torch.cuda.current_stream(self.device)
        self._stream_ptr = _stream(self.device)
        if torch.cuda.is_current_stream_capturing():
            raise _error("construct the device stream outside capture")
        table = rl.as_table(encoder.table)
        self._native = _device_encoder(encoder, self.device.index)
        # Acquiring the immutable context before capture is independent of the scratch allocation.
        self._native.scratch_bytes(self._stream_ptr, capacity)
        self.device_status = torch.zeros(3, dtype=torch.int64, device=self.device)
        self._states = torch.empty((capacity, table.words), dtype=torch.int64, device=self.device)
        self._ids = torch.empty(capacity, dtype=torch.int32, device=self.device)
        self._derived = torch.empty((capacity, table.derived_words), dtype=torch.int64, device=self.device) if derived else None
        self._goals = None
        if goals:
            self._goals = tuple(torch.empty((capacity, table.words), dtype=torch.int64, device=self.device) for _ in range(2))
            if derived_goals:
                self._goals += tuple(torch.empty((capacity, table.derived_words), dtype=torch.int64, device=self.device) for _ in range(2))
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
        return int(self.device_status[1].item())

    def append(self, states: Any, task_ids: Any = None, *, derived: Any = None, goals: Any = None) -> None:
        """Copy CUDA rows into the ring. Widths and optional inputs match the columns chosen at construction."""
        torch = _torch()
        self._check_stream()
        keep: list = []
        states = _cuda_tensor(torch, states, "states", keep)
        ids = None if task_ids is None else _cuda_tensor(torch, task_ids, "task_ids", keep)
        d = None if derived is None else _cuda_tensor(torch, derived, "derived", keep)
        g = None if goals is None else tuple(_cuda_tensor(torch, x, f"goals[{i}]", keep) for i, x in enumerate(goals))
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
        tensor properties retain their capacities, including zero tails.
        """
        torch = _torch()
        self._check_stream()
        layout = layout_of(self.encoder)
        caps = _capacities(capacities, layout)
        if caps[0] < self.capacity:
            raise _error("flush graph capacity must cover the entire device ring")
        out = _outputs(torch, layout.specs(torch, self.capacity, *caps[1:]), self.device)
        graphs = torch.empty(1, dtype=torch.int64, device=self.device)
        scratch = _scratch(torch, self._native, self._stream_ptr, self.capacity, self.device)
        self._native.flush(self._stream_ptr, self._rows.fields, self.device_status.data_ptr(), _pointers(out),
                           [self.capacity, *caps[1:]], graphs.data_ptr(), scratch.data_ptr(), scratch.numel())
        result = DeviceFlatBatch(self.encoder, out, exact=False)
        result._device_graphs = graphs
        result._keepalive = (self, scratch)
        return result


class DeviceCollator:
    """A prepared list of CUDA batches for repeated or captured rollout collation.

    Construct outside capture on the stream that will encode the batches. The batch tensors keep their addresses;
    values may be updated in stream order before each encode. Capacity calls allocate output and scratch through
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
            if batch._fingerprint != first._fingerprint:
                raise _error("collate received mixed schemas")
        if torch.cuda.is_current_stream_capturing():
            raise _error("prepare a DeviceCollator outside capture for captured collation")
        from mymyr._core._encoding import DeviceCollator as NativeCollator
        from mymyr.rl.torch._ops import _stream

        encoder = first._encoder
        arrays = [batch.as_dict("torch") for batch in batches]
        inputs = [_pointers({**a, "sizes": batch._sizes}) for a, batch in zip(arrays, batches, strict=True)]
        counts = [batch.num_graphs for batch in batches]
        native = NativeCollator(encoder, first.device.index, _stream(first.device), inputs, counts)
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
            def allocate(entities: int, tuples: int, args: int, targets: int, flags: int) -> list[int]:
                if flags & _OVERFLOW_INDEX:
                    raise OverflowError("mymyr: encoding: collated totals exceed the index type")
                if flags:
                    raise _error("cannot collate an overflowed batch")
                for name, (shape, dtype) in layout.specs(torch, graphs, entities, args, targets).items():
                    if name != "sizes":
                        out[name] = torch.empty(shape, dtype=dtype, device=first.device)
                return _pointers(out)
            native.encode_exact(scratch.data_ptr(), scratch.numel(), allocate)
            out["sizes"] = None
        else:
            if caps[0] < graphs:
                raise _error("collated graphs exceed the output capacity")
            out = _outputs(torch, layout.specs(torch, graphs, *caps[1:]), first.device)
            native.encode(_pointers(out), [graphs, *caps[1:]], scratch.data_ptr(), scratch.numel())
        result = DeviceFlatBatch(encoder, out, exact=caps is None)
        result._keepalive = (native, scratch, batches, arrays)
        return result


def collate_device(batches: list[DeviceFlatBatch], capacities: Any = None) -> DeviceFlatBatch:
    """Concatenate device batches with one count/scan/fill, rebuilding offsets on the device."""
    return DeviceCollator(batches).encode(capacities=capacities)
