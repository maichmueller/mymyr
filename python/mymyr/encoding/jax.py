"""Encoding rows inside JAX: :func:`encode` is ``FlatRelationEncoder.encode`` as the XLA FFI target
``mymyr_encode_flat``, so a jitted function (and ``lax.scan``, ``jax.vmap``) can encode states. The target runs the
CPU encoder on the CPU and the device kernels on CUDA (on XLA's stream, with XLA's memory); both give the arrays of the
encoder's capacity form bit for bit, and :class:`JaxFlatBatch` holds them with the names of ``FlatBatch``.

Rows are the u64 words of each state as JAX sees them without 64-bit types: uint32 ``[N, 2 W]`` (the low half first;
``task.encode(states)`` gives the uint64 words, which a NumPy view or ``mymyr.rl.jax.as_words`` turns into halves).
uint64 and int64 words are accepted too, and int32 halves.

Capacities: shapes are static under ``jax.jit``, so a traced call needs ``capacities=dict(graphs=, entities=, args=,
targets=)`` (``targets`` only with target sources): the variable-size arrays have the capacities' sizes, the true
sizes and the overflow flags are in ``batch.device_sizes``, and a batch that does not fit is all zero with its flag
set. An eager call without capacities waits once for the batch's sizes: it encodes at the capacities held for the encoder
and the number of rows (the first call counts the batch to find them, rounded up to powers of two), encodes again at
larger ones if the batch did not fit, and cuts the arrays to the batch. The held capacities only grow, so batches of
similar sizes share one compiled call.

Handles: the target finds the encoder through the FFI handle registry. :func:`register` enters an encoder (``encode``
does so on first use) and the registry holds it until :func:`unregister`; a compiled function also holds what it calls,
so it keeps working after its encoder is unregistered and every Python reference to it is gone.

What the target encodes: what the device encodes, on both platforms (a function gives the same arrays wherever it
runs): no action relations (they need an expansion, which JAX cannot pass), no schema the device refuses, and the
rows' derived words are required where the device cannot evaluate the axioms a graph reads (an axiom body or object
set beyond the device's matcher limits). Such calls raise ``ValueError`` (``mymyr: encoding: ...``) when the function is
traced.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from mymyr._core import _encoding_jax
from mymyr.encoding import _device
from mymyr.rl.jax import _ffi

__all__ = ["JaxFlatBatch", "encode", "register", "registered", "unregister"]

_ARRAYS = _device._ARRAYS
_SIZE_FIELDS = _device._SIZE_FIELDS
_OVERFLOW_WORD = _SIZE_FIELDS.index("overflow")


def _error(message: str) -> ValueError:
    return ValueError(f"mymyr: encoding: {message}")


# ------------------------------------------------------------------------------------------------ the registry


class _Registration:
    """What the Python side keeps of a registered encoder: its schema, which arrays it stores, what the target needs,
    the scratch sizes asked so far and the capacities of its eager calls."""

    __slots__ = ("info", "layout", "options", "schema", "fingerprint", "table_fingerprint", "scratch", "held")

    def __init__(self, encoder: Any, handle: int):
        self.schema = encoder.schema
        self.fingerprint = encoder.schema_fingerprint
        self.table_fingerprint = encoder.table_fingerprint
        self.options = encoder.options
        self.layout = _device.layout_of(encoder)
        self.info = _encoding_jax.info(handle)
        self.scratch: dict[tuple[int, int, bool], int] = {}
        # the capacities of eager calls without capacities, per number of rows: they only grow
        self.held: dict[int, tuple[int, int, int]] = {}


_lock = threading.Lock()
_handles: "weakref.WeakKeyDictionary[Any, int]" = weakref.WeakKeyDictionary()
_registrations: dict[int, _Registration] = {}


def register(encoder: Any) -> int:
    """The FFI handle of ``encoder``, a ``FlatRelationEncoder`` (the same handle for the same encoder while it is
    registered). The registry holds the encoder (its table, options and device contexts) until :func:`unregister`."""
    with _lock:
        handle = _handles.get(encoder)
        if handle is not None and handle in _registrations:
            return handle
        handle = _encoding_jax.register(encoder)
        try:
            _registrations[handle] = _Registration(encoder, handle)
        except BaseException:
            _encoding_jax.unregister(handle)
            raise
        _handles[encoder] = handle
        return handle


def registered(encoder: Any) -> Optional[int]:
    """The handle of ``encoder`` if it is registered, else None."""
    with _lock:
        handle = _handles.get(encoder)
        return handle if handle in _registrations else None


def unregister(encoder: Any) -> None:
    """Drops the registry's reference to an encoder (or to the encoder of a handle). Functions traced afterwards register
    it again; executables already compiled keep the encoder until they are destroyed (``jax.clear_caches()`` and
    dropping the jitted functions frees them)."""
    with _lock:
        if isinstance(encoder, int) and not isinstance(encoder, bool):
            handle: Optional[int] = encoder
        else:
            handle = _handles.pop(encoder, None)
        if handle is None or _registrations.pop(handle, None) is None:
            return
        _encoding_jax.unregister(handle)


def _registration(encoder: Any) -> tuple[int, _Registration]:
    """The handle and registration of an encoder (registered now if it is not) or of a handle."""
    if isinstance(encoder, int) and not isinstance(encoder, bool):
        with _lock:
            reg = _registrations.get(encoder)
        if reg is None:
            raise _error(f"no encoder is registered under handle {encoder} (see mymyr.encoding.jax.register)")
        return encoder, reg
    handle = register(encoder)
    with _lock:
        return handle, _registrations[handle]


# ------------------------------------------------------------------------------------------------ the batch


def _signed(fingerprint: int) -> int:
    return fingerprint - (1 << 64) if fingerprint >= 1 << 63 else fingerprint


class _SchemaRef:
    """The schema of a batch, held by the batch's pytree structure (so a batch outlives the unregistration of its
    encoder). Two references are equal when they name the same registration, as pytree structures compare."""

    __slots__ = ("handle", "schema")

    def __init__(self, schema: Any, handle: int):
        self.schema = schema
        self.handle = handle

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _SchemaRef) and self.handle == other.handle

    def __hash__(self) -> int:
        return hash(self.handle)


@jax.tree_util.register_pytree_node_class
class JaxFlatBatch:
    """The graphs of a batch of states as JAX arrays: what a ``FlatBatch`` is, with the same array names, shapes and
    dtypes (an array that the schema does not store raises ``AttributeError``). A pytree, so it passes through
    ``jax.jit``, ``lax.scan`` and ``jax.vmap`` (vmapped, every array gains the leading axis). Encoded with capacities,
    the variable-size arrays have the capacities' sizes and ``device_sizes`` holds the true sizes and flags; ``sizes``
    reads them on the host."""

    __slots__ = ("_arrays", "_sizes", "_static")

    # (names of the stored arrays, num_relations, layout, index_dtype, schema fingerprint, implicit entities, schema,
    # table fingerprint)
    def __init__(self, arrays: dict[str, Any], sizes: Optional[Any], static: tuple):
        self._arrays = arrays
        self._sizes = sizes
        self._static = static

    def tree_flatten(self) -> tuple[tuple, tuple]:
        names = self._static[0]
        return (*(self._arrays[n] for n in names), self._sizes), self._static

    @classmethod
    def tree_unflatten(cls, static: tuple, children: tuple) -> "JaxFlatBatch":
        names = static[0]
        return cls(dict(zip(names, children[:-1], strict=True)), children[-1], static)

    @property
    def num_graphs(self) -> int:
        """The number of graphs (states) in the batch."""
        return int(self._arrays["task_ids"].shape[-1])

    @property
    def num_relations(self) -> int:
        """The number of relations of the schema the batch was encoded with."""
        return self._static[1]

    @property
    def schema(self) -> Any:
        """The EncodingSchema the batch was encoded for."""
        return self._static[6].schema

    @property
    def implicit_entities(self) -> bool:
        """True if the batch has no entity_role and entity_ref (see ``FlatBatch.implicit_entities``)."""
        return self._static[5]

    @property
    def device_sizes(self) -> Optional[Any]:
        """After an encoding with capacities: uint32 [S, 2], the S 64-bit words named by ``DeviceFlatBatch.sizes`` (entities,
        tuples, args, targets, history and the overflow flags: 1 a capacity, 2 the index type, 4 an invalid row) as
        uint32 halves, low half first; None for an exact batch."""
        return self._sizes

    @property
    def overflow(self) -> Optional[Any]:
        """After an encoding with capacities: the overflow flags as a uint32 scalar (0: the batch fits), usable under
        ``jax.jit``; None for an exact batch."""
        return None if self._sizes is None else self._sizes[..., _OVERFLOW_WORD, 0]

    @property
    def sizes(self) -> Optional[dict[str, Any]]:
        """After an encoding with capacities: the true entities, tuples, args and targets of the batch, and whether it
        overflowed (``overflow``: the batch is all zero) because of the capacities (``capacity_overflow``), the index
        type (``index_overflow``) or an invalid row (``invalid_input``); None for an exact batch. It reads
        ``device_sizes`` on the host, which waits for the computation (not under ``jax.jit``). Of a batch stacked by
        ``lax.scan`` or ``jax.vmap`` every value is a NumPy array over the stacked axes."""
        if self._sizes is None:
            return None
        return _host_sizes(np.asarray(self._sizes))

    def as_dict(self, framework: Optional[str] = None) -> dict[str, Any]:
        """The scalars (version, num_graphs, num_relations, layout, index_dtype, schema_fingerprint, table_fingerprint,
        statics) and the arrays
        of the batch. framework: ``"jax"`` (default) or ``"dlpack"`` (the JAX arrays, which speak DLPack) or ``"numpy"``
        (host copies, which wait for the computation)."""
        _, relations, layout, index, fingerprint, _, _, table_fingerprint = self._static
        scalars = {
            "version": 1,
            "num_graphs": self.num_graphs,
            "num_relations": relations,
            "layout": layout,
            "index_dtype": index,
            "schema_fingerprint": fingerprint,
            "table_fingerprint": table_fingerprint,
            "statics": 0,
        }
        if framework in (None, "jax", "dlpack"):
            return {**scalars, **self._arrays}
        if framework == "numpy":
            return {**scalars, **{name: np.asarray(a) for name, a in self._arrays.items()}}
        raise _error(f"a JAX batch exports to 'jax', 'dlpack' or 'numpy', not {framework!r}")

    def __getattr__(self, name: str) -> Any:
        if name in _ARRAYS:
            raise AttributeError(f"mymyr: encoding: this batch has no {name}")
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"JaxFlatBatch(num_graphs={self.num_graphs}, num_relations={self.num_relations})"

    array_names = _ARRAYS


def _array_property(name: str) -> property:
    def get(self: JaxFlatBatch) -> Any:
        try:
            return self._arrays[name]
        except KeyError:
            raise AttributeError(f"mymyr: encoding: this batch has no {name}") from None

    return property(get, doc=getattr(_device.FlatBatch, name).__doc__)


for _name in _ARRAYS:
    setattr(JaxFlatBatch, _name, _array_property(_name))
del _name


def _host_sizes(words: np.ndarray) -> dict[str, Any]:
    """The sizes of ``device_sizes`` words (uint32 [..., S, 2]): Python values for one batch, NumPy arrays over the
    leading axes of a stacked batch."""
    halves = np.ascontiguousarray(words, dtype=np.uint32).astype(np.uint64)
    packed = halves[..., 0] | (halves[..., 1] << np.uint64(32))  # [..., S]
    # a batch of this target has no history steps (the device refuses such schemas), as a host batch without history
    values = {name: packed[..., k] for k, name in enumerate(_SIZE_FIELDS) if name != "history"}
    flags = values["overflow"]
    values["overflow"] = flags != 0
    values["capacity_overflow"] = (flags & _device._OVERFLOW_CAPACITY) != 0
    values["index_overflow"] = (flags & _device._OVERFLOW_INDEX) != 0
    values["invalid_input"] = (flags & _device._OVERFLOW_INPUT) != 0
    if packed.ndim == 1:
        return {name: (bool(v) if v.dtype == np.bool_ else int(v)) for name, v in values.items()}
    return values


# ------------------------------------------------------------------------------------------------ the arguments


def _traced(*values: Any) -> bool:
    return any(isinstance(v, jax.core.Tracer) for v in values)


def _halves(x: Any, name: str, one_row: bool) -> Any:
    """``x`` as uint32 [rows, 2 words]: the u64 words of each row as halves, low half first. A single row [2 words]
    becomes [1, 2 words] when ``one_row``."""
    if isinstance(x, jax.Array):
        if x.dtype in (jnp.uint64, jnp.int64):
            x = jax.lax.bitcast_convert_type(x, jnp.uint32).reshape(*x.shape[:-1], 2 * x.shape[-1])
        elif x.dtype == jnp.int32:
            x = jax.lax.bitcast_convert_type(x, jnp.uint32)
        elif x.dtype != jnp.uint32:
            raise _error(f"'{name}' must hold uint32 or int32 halves or uint64 or int64 words, not {x.dtype}")
    else:
        a = np.asarray(x)
        if a.dtype in (np.uint64, np.int64):
            a = np.ascontiguousarray(a).view(np.uint32)
        elif a.dtype in (np.uint32, np.int32):
            a = a.view(np.uint32)
        else:
            raise _error(f"'{name}' must hold uint32 or int32 halves or uint64 or int64 words (task.encode(states)), "
                         f"not {a.dtype}")
        x = jnp.asarray(a)
    if x.ndim == 1 and one_row:
        x = x.reshape(1, x.shape[0])
    if x.ndim != 2 or x.shape[1] % 2:
        raise _error(f"'{name}' must be [rows, 2 words] uint32 halves (or [rows, words] 64-bit words), got shape "
                     f"{tuple(x.shape)}")
    return x


def _task_ids(task_ids: Any, graphs: int, instances: int) -> Any:
    if task_ids is None:
        if instances > 1 and graphs:
            raise _error("task_ids are required for a table of several instances")
        return jnp.zeros((0,), jnp.int32)
    if isinstance(task_ids, jax.Array):
        if not jnp.issubdtype(task_ids.dtype, jnp.integer):
            raise _error(f"task_ids must be integers, not {task_ids.dtype}")
        ids = task_ids.astype(jnp.int32)
    else:
        a = np.asarray(task_ids)
        if a.size and not np.issubdtype(a.dtype, np.integer):
            raise _error(f"task_ids must be integers, not {a.dtype}")
        ids = jnp.asarray(a.astype(np.int32))
    if ids.ndim == 0:
        ids = ids.reshape(1)
    if ids.shape != (graphs,):
        raise _error(f"task_ids must be [{graphs}] (one per row), got shape {tuple(ids.shape)}")
    return ids


def _goals(goals: Any, graphs: int) -> tuple[Any, Any, Any, Any]:
    empty = jnp.zeros((0,), jnp.uint32)
    if goals is None:
        return empty, empty, empty, empty
    if not isinstance(goals, (tuple, list)) or len(goals) not in (2, 4):
        raise _error("goals are (goal_pos, goal_neg) or (goal_pos, goal_neg, goal_derived_pos, goal_derived_neg)")
    names = ("goal_pos", "goal_neg", "goal_derived_pos", "goal_derived_neg")
    masks = [_halves(g, names[k], True) for k, g in enumerate(goals)]
    for k in range(0, len(masks), 2):
        p, n = masks[k], masks[k + 1]
        if p.shape != n.shape:
            raise _error(f"'{names[k]}' and '{names[k + 1]}' must have the same shape")
        if p.shape[0] not in (graphs, 1):
            raise _error(f"'{names[k]}' must have one row per state ({graphs}) or one row for all")
    masks += [empty] * (4 - len(masks))
    return masks[0], masks[1], masks[2], masks[3]


def _cuda_device() -> Optional[Any]:
    """JAX's first CUDA device, or None when JAX has no CUDA backend."""
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        return None
    return devices[0] if devices else None


def _ordinal(device: Any) -> int:
    """The CUDA ordinal of a JAX device: its local hardware id (0 is a valid one), else its id."""
    local = getattr(device, "local_hardware_id", None)
    return int(device.id if local is None else local)


def _scratch(reg: _Registration, handle: int, graphs: int, derived: bool) -> Optional[int]:
    """The bytes of the call's scratch result: the device encoder's scratch for `graphs` rows (with `derived`: the rows
    come with their derived words, otherwise the device evaluates them) plus 255 bytes of alignment slack, or None
    without a CUDA backend (the CPU target needs none)."""
    if not reg.info["cuda"]:
        return None
    device = _cuda_device()
    if device is None:
        return None
    ordinal = _ordinal(device)
    key = (graphs, ordinal, derived)
    n = reg.scratch.get(key)
    if n is None:
        n = reg.scratch[key] = _encoding_jax.scratch_bytes(handle, graphs, ordinal, derived) + 255
    return n


def _call(handle: int, reg: _Registration, operands: tuple, graphs: int, entities: int, args: int, targets: int,
          threads: int) -> tuple[dict[str, Any], Any]:
    """One mymyr_encode_flat call: the stored arrays by name and the sizes."""
    specs = reg.layout.specs(jnp, graphs, entities, args, targets)
    del specs["sizes"]
    results = [jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in specs.values()]
    results.append(jax.ShapeDtypeStruct((len(_SIZE_FIELDS), 2), jnp.uint32))

    def call(rows: tuple, scratch: Optional[int]) -> tuple:
        shapes = results if scratch is None else [*results, jax.ShapeDtypeStruct((scratch,), jnp.uint8)]
        out = jax.ffi.ffi_call("mymyr_encode_flat", shapes, vmap_method="sequential")(
            *rows, args=np.int64(args), entities=np.int64(entities), handle=np.int64(handle),
            targets=np.int64(targets), threads=np.int64(threads),
        )  # fmt: skip
        return tuple(out[: len(results)])

    scratch = _scratch(reg, handle, graphs, operands[2].shape[0] != 0)
    if scratch is None:
        out = call(operands, None)
    else:  # only a CUDA lowering carries the device scratch
        out = jax.lax.platform_dependent(*operands, cpu=lambda *rows: call(rows, None),
                                         cuda=lambda *rows: call(rows, scratch))
    arrays = {name: out[k] for k, name in enumerate(specs) if reg.layout.present(name)}
    return arrays, out[len(specs)]


def encode(
    encoder: Any,
    states: Any,
    task_ids: Any = None,
    *,
    derived: Any = None,
    goals: Any = None,
    capacities: Optional[dict] = None,
    threads: int = 1,
) -> JaxFlatBatch:
    """``FlatRelationEncoder.encode`` of the rows ``states`` as an XLA FFI call: the CPU encoder on the CPU, the device
    kernels on CUDA, with equal arrays. Usable inside ``jax.jit``, ``lax.scan`` and ``jax.vmap`` (each vmapped batch
    is encoded on its own, with the same capacities). ``encoder`` is a ``FlatRelationEncoder`` (registered on first
    use) or the handle :func:`register` returned, which a function can hold instead of the encoder.

    states: uint32 [N, 2 W] (the u64 words of each row as halves, low first), or uint64 / int64 [N, W] words, or one row.
    task_ids: int32 [N], the instance of each row (None: instance 0; required for a table of several instances).
    derived: the derived words of the rows, [N, 2 D] halves or [N, D] words, D at least the table's derived words;
    required when a graph reads derived atoms. goals: ``(goal_pos, goal_neg)`` or ``(goal_pos, goal_neg,
    goal_derived_pos, goal_derived_neg)``, each with N rows or one row for all (halves or words), replacing the
    instances' goals as in ``FlatRelationEncoder.encode``.

    capacities: ``dict(graphs=, entities=, args=, targets=)`` (``targets`` only with target sources; ``graphs`` at least
    N), required under ``jax.jit``: the arrays have static shapes, the true sizes are in ``batch.device_sizes`` and a
    batch that does not fit is all zero with its flag set. Without capacities (eager calls only) the call waits once
    for the batch's sizes, raises ``ValueError`` for an invalid row and ``OverflowError`` when the index type is too
    narrow, and returns exact arrays.

    threads: the CPU target's worker threads (0: one per hardware thread); the result does not depend on it.
    ``index_dtype='int64'`` needs ``jax_enable_x64``."""
    _ffi.register()
    handle, reg = _registration(encoder)
    info = reg.info
    if info["device_refusal"]:
        raise ValueError(info["device_refusal"])
    if reg.options["index_dtype"] == "int64" and not jax.config.read("jax_enable_x64"):
        raise _error("index_dtype='int64' needs 64-bit JAX arrays: enable jax_enable_x64")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 0:
        raise _error(f"threads must be an int of at least 0, got {threads!r}")
    rows = _halves(states, "states", True)
    graphs = int(rows.shape[0])
    ids = _task_ids(task_ids, graphs, info["instances"])
    goal_pos, goal_neg, goal_derived_pos, goal_derived_neg = _goals(goals, graphs)
    if goals is not None and not reg.layout.goal_complete:
        raise _error("goals are given, but the schema has no goal relations")
    if derived is None:
        need = info["needs_derived"]
        if need and need[0 if goals is None else (1 if len(goals) == 2 else 2)]:
            raise _error("the rows' derived words are required: a graph reads derived atoms whose axioms the device "
                         "cannot evaluate (pass derived=)")
        derived_words = jnp.zeros((0,), jnp.uint32)
    else:
        derived_words = _halves(derived, "derived", True)
        if derived_words.shape[0] != graphs:
            raise _error(f"'derived' has {derived_words.shape[0]} rows, the states {graphs}")
        if derived_words.shape[1] // 2 < info["derived_words"]:
            raise _error(f"'derived' needs at least {info['derived_words']} words per row")
    operands = (rows, ids, derived_words, goal_pos, goal_neg, goal_derived_pos, goal_derived_neg)
    static = (
        tuple(n for n in _ARRAYS if reg.layout.present(n)),
        reg.layout.relations,
        1 if reg.options["layout"] == "relation" else 0,
        1 if reg.options["index_dtype"] == "int64" else 0,
        _signed(reg.fingerprint),
        bool(not reg.layout.entity_arrays),
        _SchemaRef(reg.schema, handle),
        _signed(reg.table_fingerprint),
    )
    if capacities is not None:
        caps_graphs, entities, args, targets = _device._capacities(capacities, reg.layout)
        if caps_graphs < graphs:
            raise _error(f"the batch has {graphs} graphs, more than the capacity {caps_graphs}")
        arrays, sizes = _call(handle, reg, operands, graphs, entities, args, targets, threads)
        return JaxFlatBatch(arrays, sizes, static)
    if _traced(*operands):
        raise _error("capacities are required under jax.jit and other transformations (the arrays need static shapes): "
                     "pass capacities=dict(graphs=, entities=, args=, targets=)")
    arrays, sizes = _eager(handle, reg, operands, graphs, threads)
    cut = {"entity_role": "entities", "entity_ref": "entities", "relation_args": "args", "target_entity": "targets",
           "target_group": "targets"}  # fmt: skip
    arrays = {name: a[: sizes[cut[name]]] if name in cut else a for name, a in arrays.items()}
    return JaxFlatBatch(arrays, None, static)


def _eager(handle: int, reg: _Registration, operands: tuple, graphs: int, threads: int) -> tuple[dict[str, Any], dict]:
    """The arrays and sizes of an eager call without capacities. The call runs at the capacities held for this encoder
    and number of rows (the first call counts the batch at zero capacities to find them), reads its sizes and, if the
    batch did not fit, runs once more at capacities that hold it; the held capacities only grow, so that batches of
    similar sizes reuse one compiled call."""
    with _lock:
        held = reg.held.get(graphs)
    if held is None:
        _, counted = _call(handle, reg, operands, graphs, 0, 0, 0, threads)
        sizes = _host_sizes(np.asarray(counted))
        _check_sizes(sizes)
        held = (_rounded(sizes["entities"]), _rounded(sizes["args"]), _rounded(sizes["targets"]))
    while True:
        arrays, words = _call(handle, reg, operands, graphs, *held, threads)
        sizes = _host_sizes(np.asarray(words))
        _check_sizes(sizes)
        if not sizes["capacity_overflow"]:
            break
        held = tuple(max(h, _rounded(sizes[key])) for h, key in zip(held, ("entities", "args", "targets")))
    with _lock:
        old = reg.held.get(graphs, (0, 0, 0))
        reg.held[graphs] = tuple(max(a, b) for a, b in zip(old, held))
    return arrays, sizes


def _check_sizes(sizes: dict) -> None:
    """Raises for a batch with an invalid row or one that its index type cannot hold."""
    if sizes["invalid_input"]:
        raise _error("a row is invalid: a task id outside the table, a state row or goal mask narrower than its "
                     "instance, or a bit that its instance has no atom for")
    if sizes["index_overflow"]:
        raise OverflowError(
            f"mymyr: encoding: the batch needs more indices than its index type holds (entities {sizes['entities']}, "
            f"tuples {sizes['tuples']}, args {sizes['args']}, targets {sizes['targets']}): use index_dtype='int64'"
        )


def _rounded(n: int) -> int:
    return 0 if n == 0 else max(64, 1 << (n - 1).bit_length())
