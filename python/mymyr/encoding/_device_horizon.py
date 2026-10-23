"""Device expansion labels and horizon arrays, in the encoder's stream order."""

from __future__ import annotations

import threading
import weakref
from types import SimpleNamespace
from typing import Any

from mymyr.encoding import _device as d
from mymyr.rl import as_table


_COLUMNS = ("schema", "binding", "offsets", "parent", "succ", "derived")

# The columns of device Expansions imported outside capture, for encodes captured into CUDA graphs later: an import
# makes the current stream wait for the expansion's producer, which a capture cannot do.
_imported: "weakref.WeakKeyDictionary[Any, SimpleNamespace]" = weakref.WeakKeyDictionary()
_imported_lock = threading.Lock()


def _expansion_type() -> Any:
    try:
        from mymyr.cuda import DeviceExpansion
    except ImportError:
        return None
    return DeviceExpansion


def _integer(torch: Any, value: Any, name: str, device: Any, keep: list, ndim: int) -> Any:
    if hasattr(value, "__dlpack_device__") and value.__dlpack_device__()[0] != 2:
        raise d._error(f"'{name}' must be a CUDA array on {device}")
    if not isinstance(value, torch.Tensor) and torch.cuda.is_current_stream_capturing():
        raise d._error(f"'{name}' is not a torch tensor: during CUDA graph capture, the arrays of an encode are torch "
                       "tensors or the columns of an expansion that an encode used before the capture")
    t = d._cuda_tensor(torch, value, name, keep)
    if t.device != device or t.dtype != torch.int32 or t.ndim != ndim or not t.is_contiguous():
        raise d._error(f"'{name}' must be a contiguous int32 CUDA tensor with {ndim} dimensions on {device}")
    keep.append(t)
    return t


def _expansion_columns(torch: Any, expansion: Any) -> SimpleNamespace:
    """A device Expansion's columns as torch tensors. Outside capture they are imported (the current stream waits for
    the expansion) and kept with the expansion; during capture the kept columns serve."""
    if torch.cuda.is_current_stream_capturing():
        with _imported_lock:
            columns = _imported.get(expansion)
        if columns is None:
            raise d._error("the expansion's columns were never imported: encode with it once before the CUDA graph "
                           "capture (a warm-up), or pass its columns as torch tensors")
        return columns
    keep: list = []
    columns = SimpleNamespace(table=expansion.table, overflow=expansion.overflow, status=None)
    for name in _COLUMNS:
        value = getattr(expansion, name)
        setattr(columns, name, None if value is None else d._cuda_tensor(torch, value, f"actions.{name}", keep))
    with _imported_lock:
        _imported[expansion] = columns
    return columns


def columns_of(torch: Any, actions: Any) -> SimpleNamespace:
    """The columns of a device expansion, its (schema, binding, offsets), or expand_flat's six outputs."""
    if isinstance(actions, (tuple, list)):
        if len(actions) == 3:
            return SimpleNamespace(schema=actions[0], binding=actions[1], offsets=actions[2], parent=None, succ=None,
                                   derived=None, status=None)
        if len(actions) == 6:
            return SimpleNamespace(succ=actions[0], parent=actions[1], schema=actions[2], binding=actions[3],
                                   offsets=actions[5], derived=None, status=None)
        raise d._error("device actions are an expansion, (schema, binding, offsets), or expand_flat's six outputs")
    expansion = _expansion_type()
    if expansion is not None and isinstance(actions, expansion):
        return _expansion_columns(torch, actions)
    return actions


def action_arrays(encoder: Any, actions: Any, device: Any, graphs: int, keep: list) -> Any:
    torch = d._torch()
    actions = columns_of(torch, actions)
    if getattr(actions, "overflow", False):
        raise d._error("the expansion overflowed")
    if hasattr(actions, "table") and as_table(actions.table).fingerprint != as_table(encoder.table).fingerprint:
        raise d._error("the action expansion belongs to another table")
    arrays = {}
    for name, ndim in (("schema", 1), ("binding", 2), ("offsets", 1)):
        if getattr(actions, name, None) is None:
            raise d._error(f"the device expansion needs '{name}'")
        arrays[name] = _integer(torch, getattr(actions, name), f"actions.{name}", device, keep, ndim)
    parent = getattr(actions, "parent", None)
    arrays["parent"] = None if parent is None else _integer(torch, parent, "actions.parent", device, keep, 1)
    capacity = arrays["schema"].shape[0]
    if arrays["offsets"].shape != (graphs + 1,) or arrays["binding"].shape[0] != capacity or (
        arrays["parent"] is not None and arrays["parent"].shape != (capacity,)
    ):
        raise d._error("the expansion needs G + 1 offsets and equally many schema, binding and parent rows")
    if arrays["binding"].shape[1] < as_table(encoder.table).label_width:
        raise d._error("the action label width is narrower than the table's")
    return SimpleNamespace(**arrays, capacity=capacity, succ=getattr(actions, "succ", None),
                           derived=getattr(actions, "derived", None), status=getattr(actions, "status", None))


def bind_actions(encoder: Any, rows: Any, actions: Any) -> None:
    if not encoder.options["include_actions"]:
        raise d._error("actions are given, but the schema has no action relations")
    a = action_arrays(encoder, actions, rows.device, rows.graphs, rows.tensors)
    rows.fields[35:42] = [a.offsets.data_ptr(), a.schema.data_ptr(), a.binding.data_ptr(),
                          0 if a.parent is None else a.parent.data_ptr(), a.capacity, a.binding.shape[1], 0]


def horizon_rows(encoder: Any, view: Any, derived: Any, goals: Any, subgoals: Any, status: Any = None) -> Any:
    """The row fields of a horizon view; ``status`` is the device status word of the expansion that produced it."""
    torch = d._torch()
    keep: list = []
    nodes = d._words(torch, d._cuda_tensor(torch, view.nodes, "nodes", keep), "nodes")
    device = nodes.device
    no = _integer(torch, view.node_offsets, "node_offsets", device, keep, 1)
    eo = _integer(torch, view.edge_offsets, "edge_offsets", device, keep, 1)
    graphs = no.shape[0] - 1
    if graphs < 0 or eo.shape != no.shape or nodes.shape[0] < graphs:
        raise d._error("horizon offsets need G + 1 entries and at least one node per graph")
    ids = getattr(view, "task_ids", None)
    if ids is not None:
        ids = _integer(torch, ids, "task_ids", device, keep, 1)
    g = d._cuda_tuple(torch, goals, "goals", keep)
    sub = d._cuda_tuple(torch, subgoals, "subgoals", keep)
    rows = d._Rows(torch, nodes[:graphs], ids, None, g, sub)
    rows.fields[1:4] = [nodes.data_ptr(), d._stride(nodes), nodes.shape[1]]
    if derived is not None:
        words = d._words(torch, d._cuda_tensor(torch, derived, "derived", keep), "derived")
        if words.device != device or words.shape[0] != nodes.shape[0]:
            raise d._error("derived words need one row per horizon node on the nodes' device")
        rows.fields[5:7] = [words.data_ptr(), d._stride(words)]
        keep.append(words)
    edge_arrays = []
    edges = None
    for name, ndim in (("edge_src", 1), ("edge_dst", 1), ("edge_schema", 1), ("edge_binding", 2)):
        value = getattr(view, name, None)
        t = None if value is None else _integer(torch, value, name, device, keep, ndim)
        if t is not None:
            if edges is not None and edges != t.shape[0]:
                raise d._error("horizon edge arrays need equally many rows")
            edges = t.shape[0]
        edge_arrays.append(t)
    edges = 0 if edges is None else edges
    width = 0 if edge_arrays[3] is None else edge_arrays[3].shape[1]
    if encoder.options["include_actions"] and edge_arrays[3] is None:
        width = as_table(encoder.table).label_width
    size = 6 * graphs + 8 * nodes.shape[0] + 5 * edges
    scratch = torch.empty(max(1, size), dtype=torch.int32, device=device)
    rows.fields[42:54] = [no.data_ptr(), eo.data_ptr(),
                          *(0 if a is None else a.data_ptr() for a in edge_arrays),
                          nodes.shape[0], edges, width, scratch.data_ptr(), scratch.numel(),
                          0 if status is None else status.data_ptr()]
    rows.tensors.extend([nodes, scratch, *keep, *([] if status is None else [status])])
    return rows


def encode_horizon(encoder: Any, view: Any, derived: Any, goals: Any, subgoals: Any,
                   framework: Any, threads: Any, pool: Any, capacities: Any, status: Any = None) -> Any:
    if pool is not None:
        raise d._error("pool is a pool of host threads: it has no meaning for rows on the device")
    if framework not in (None, "torch", "pytorch"):
        raise d._error("device horizons are encoded into torch tensors: framework must be 'torch' or None")
    torch = d._torch()
    d._drain(torch)
    layout = d.layout_of(encoder)
    caps = None if capacities is None else d._capacities(capacities, layout)
    rows = horizon_rows(encoder, view, derived, goals, subgoals, status)
    arrays = d._encode_rows(encoder, rows, caps, rows.tensors)
    return d.DeviceFlatBatch(encoder, arrays, exact=caps is None)


def build_horizon(encoder: Any, parents: Any, task_ids: Any = None, actions: Any = None, *, merge: bool = True) -> Any:
    """A compact, first-seen depth-one view from device parents and their expansion."""
    import mymyr.rl as rl
    from mymyr.encoding._horizon import HorizonView
    from mymyr.rl.torch._ops import _stream

    torch = d._torch()
    d._drain(torch)
    if as_table(encoder.table).numeric:
        raise d._error("horizon encodings do not encode numeric state rows")
    keep: list = []
    states = d._words(torch, d._cuda_tensor(torch, parents, "states", keep), "states")
    ids = None if task_ids is None else d._cuda_tensor(torch, task_ids, "task_ids", keep)
    rows = d._Rows(torch, states, ids, None, None)
    if actions is None:
        actions = rl.expand(encoder.table, states, ids, framework="torch", stream=_stream(states))
    a = action_arrays(encoder, actions, states.device, rows.graphs, keep)
    if a.succ is None:
        raise d._error("transition actions need successor rows as well as labels and offsets")
    if not isinstance(a.succ, torch.Tensor) and torch.cuda.is_current_stream_capturing():
        raise d._error("'actions.succ' is not a torch tensor: during CUDA graph capture, the arrays of an encode are "
                       "torch tensors or the columns of an expansion that an encode used before the capture")
    children = d._words(torch, d._cuda_tensor(torch, a.succ, "actions.succ", keep), "actions.succ")
    if children.device != states.device or children.shape[0] != a.capacity:
        raise d._error("the successor rows need the parents' device and the label arrays' row count")
    width = max(states.shape[1], children.shape[1])
    if states.shape[1] < width:
        states = torch.nn.functional.pad(states, (0, width - states.shape[1]))
        rows = d._Rows(torch, states, ids, None, None)
    if children.shape[1] < width:
        children = torch.nn.functional.pad(children, (0, width - children.shape[1]))
    rows.fields[35:42] = [a.offsets.data_ptr(), a.schema.data_ptr(), a.binding.data_ptr(),
                          0 if a.parent is None else a.parent.data_ptr(), a.capacity, a.binding.shape[1], 0]
    nodes = torch.empty((rows.graphs + a.capacity, states.shape[1]), dtype=torch.int64, device=states.device)
    no = torch.empty(rows.graphs + 1, dtype=torch.int32, device=states.device)
    src = torch.empty(a.capacity, dtype=torch.int32, device=states.device)
    dst = torch.empty_like(src)
    schema, binding = torch.empty_like(a.schema), torch.empty_like(a.binding)
    native = d._device_encoder(encoder, states.device.index)
    scratch = torch.empty(native.horizon_scratch_bytes(a.capacity, merge), dtype=torch.uint8, device=states.device)
    native.build_horizon(_stream(states), rows.fields, children.data_ptr(), d._stride(children),
                         [nodes.data_ptr(), no.data_ptr(), src.data_ptr(), dst.data_ptr(), schema.data_ptr(), binding.data_ptr()], merge,
                         scratch.data_ptr(), scratch.numel())
    d._retain(torch, [actions, children, scratch, *rows.tensors, *keep], states.device)
    return HorizonView(nodes, no, a.offsets, src, dst, schema, binding, ids)


def _bounded_expansion(encoder: Any, states: Any, task_ids: Any, derived: bool, capacities: Any) -> SimpleNamespace:
    """The expansion of the parents into arrays of the entity capacity's rows, without waiting for the device: every
    successor is a node with a state entity of its own, so an encoding within the capacities has at most that many.
    Its total and offsets stay on the device; its status word (bit 0: more successors than rows, bit 1: an invalid
    row) reaches the encoding's overflow flags."""
    import mymyr.rl as rl
    from mymyr.rl.torch._ops import _stream

    torch = d._torch()
    keep: list = []
    rows = d._words(torch, d._cuda_tensor(torch, states, "states", keep), "states")
    ids = None if task_ids is None else d._cuda_tensor(torch, task_ids, "task_ids", keep)
    table = as_table(encoder.table)
    m = d._capacities(capacities, d.layout_of(encoder))[1]
    int32 = dict(dtype=torch.int32, device=rows.device)
    x = SimpleNamespace(
        succ=torch.empty((m, rows.shape[1]), dtype=torch.int64, device=rows.device),
        parent=torch.empty(m, **int32), schema=torch.empty(m, **int32),
        binding=torch.empty((m, table.label_width), **int32), offsets=torch.empty(rows.shape[0] + 1, **int32),
        derived=torch.empty((m, table.derived_words), dtype=torch.int64, device=rows.device) if derived else None,
        status=torch.empty(1, **int32),
    )
    rl.expand_into(encoder.table, rows, ids, succ=x.succ, parent=x.parent, schema=x.schema, binding=x.binding,
                   derived=x.derived, offsets=x.offsets, status=x.status, stream=_stream(rows))
    d._retain(torch, keep, rows.device)
    return x


def encode_transitions(encoder: Any, states: Any, task_ids: Any, derived: Any, goals: Any, actions: Any,
                       subgoals: Any, framework: Any, threads: Any, pool: Any, capacities: Any) -> Any:
    if pool is not None or framework not in (None, "torch", "pytorch"):
        raise d._error("device transitions need torch output and no host thread pool")
    torch = d._torch()
    if actions is None and torch.cuda.is_current_stream_capturing():
        raise d._error("transitions expand their successors outside CUDA graph capture: to capture an encode, pass "
                       "actions= (an expansion made before the capture)")
    if actions is None and capacities is not None:
        actions = _bounded_expansion(encoder, states, task_ids, derived is not None, capacities)
    elif actions is None and derived is not None:
        import mymyr.rl as rl
        from mymyr.rl.torch._ops import _stream
        actions = rl.expand(encoder.table, states, task_ids, derived=True, framework="torch", stream=_stream(states))
    actions = columns_of(torch, actions)
    view = build_horizon(encoder, states, task_ids, actions, merge=False)
    node_derived = None
    if derived is not None:
        children = getattr(actions, "derived", None)
        if children is None:
            raise d._error("derived needs successors that hold their derived atoms: expand with derived=True")
        labels = SimpleNamespace(succ=children, schema=actions.schema, binding=actions.binding,
                                 parent=getattr(actions, "parent", None), offsets=actions.offsets)
        node_derived = build_horizon(encoder, derived, task_ids, labels, merge=False).nodes
    return encode_horizon(encoder, view, node_derived, goals, subgoals, framework, threads, pool, capacities,
                          getattr(actions, "status", None))


def build_space_horizons(space: Any, roots: Any, depth: int, max_nodes: int | None,
                         truncate: bool, merge: bool) -> Any:
    """Traverse recorded unique state ids into CUDA horizon arrays without reading state words on the host."""
    import operator

    from mymyr.encoding import FlatHorizonEncoder, HorizonView
    from mymyr.rl.torch._ops import _stream

    torch = d._torch()
    d._drain(torch)
    depth = operator.index(depth)
    if not 0 <= depth <= 0xffffffff:
        raise d._error("depth must be between 0 and 4294967295")
    if not merge:
        raise d._error("device state-space horizons traverse recorded unique ids and require merge=True")
    if space.numeric_words:
        raise d._error("horizons do not encode numeric state spaces")
    if not space.has_labels:
        raise d._error("the state space needs transition labels: generate it with labels=True")
    limit = 0 if max_nodes is None else operator.index(max_nodes)
    if max_nodes is not None and not 1 <= limit <= 0x7fffffff:
        raise d._error("max_nodes must be between 1 and 2147483647")
    device = torch.device("cuda", space.device)
    keep: list = []
    if roots is None:
        ids = torch.arange(space.num_states, dtype=torch.int32, device=device)
    elif hasattr(roots, "__dlpack_device__") and roots.__dlpack_device__()[0] == 2:
        ids = _integer(torch, roots, "roots", device, keep, 1)
    else:
        values = [operator.index(x) for x in roots]
        if any(x < 0 or x >= space.num_states for x in values):
            raise d._error("a root is outside the state space")
        ids = torch.tensor(values, dtype=torch.int32, device=device)
    graphs = ids.numel()
    nodes = graphs * min(limit or space.num_states, space.num_states)
    edges = graphs * space.num_transitions
    if max(graphs, nodes, edges) > 0x7fffffff:
        raise d._error("state-space horizon capacities must fit int32 offsets")
    arrays = space.arrays(framework="torch")
    rows = d._words(torch, arrays["state_words"], "state_words")
    output_nodes = torch.empty((nodes, space.words), dtype=torch.int64, device=device)
    no = torch.empty(graphs + 1, dtype=torch.int32, device=device)
    eo = torch.empty_like(no)
    src = torch.empty(edges, dtype=torch.int32, device=device)
    dst, schema = torch.empty_like(src), torch.empty_like(src)
    binding = torch.empty((edges, space.label_width), dtype=torch.int32, device=device)
    truncated = torch.zeros(graphs, dtype=torch.uint8, device=device)
    encoder = FlatHorizonEncoder(space.task, horizon_mode="full")
    native = d._device_encoder(encoder, space.device)
    scratch = torch.empty(native.space_horizon_scratch_bytes(space.num_states, space.num_transitions, graphs),
                          dtype=torch.uint8, device=device)
    inputs = [rows, arrays["forward_offsets"], arrays["forward_targets"], arrays["label_schemas"],
              arrays["label_bindings"], ids]
    outputs = [output_nodes, no, eo, src, dst, schema, binding, scratch, truncated]
    native.build_space_horizons(_stream(rows), [x.data_ptr() for x in inputs], [x.data_ptr() for x in outputs],
                               space.num_states, space.num_transitions, space.words, space.label_width,
                               graphs, depth, limit, truncate, scratch.numel())
    d._retain(torch, [space, arrays, *inputs, *outputs, *keep], device)
    return HorizonView(output_nodes, no, eo, src, dst, schema, binding, truncated=truncated)
