"""The HGraph view of a relation-major batch."""

from __future__ import annotations

from mymyr._core._encoding import FlatBatch, HGraphView
from mymyr.encoding._device import DeviceFlatBatch


def hgraph(batch: FlatBatch) -> HGraphView:
    """The HGraph view of ``batch``: its tuples as a heterogeneous graph, without copying them (see ``HGraphView``).

    ``batch`` is a FlatBatch of the single-state, horizon, rooted-horizon or transition encoders, relation-major
    (``layout='relation'``), encoded without ``predicate_virtual_nodes``. The view keeps the batch alive. Raises
    TypeError for a batch on the device and ValueError for a batch the view cannot be made of: one that is graph-major,
    has predicate virtual nodes or overflowed its capacities.
    """
    if isinstance(batch, DeviceFlatBatch):
        raise TypeError("mymyr: encoding: the HGraph view is of host batches; got a DeviceFlatBatch")
    if isinstance(batch, FlatBatch) and batch.sizes is not None and batch.sizes["overflow"]:
        raise ValueError("mymyr: encoding: the batch overflowed its capacities, so it holds no encoding to view")
    return HGraphView(batch)
