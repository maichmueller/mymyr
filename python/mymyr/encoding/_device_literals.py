"""Literal lookup and mask construction on torch's current CUDA stream."""

import operator

import torch

from mymyr.rl import as_table
from . import _device


def _tensor(value, name, dtype, shape, device):
    if not isinstance(value, torch.Tensor) or value.device != device or value.dtype != dtype:
        raise TypeError(f"mymyr: encoding: {name} must be a {dtype} tensor on {device}")
    if value.shape != shape:
        raise ValueError(f"mymyr: encoding: {name} has shape {value.shape}, expected {shape}")
    if not value.is_contiguous():
        raise ValueError(f"mymyr: encoding: {name} must be contiguous")


def _inputs(encoder, predicates, objects, task_ids, rows):
    if not hasattr(encoder, "schema") or not hasattr(encoder, "table"):
        raise TypeError("mymyr: encoding: CUDA literals require an encoder")
    if not isinstance(predicates, torch.Tensor) or not predicates.is_cuda or predicates.ndim != 1:
        raise TypeError("mymyr: encoding: predicates must be a CUDA int32 tensor [N]")
    device = predicates.device
    count = predicates.shape[0]
    _tensor(predicates, "predicates", torch.int32, (count,), device)
    if not isinstance(objects, torch.Tensor) or objects.ndim != 2:
        raise ValueError("mymyr: encoding: objects must have shape [N, k]")
    _tensor(objects, "objects", torch.int32, (count, objects.shape[1]), device)
    table = as_table(encoder.table)
    if task_ids is not None:
        _tensor(task_ids, "task_ids", torch.int32, (rows,), device)
    elif rows and len(table) != 1:
        raise ValueError("mymyr: encoding: task_ids are required for a table of several instances")
    return device, count, table


def _pointers(arrays):
    return [0 if array is None else array.data_ptr() for array in arrays]


def _record(arrays, stream):
    for array in arrays:
        if array is not None:
            array.record_stream(stream)


def slots(encoder, predicates, objects, task_ids=None):
    device, count, _ = _inputs(encoder, predicates, objects, task_ids, predicates.shape[0])
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        native = _device._device_encoder(encoder, device.index)
        kinds = torch.empty((count,), dtype=torch.uint8, device=device)
        slots = torch.empty((count,), dtype=torch.int32, device=device)
        status = torch.zeros((), dtype=torch.int64, device=device)
        inputs = [predicates, objects, task_ids]
        outputs = [kinds, slots, status]
        native.literal_slots(stream.cuda_stream, count, objects.shape[1], _pointers(inputs), _pointers(outputs))
        _record(inputs, stream)
    return kinds, slots, status


def masks(encoder, predicates, objects, positive, rows, num_rows, task_ids=None, layers=None, num_layers=1):
    num_rows, num_layers = operator.index(num_rows), operator.index(num_layers)
    if not 0 <= num_rows <= 0xffffffff or not 0 <= num_layers <= 0xffffffff:
        raise ValueError("mymyr: encoding: num_rows and num_layers must be in [0, 2**32)")
    if layers is None and num_layers != 1:
        raise ValueError("mymyr: encoding: num_layers must be 1 without layers")
    device, count, table = _inputs(encoder, predicates, objects, task_ids, num_rows)
    for value, name, dtype in ((positive, "positive", torch.bool), (rows, "rows", torch.int32)):
        _tensor(value, name, dtype, (count,), device)
    if layers is not None:
        _tensor(layers, "layers", torch.int32, (count,), device)
    shape = (num_rows,) if layers is None else (num_rows, num_layers)
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        native = _device._device_encoder(encoder, device.index)
        outputs = [torch.zeros((*shape, words), dtype=torch.int64, device=device)
                   for words in (table.words, table.words, table.derived_words, table.derived_words)]
        unsatisfiable = torch.zeros(shape, dtype=torch.int32, device=device)
        status = torch.zeros((), dtype=torch.int64, device=device)
        inputs = [predicates, objects, task_ids, positive, rows, layers]
        native.literal_masks(stream.cuda_stream, count, objects.shape[1], num_rows, num_layers,
                             _pointers(inputs), _pointers([*outputs, unsatisfiable, status]))
        _record(inputs, stream)
    return [*outputs, unsatisfiable.bool(), status]
