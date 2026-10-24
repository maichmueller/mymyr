"""Observation payload bytes and captured device encode time per state.

Run from the repository root with a CUDA build and torch, for example:
  CUDA_VISIBLE_DEVICES=0 .venv-rl/bin/python bench/python/encoding_bytes.py --tasks blocks-p01e --rows 1024
The default tasks are the encoding GPU benchmark tasks; --data names their IPC directory. Missing tasks are reported.
The numeric fixture is included with --numeric. CPU-only builds report host bytes. Capacity outputs reserve 25% extra
variable entries, with graph slots fixed by the input row count. HGraph reports its backing batch; requested ptr and
edge-index arrays are excluded. Device timings cover encoding prepared inputs with capacities, excluding expansion,
HGraph metadata gathering, allocation and host dispatch, as median CUDA event time over captured replays.
"""

import argparse
from dataclasses import fields
import gc
import json
from pathlib import Path
import statistics

import numpy as np

import mymyr
from mymyr import encoding, rl
from bench_encoding_gpu import TASKS, load_task, tile, walk_rows

ROOT = Path(__file__).resolve().parents[2]
FORMS = ("relations", "coo", "shared", "hgraph", "simple", "both", "actions", "horizon", "delta")


def capacity_of(batch, headroom):
    arrays = batch.as_dict()
    grow = lambda size: int(np.ceil(size * headroom))
    return dict(graphs=batch.num_graphs, entities=grow(int(batch.entity_offsets[-1])),
                args=grow(len(batch.relation_args) if "relation_args" in arrays else int(batch.edge_offsets[-1]) // 2),
                **({"targets": grow(len(batch.target_entity))} if "target_entity" in arrays else {}),
                **({"numeric": grow(len(batch.numeric_values))} if "numeric_values" in arrays else {}),
                **({"static_instances": len(batch.static_task_ids), "static_args": grow(len(batch.static_args))}
                   if "static_args" in arrays else {}))


def prepare(task, rows, form):
    cls = encoding.FlatRelationEncoder
    options = dict(goal_derivations=())
    kwargs = {}
    if form == "coo":
        options.update(emit_counts=False, entity_arrays="auto", include_actions=False)
    elif form == "shared":
        options["statics"] = "shared"
    elif form == "hgraph":
        options["layout"] = "relation"
    elif form == "simple":
        cls = encoding.SimpleEncoder
    elif form == "both":
        options["outputs"] = ("relations", "simple")
    elif form == "actions":
        options["target_sources"] = ("action",)
        expansion = rl.expand(task, rows)
        kwargs["actions"] = expansion
    elif form in ("horizon", "delta"):
        cls = encoding.FlatHorizonEncoder
        options["horizon_mode"] = "delta" if form == "delta" else "full"
        rows = encoding.horizon(task, rows, depth=1)
    return cls(task, **options), rows, kwargs


def upload(torch, array):
    array = np.asarray(array)
    if array.dtype == np.uint64:
        array = array.view(np.int64)
    return torch.as_tensor(array.copy(), device="cuda:0")


def device_inputs(torch, rows, kwargs):
    if isinstance(rows, encoding.HorizonView):
        rows = encoding.HorizonView(**{f.name: None if getattr(rows, f.name) is None else upload(torch, getattr(rows, f.name))
                                       for f in fields(rows)})
    else:
        rows = upload(torch, rows)
    return rows, {key: tuple(upload(torch, a) for a in ((value.schema, value.binding, value.offsets)
                                                     if key == "actions" else value))
                  for key, value in kwargs.items()}


def device_time(torch, call, rows, rounds):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        batch = call()
    samples = []
    with torch.cuda.stream(stream):
        for _ in range(rounds):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1e6 / (10 * rows))
    if batch.sizes["overflow"]:
        raise RuntimeError("timed batch overflowed")
    return statistics.median(samples)


def measure(task, rows, form, args, torch):
    encoder, inputs, kwargs = prepare(task, rows, form)
    exact = encoder.encode(inputs, **kwargs)
    capacities = capacity_of(exact, args.headroom)
    bounded = encoder.encode(inputs, capacities=capacities, **kwargs)
    record = dict(cpu_exact=exact.nbytes / args.rows, cpu_capacity=bounded.nbytes / args.rows,
                  capacities=capacities, options=encoder.options,
                  cpu_exact_arrays=exact.nbytes_by_array, cpu_capacity_arrays=bounded.nbytes_by_array)
    if form == "hgraph":
        assert encoding.hgraph(exact).nbytes == exact.nbytes
    del exact, bounded
    if torch is not None:
        device_rows, device_kwargs = device_inputs(torch, inputs, kwargs)
        try:
            device_exact = encoder.encode(device_rows, **device_kwargs)
            device_capacity = encoder.encode(device_rows, capacities=capacities, **device_kwargs)
        except ValueError as error:
            # Unsupported output forms are reported rather than assigned misleading byte or time values.
            if "numeric" not in str(error) or form not in ("simple", "both", "actions", "horizon", "delta"):
                raise
            record["device_unavailable"] = str(error)
            return record
        record.update(device_exact=device_exact.nbytes / args.rows, device_capacity=device_capacity.nbytes / args.rows,
                      device_exact_arrays=device_exact.nbytes_by_array, device_capacity_arrays=device_capacity.nbytes_by_array)
        if form == "hgraph":
            assert encoding.hgraph(device_exact).nbytes == device_exact.nbytes
        torch.cuda.synchronize()
        del device_exact, device_capacity
        torch.cuda.empty_cache()
        record["device_ns"] = device_time(torch, lambda: encoder.encode(device_rows, capacities=capacities,
                                                                       **device_kwargs), args.rows, args.rounds)
        torch.cuda.synchronize()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=[*TASKS, "blocks-8"], default=list(TASKS))
    parser.add_argument("--forms", nargs="+", choices=FORMS, default=FORMS)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--headroom", type=float, default=1.25)
    parser.add_argument("--numeric", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.rows < 1 or args.rounds < 1 or args.headroom < 1:
        parser.error("rows and rounds must be positive, headroom at least 1")
    torch = None
    if not args.cpu:
        try:
            import torch
            import mymyr.cuda as cuda
            if not cuda.available() or not torch.cuda.is_available():
                torch = None
        except ImportError:
            torch = None
    metadata = dict(gpu=None if torch is None else torch.cuda.get_device_name(0), rows=args.rows,
                    rounds=args.rounds, headroom=args.headroom, goal_derivations=[], index_dtype="int32",
                    horizon_depth=1, seed=0, walk_rows=8, mymyr_version=mymyr.__version__)
    print(json.dumps(metadata))
    print("task form CPU-exact-B/state CPU-cap-B/state device-exact-B/state device-cap-B/state device-ns/state")
    records = []
    for key in [*args.tasks, *(["numeric-counters"] if args.numeric else [])]:
        if key == "numeric-counters":
            task, reason = mymyr.Task.from_text(str(ROOT / "tests/data/numeric_tasks/cs-counters.txt")), None
        else:
            task, reason = load_task(key, args.data)
        if task is None:
            print(f"{key}: unavailable: {reason}")
            records.append(dict(task=key, unavailable=reason))
            continue
        rows = tile(walk_rows(task, 8, seed=0, depth=16, patience=2), args.rows)
        for form in args.forms:
            if key == "numeric-counters" and form in ("actions", "horizon", "delta"):
                continue
            record = dict(task=key, form=form, **measure(task, rows, form, args, torch))
            records.append(record)
            values = [record.get(name) for name in ("cpu_exact", "cpu_capacity", "device_exact", "device_capacity", "device_ns")]
            print(f"{key} {form} " + " ".join("-" if value is None else f"{value:.2f}" for value in values), flush=True)
            gc.collect()
            if torch is not None:
                torch.cuda.empty_cache()
    if args.json:
        args.json.write_text(json.dumps(dict(environment=metadata, records=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
