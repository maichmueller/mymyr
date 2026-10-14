"""bench/python/bench_encoding_gpu.py: the command line runs and writes the documented JSON without CUDA, the walk rows
are deterministic and distinct, and a gate with a skipped or errored task or mode is reported as incomplete, error or n/a, never as a pass."""

import contextlib
import gc
import importlib.util
import json
import subprocess
import sys
import types

import numpy as np
import pytest

from conftest import ROOT, text_task

SCRIPT = ROOT / "bench/python/bench_encoding_gpu.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_encoding_gpu", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_script(tmp_path, *args):
    out = tmp_path / "bench.json"
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--quick", "--json", str(out), *args], capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, done.stdout + done.stderr
    return json.loads(out.read_text()), done.stdout


def test_quick_cpu_run_writes_the_documented_json_and_a_gate_table(tmp_path, bench):
    result, stdout = run_script(tmp_path, "--modes", "cpu", "--rounds", "2")
    assert {"environment", "config", "targets", "baselines", "records", "gates", "gate_details"} <= set(result)
    environment = result["environment"]
    assert {"mymyr_version", "mymyr_commit", "cpu_count", "load_average_start", "load_average_end"} <= set(environment)
    records = {r["mode"]: r for r in result["records"]}
    for mode in ("cpu_single", "cpu_batch256", "cpu_batch_all"):
        record = records[mode]
        assert record["status"] == "ok" and record["task"] == "blocks-8"
        assert 0 < record["min"] <= record["median"] and record["rounds"] == 2
        assert record["bytes_per_state"] > 0 and record["ints_per_state"] > 0
    assert records["cpu_batch256"]["bytes_per_state"] == records["cpu_batch_all"]["bytes_per_state"]
    # native threads, a held pool, capacity-sized arrays and the relation-major layout run as modes of their own
    for mode, fields in (("cpu_batch_all_t8", dict(threads=8)), ("cpu_batch_all_pool8", dict(threads=8)),
                         ("cpu_batch_all_cap", dict(threads=1)), ("cpu_batch_all_rel", dict(layout="relation")),
                         ("cpu_batch_all_rel_t8", dict(layout="relation", threads=8))):
        assert records[mode]["status"] == "ok", mode
        assert all(records[mode][k] == v for k, v in fields.items()), mode
    assert not any(r["mode"] == "cpu_batch256_t8" for r in result["records"])
    # modes outside the selection do not appear; every gate is listed, and the ones without data are n/a
    assert not any(r["group"] in ("env", "device") for r in result["records"])
    gates = {g["id"]: g for g in result["gates"]}
    assert {t["id"] for t in result["targets"]} == set(gates)
    for gate in ("T-G1", "T-G2", "T-G3", "T-G5", "ENV"):
        assert gates[gate]["status"] == "n/a"
    assert gates["T1"]["status"] in ("pass", "fail") and gates["T7"]["status"] in ("pass", "fail")
    assert "T-G1" in stdout and "n/a" in stdout


def test_unknown_and_absent_tasks_are_reported_not_fatal(tmp_path, bench):
    result, _ = run_script(
        tmp_path, "--tasks", "no-such-task,rovers-p16e,blocks-8", "--modes", "cpu_batch256", "--rounds", "1",
        "--data", str(tmp_path / "none"),
    )
    first, second = result["records"][:2]
    assert first["task"] == "no-such-task" and first["status"] == "skipped" and "unknown task" in first["reason"]
    assert second["task"] == "rovers-p16e" and second["status"] == "skipped"
    if hasattr(bench.mymyr, "Domain"):
        assert str(tmp_path / "none") in second["reason"]
    assert any(r["task"] == "blocks-8" and r["status"] == "ok" for r in result["records"])


def test_walk_rows_are_deterministic_distinct_and_reachable(bench):
    task = text_task("blocks__probBLOCKS-8-0")
    rows = bench.walk_rows(task, 64, seed=3)
    assert rows.dtype == np.uint64 and rows.shape == (64, task.words)
    assert len({r.tobytes() for r in rows}) == 64
    assert np.array_equal(rows, bench.walk_rows(task, 64, seed=3))
    assert not np.array_equal(rows, bench.walk_rows(task, 64, seed=4))
    assert np.array_equal(rows[0], task.encode([task.initial_state])[0])
    tiled = bench.tile(rows, 150)
    assert tiled.shape[0] == 150 and np.array_equal(tiled[64:128], rows)


def record(task, mode, value, **fields):
    defaults = dict(units=4096, group="cpu", bytes_per_state=400.0, ints_per_state=100.0)
    return dict(task=task, mode=mode, status="ok", min=value, median=value, **{**defaults, **fields})


def gate_rows(bench, records, tasks):
    results = bench.gate_results(bench.Index(records), tasks)
    rows, table = bench.gate_table(results)
    return results, {row["id"]: row for row in rows}, table


def test_gates_compare_against_their_thresholds(bench):
    records = [
        record("blocks-p01e", "cpu_single", 2.0),
        record("blocks-p01e", "cpu_batch256", 0.9),
        record("blocks-p01e", "cpu_batch_all", 0.30),
        record("blocks-p01e", "cpu_batch_all_t8", 0.06),
        record("blocks-p01e", "cpu_batch_all_rel", 0.40),
    ]
    results, status, table = gate_rows(bench, records, ["blocks-p01e"])
    assert results["T-C2"]["blocks-p01e"]["status"] == "pass"
    assert results["T1"]["blocks-p01e"]["status"] == "pass"  # 0.30 <= 0.3 + 100 * 1e-3
    assert results["T-C1"]["blocks-p01e"]["status"] == "pass"  # 5x
    assert results["T6"]["blocks-p01e"]["status"] == "fail"  # 0.9 > 14 / 20
    assert results["T7"]["blocks-p01e"]["status"] == "fail"  # 1.33x
    assert status["T-C2"]["status"] == "pass" and status["T6"]["status"] == "fail" and status["T7"]["status"] == "fail"
    assert status["T-G1"]["status"] == "n/a" and status["T-G5"]["status"] == "n/a" and status["T2"]["status"] == "n/a"
    assert "n/a" in table and "pass" in table


def test_a_gate_with_a_missing_task_is_incomplete_and_names_it(bench):
    tasks = ["blocks-p01e", "transport-p30m", "rovers-p16e"]
    records = [
        record("blocks-p01e", "cpu_batch_all", 0.2), record("blocks-p01e", "cpu_batch_all_rel", 0.22),
        record("transport-p30m", "cpu_batch_all", 0.3),
        dict(task="transport-p30m", mode="cpu_batch_all_rel", status="error", reason="RuntimeError: boom", units=4096),
        dict(task="rovers-p16e", mode="*", status="skipped", reason="no PDDL"),
    ]  # fmt: skip
    results, status, table = gate_rows(bench, records, tasks)
    assert results["T7"]["blocks-p01e"]["status"] == "pass"
    assert results["T7"]["transport-p30m"]["status"] == "error" and "boom" in results["T7"]["transport-p30m"]["reason"]
    assert results["T7"]["rovers-p16e"]["status"] == "skipped"
    assert status["T7"]["status"] == "error" and "transport-p30m" in status["T7"]["detail"]
    # T1 lacks only the skipped task: never a pass, and k of n is reported
    assert status["T1"]["status"] == "incomplete" and (status["T1"]["measured"], status["T1"]["applicable"]) == (2, 3)
    assert "rovers-p16e" in status["T1"]["detail"] and "incomplete (2 of 3)" in table
    # a failing task wins over a missing one but still names it
    records.append(record("blocks-p01e", "cpu_single", 9.0))
    _, status, _ = gate_rows(bench, records, tasks)
    assert status["T-C2"]["status"] == "fail" and "incomplete" in status["T-C2"]["detail"]


def test_gates_without_a_baseline_or_a_large_batch_are_not_applicable_with_the_reason(bench):
    records = [
        record("blocks-8", "cpu_batch256", 0.5, units=256), record("blocks-8", "cpu_batch_all", 0.4, units=256),
        record("blocks-8", "cpu_batch_all_t8", 0.1, units=256),
    ]  # fmt: skip
    results, status, _ = gate_rows(bench, records, ["blocks-8"])
    assert status["T6"]["status"] == "n/a" and "no recorded baseline" in status["T6"]["detail"]
    assert status["T-C1"]["status"] == "n/a" and "under 4096" in status["T-C1"]["detail"]
    assert status["ENV"]["status"] == "n/a" and "no recorded baseline" in status["ENV"]["detail"]
    assert status["T-G1"]["status"] == "n/a"
    assert results["T1"]["blocks-8"]["status"] in ("pass", "fail")


def test_a_device_encode_gate_needs_4096_states_and_unsupported_device_modes_are_not_applicable(bench):
    small = [
        record("blocks-p01e", "env_step_rand_n1024", 20.0, group="env", N=1024, min_ns_per_env=20.0),
        record("blocks-p01e", "env_step_encode_n1024", 20.5, group="env", N=1024, min_ns_per_env=20.5),
        record("blocks-p01e", "device_encode_graph", 0.01, group="device", N=1024),
        record("blocks-p01e", "cpu_batch_all", 0.2), record("blocks-p01e", "cpu_batch_all_t8", 0.05),
    ]  # fmt: skip
    results, status, _ = gate_rows(bench, small, ["blocks-p01e"])
    for gate in ("T-G1", "T-G3", "T-G5"):
        assert results[gate]["blocks-p01e"]["status"] == "n/a" and "under 4096" in results[gate]["blocks-p01e"]["reason"]
        assert status[gate]["status"] == "n/a"
    unsupported = [
        dict(task="blocks-p01e", mode=m, group="device", status="skipped", unavailable=True, N=16384,
             reason="not supported on the device (TypeError: rows)")
        for m in ("device_encode_graph", "device_encode_rel")
    ]  # fmt: skip
    _, status, _ = gate_rows(bench, unsupported, ["blocks-p01e"])
    assert status["T-G1"]["status"] == "n/a" and "not supported on the device" in status["T-G1"]["detail"]
    assert status["T-G2"]["status"] == "n/a"
    failed_setup = [dict(unsupported[0], unavailable=False, reason="device environment unavailable: x")]
    _, status, _ = gate_rows(bench, failed_setup, ["blocks-p01e"])
    assert status["T-G1"]["status"] == "incomplete"


def test_the_environment_gate_fails_a_slower_step_and_warns_about_a_much_faster_one(bench):
    base = bench.ENV_STEP_US["blocks-p01e"][16384]
    for factor, expected, warned in ((1.0, "pass", False), (1.19, "pass", False), (0.5, "pass", True),
                                     (1.3, "fail", False)):  # fmt: skip
        records = [record("blocks-p01e", "env_step_n16384", base * factor, group="env", N=16384)]
        results, status, table = gate_rows(bench, records, ["blocks-p01e"])
        assert results["ENV"]["blocks-p01e"]["status"] == expected, factor
        assert ("warning" in status["ENV"]) == warned and ("faster than recorded by >20%" in table) == warned
    errored = [dict(task="blocks-p01e", mode="env_step_n1024", status="error", reason="boom", units=10)]
    _, status, _ = gate_rows(bench, errored, ["blocks-p01e"])
    assert status["ENV"]["status"] == "error"


def test_t_c1_is_soft_and_the_table_says_so(bench):
    assert {t["id"]: t["soft"] for t in bench.TARGETS}["T-C1"] is True
    records = [record("blocks-p01e", "cpu_batch_all", 1.0), record("blocks-p01e", "cpu_batch_all_t8", 0.9)]
    _, status, _ = gate_rows(bench, records, ["blocks-p01e"])
    assert status["T-C1"]["status"] == "fail" and "rerun on a quiet host" in status["T-C1"]["detail"]


class FakeGpu:
    """What the device helpers use of the GPU: ok, sync and empty_cache, plus a torch.cuda that records the calls."""

    def __init__(self, calls, elapsed_ms=50.0):
        self.calls, self.ok, self.device = calls, True, "cuda:fake"
        gpu = self

        class Graph:
            def replay(self):
                gpu.calls.append("replay")

        class Event:
            def __init__(self, enable_timing=False):
                assert enable_timing

            def record(self):
                gpu.calls.append("event")

            def synchronize(self):
                gpu.calls.append("event_synchronize")

            def elapsed_time(self, other):
                return elapsed_ms

        @contextlib.contextmanager
        def graph(g, stream=None):
            gpu.calls.append("capture_begin" if stream is not None else "capture_on_default_stream")
            yield
            gpu.calls.append("capture_end")

        self.torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(CUDAGraph=Graph, Event=Event, graph=graph, Stream=lambda: object())
        )

    def sync(self):
        self.calls.append("sync")

    def empty_cache(self):
        self.calls.append("empty_cache")


class FakeStates:
    def __init__(self, rows):
        self.rows = rows

    def cpu(self):
        return types.SimpleNamespace(numpy=lambda: self.rows)


class FakeDeviceEncoder:
    """Delegates the host encodes to a real encoder and records the device encodes and their capacities."""

    def __init__(self, real, calls):
        self.real, self.calls, self.capacities = real, calls, []

    def encode(self, states, capacities=None):
        if isinstance(states, FakeStates):
            self.calls.append("encode")
            self.capacities.append(capacities)
            return object()
        return self.real.encode(states, capacities=capacities) if capacities else self.real.encode(states)


@pytest.fixture
def device_setup(bench):
    task = text_task("blocks__probBLOCKS-8-0")
    rows = bench.walk_rows(task, 64, seed=1)
    real, _ = bench.make_encoder(task)
    return real, rows


def test_a_captured_device_encode_is_built_outside_the_capture_and_timed_with_events(bench, device_setup):
    real, rows = device_setup
    calls = []
    gpu, encoder = FakeGpu(calls), FakeDeviceEncoder(real, calls)
    with bench.encode_session(gpu, encoder, FakeStates(rows), eager=False) as run:
        # warmed up (an encode and a synchronize) before the capture, one encode inside it
        assert calls == ["encode", "sync", "capture_begin", "encode", "capture_end"]
        calls.clear()
        assert run() == pytest.approx(50.0 * 1e3)
        assert calls == ["event"] + ["replay"] * bench.REPLAYS + ["event", "event_synchronize"]
    exact = bench.capacities_of(real, rows)
    assert encoder.capacities[0] == {k: -(-v * 5 // 4) for k, v in exact.items()} and len(encoder.capacities) == 2
    assert encoder.capacities[0] == encoder.capacities[1]
    assert calls[-1] == "sync"  # the run is closed after the graph is dropped


def test_the_eager_device_encode_is_one_call_and_a_synchronize(bench, device_setup):
    real, rows = device_setup
    calls = []
    with bench.encode_session(FakeGpu(calls), FakeDeviceEncoder(real, calls), FakeStates(rows), eager=True) as run:
        calls.clear()
        assert run() is None
        assert calls == ["encode", "sync"]


def test_a_mode_that_reports_its_time_is_divided_by_the_replayed_states(bench):
    mode = bench.Mode("device_encode_graph", "device", units=100 * bench.REPLAYS, N=100, reports_time=True)
    mode.run = lambda: 2000.0
    bench.time_rounds([mode], 3)
    assert mode.times == [2000.0 / 1000] * 3 and bench.summarize(mode)["min"] == pytest.approx(2.0)


def test_modes_that_open_their_own_run_are_timed_one_after_the_other_and_closed(bench):
    calls = []

    def opened(name):
        @contextlib.contextmanager
        def session():
            calls.append(f"open {name}")
            yield lambda: calls.append(f"run {name}")
            calls.append(f"close {name}")

        return session

    modes = [bench.Mode(n, "device", open=opened(n)) for n in ("a", "b")]
    bench.time_rounds(modes, 2)
    assert calls == ["open a", *["run a"] * 4, "close a", "open b", *["run b"] * 4, "close b"]  # 2 warm-up + 2 rounds
    assert all(m.run is None and len(m.times) == 2 for m in modes)


def test_a_probe_skips_rows_the_encoder_cannot_take_and_fails_when_a_device_path_exists(bench, device_setup):
    real, rows = device_setup

    class HostOnly(FakeDeviceEncoder):
        def encode(self, states, capacities=None):
            if isinstance(states, FakeStates):
                raise NotImplementedError("rows on the device")
            return super().encode(states, capacities)

    gpu = FakeGpu([])
    gpu.torch.ops = types.SimpleNamespace(mymyr=types.SimpleNamespace())
    reason = bench.probe_device_encode(gpu, HostOnly(real, []), FakeStates(rows))
    assert reason.startswith("not supported on the device") and "rows on the device" in reason
    assert bench.probe_device_encode(gpu, FakeDeviceEncoder(real, []), FakeStates(rows)) is None
    gpu.torch.ops.mymyr.encode_flat = object()
    with pytest.raises(bench.DevicePathError, match="cannot drive"):
        bench.probe_device_encode(gpu, HostOnly(real, []), FakeStates(rows))


class Tracked:
    destroyed = []

    def __init__(self, name):
        self.name = name

    def __del__(self):
        Tracked.destroyed.append(self.name)


def test_release_drops_the_mode_references_then_the_graphs_then_the_environments(bench):
    Tracked.destroyed = []
    calls = []
    resources = bench.Resources(FakeGpu(calls))
    graph, env = Tracked("graph"), Tracked("env")
    # the mode closure references the graph and the environment, as a replay mode does
    mode = bench.Mode("env_step_n1", "env", lambda g=graph, e=env: (g, e))
    resources.modes.append(mode)
    resources.graphs.append(graph)
    resources.envs.append(env)
    del graph, env
    gc.collect()
    assert Tracked.destroyed == []
    resources.release()
    assert Tracked.destroyed == ["graph", "env"] and mode.run is None
    assert calls == ["sync", "sync", "empty_cache"]


def test_modes_outside_the_selection_build_nothing_on_the_device(bench):
    def refuse(*args, **kwargs):
        raise AssertionError("an environment was built")

    gpu = FakeGpu([])
    gpu.rt = types.SimpleNamespace(BatchedEnv=refuse)
    args = types.SimpleNamespace(data=None, modes={"cpu_batch256"}, rows=32, seed=0, min_batch=64, rounds=1,
                                 env_sizes=(256, 1024))
    records = bench.run_task("blocks-8", args, gpu, pool=None)
    assert [r["mode"] for r in records] == ["cpu_batch256"] and records[0]["status"] == "ok"
    assert gpu.calls[-1] == "empty_cache"


def test_the_bytes_of_a_record_come_from_the_batch_the_mode_timed(bench, device_setup):
    real, rows = device_setup
    cache = {}
    per_state = []
    for data in (rows[:2], rows):
        batch = real.encode(data)
        nbytes, nints = bench.batch_stats(batch)
        per_state.append((nbytes / batch.num_graphs, nints / batch.num_graphs))
        assert bench.stats_of(bench.Mode("m", "cpu", stats=(real, data)), cache) == per_state[-1]
    assert per_state[0] != per_state[1]


def test_zero_rounds_are_rejected():
    done = subprocess.run([sys.executable, str(SCRIPT), "--quick", "--rounds", "0"], capture_output=True, text=True)
    assert done.returncode != 0 and "--rounds must be at least 1" in done.stderr


def test_device_modes_are_measured_or_skipped_as_not_supported_never_errors(tmp_path):
    result, _ = run_script(tmp_path, "--modes", "device,env_step_encode_n256", "--rounds", "1", "--env-sizes", "256")
    records = [r for r in result["records"] if r["group"] in ("device", "env")]
    assert records and not any(r["status"] == "error" for r in records)
    for r in records:
        if r["status"] == "skipped":
            assert r["unavailable"] and (r["reason"].startswith("not supported on the device") or "no " in r["reason"]), r
    status = {g["id"]: g["status"] for g in result["gates"]}
    assert all(status[g] in ("n/a", "pass", "fail") for g in ("T-G1", "T-G2", "T-G3", "T-G5"))


def test_gpu_modes_run_the_device_environment_when_cuda_is_present(tmp_path):
    torch = pytest.importorskip("torch")
    cuda = pytest.importorskip("mymyr.cuda", exc_type=ImportError)
    if not torch.cuda.is_available() or not cuda.available():
        pytest.skip("no CUDA device")
    result, _ = run_script(tmp_path, "--modes", "env", "--rounds", "2", "--env-sizes", "256")
    records = {r["mode"]: r for r in result["records"]}
    assert records["env_step_n256"]["status"] == "ok" and records["env_step_n256"]["min_ns_per_env"] > 0
    assert records["env_step_rand_n256"]["status"] == "ok"
    assert result["environment"]["gpu"]["name"]
