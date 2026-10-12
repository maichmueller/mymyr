"""bench/python/bench_encoding_gpu.py: the command line runs and writes the documented JSON without CUDA, the walk rows
are deterministic and distinct, and the gate table reports a gate whose modes were skipped as n/a, never as a pass."""

import importlib.util
import json
import subprocess
import sys

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
    # modes outside the selection do not appear; every gate is listed, and the ones without data are n/a
    assert not any(r["group"] in ("env", "device") for r in result["records"])
    gates = {g["id"]: g for g in result["gates"]}
    assert {t["id"] for t in result["targets"]} == set(gates)
    for gate in ("T-G1", "T-G2", "T-G3", "T-G5", "ENV"):
        assert gates[gate]["status"] == "n/a"
    assert gates["T1"]["status"] in ("pass", "fail")
    assert "T-G1" in stdout and "n/a" in stdout


def test_modes_without_a_device_encoder_are_skipped_with_a_reason(tmp_path, bench):
    if bench.device_encoder_class() is not None:
        pytest.skip("the device encoder exists")
    result, _ = run_script(tmp_path, "--modes", "device,env_step_encode_n256", "--rounds", "1", "--env-sizes", "256")
    skipped = {r["mode"]: r for r in result["records"] if r["status"] == "skipped"}
    assert skipped["device_encode_graph"]["reason"] == "no device encoder"
    assert skipped["device_encode_rel"]["reason"] == "no device encoder"
    assert not any(r["status"] == "ok" for r in result["records"])


def test_unknown_and_absent_tasks_are_reported_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("MYMYR_IPC", str(tmp_path / "none"))
    monkeypatch.setenv("MYMYR_FORK_DATA_DIR", str(tmp_path / "none"))
    result, _ = run_script(tmp_path, "--tasks", "no-such-task,blocks-8", "--modes", "cpu_batch256", "--rounds", "1")
    first = result["records"][0]
    assert first["task"] == "no-such-task" and first["status"] == "skipped" and "unknown task" in first["reason"]
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
    return dict(task=task, mode=mode, status="ok", min=value, median=value, units=4096, group="cpu",
                bytes_per_state=400.0, ints_per_state=100.0, **fields)  # fmt: skip


def test_gates_compare_against_their_thresholds_and_skipped_modes_are_not_a_pass(bench):
    records = [
        record("blocks-p01e", "cpu_single", 2.0),
        record("blocks-p01e", "cpu_batch256", 0.9),
        record("blocks-p01e", "cpu_batch_all", 0.30),
        record("blocks-p01e", "cpu_batch_all_t8", 0.06),
        record("blocks-p01e", "cpu_batch_all_rel", 0.40),
    ]
    results = bench.gate_results(bench.Index(records))
    assert results["T-C2"]["blocks-p01e"]["status"] == "pass"
    assert results["T1"]["blocks-p01e"]["status"] == "pass"  # 0.30 <= 0.3 + 100 * 1e-3
    assert results["T-C1"]["blocks-p01e"]["status"] == "pass"  # 5x
    assert results["T6"]["blocks-p01e"]["status"] == "fail"  # 0.9 > 14 / 20
    assert results["T7"]["blocks-p01e"]["status"] == "fail"  # 1.33x
    assert results["T-G1"] == {} and results["ENV"] == {}
    rows, table = bench.gate_table(results)
    status = {row["id"]: row["status"] for row in rows}
    assert status["T-C2"] == "pass" and status["T6"] == "fail" and status["T7"] == "fail"
    assert status["T-G1"] == "n/a" and status["T-G5"] == "n/a" and status["T2"] == "n/a"
    assert "n/a" in table and "pass" in table


def test_env_gate_fails_a_step_more_than_20_percent_slower_than_recorded(bench):
    base = bench.ENV_STEP_US["blocks-p01e"][16384]
    for factor, expected in ((1.0, "pass"), (1.19, "pass"), (0.5, "pass"), (1.3, "fail")):
        results = bench.gate_results(bench.Index([record("blocks-p01e", "env_step_n16384", base * factor)]))
        assert results["ENV"]["blocks-p01e"]["status"] == expected, factor


def test_gpu_modes_run_the_device_environment_when_cuda_is_present(tmp_path, bench):
    torch = pytest.importorskip("torch")
    cuda = pytest.importorskip("mymyr.cuda", exc_type=ImportError)
    if not torch.cuda.is_available() or not cuda.available():
        pytest.skip("no CUDA device")
    result, _ = run_script(tmp_path, "--modes", "env", "--rounds", "2", "--env-sizes", "256")
    records = {r["mode"]: r for r in result["records"]}
    assert records["env_step_n256"]["status"] == "ok" and records["env_step_n256"]["min_ns_per_env"] > 0
    assert records["env_step_rand_n256"]["status"] == "ok"
    assert result["environment"]["gpu"]["name"]
