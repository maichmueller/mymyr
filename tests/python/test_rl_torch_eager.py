"""Eager custom-op entry points keep compiler modules unloaded in a fresh interpreter."""

import inspect

import pytest

from python_children import run_code


def _exercise_ops(compile_ops=False):
    import sys
    import sysconfig
    from pathlib import Path

    import numpy as np
    import torch

    import mymyr
    from mymyr import encoding, rl
    from mymyr.rl import torch as rt
    from mymyr.rl.torch import _masks, _ops

    def check_imports():
        if compile_ops:
            return
        assert not {"torch._dynamo", "torch._inductor", "triton"}.intersection(sys.modules)
        if sysconfig.get_config_var("Py_GIL_DISABLED"):
            assert not sys._is_gil_enabled()

    check_imports()
    registered = {name.split("::")[1] for name in torch._C._dispatch_get_all_op_names()
                  if name.startswith("mymyr::")}
    task = mymyr.Task.from_text(str(Path("tests/data/tasks/gripper__prob05.txt")))
    table = rl.TaskTable([task])
    encoder = encoding.FlatRelationEncoder(table, target_sources=("action", "goal"))
    templates = rl.lifted_goal_templates(task)
    handles = [rt.register(obj) for obj in (table, encoder, templates)]
    table_handle, encoder_handle, template_handle = handles
    devices = ["cpu"]
    if torch.cuda.is_available() and hasattr(mymyr._core, "_cuda"):
        torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(0).total_memory, 0)
        devices.append("cuda:0")
    cuda_only = {"bounded_actions", "encode_flat", "literal_slots", "literal_masks"}
    try:
        for device in devices:
            if device != "cpu":
                template_handle = rt.register(templates.to(device))
                handles.append(template_handle)
            env = rt.BatchedEnv(table, 2, device=device, track_visited=32, observation=encoder, obs_actions=True,
                                obs_capacities=dict(entities=256, args=1024, targets=256))
            states = env.states
            schema = torch.zeros(2, dtype=torch.int32, device=device)
            binding = torch.zeros((2, table.label_width), dtype=torch.int32, device=device)
            refresh = (env.handle, states, None, env.counts, env.views, None, None, env.count, env.at_goal)
            step = (env.handle, states, None, env.steps, env.draws, env.counts, env.views, env.count,
                    env.at_goal, None, None, None, None, 0, True)
            hashes = torch.ones(2, dtype=torch.int64, device=device)
            parents = torch.arange(2, dtype=torch.int32, device=device)
            predicates = torch.zeros(2, dtype=torch.int32, device=device)
            objects = torch.zeros((2, 2), dtype=torch.int32, device=device)
            positive = torch.ones(2, dtype=torch.bool, device=device)
            args = {
                "reset": (env.handle, states, None, env.steps, env.counts, env.views, None, None,
                          env.count, env.at_goal, None, False, env.visited, env.visited_count, env.visited_status),
                "refresh": refresh,
                "refresh_sync": refresh,
                "step": step,
                "step_sync": step,
                "random_actions": (env.handle, env.draws, env.count, 0, None),
                "clear_visited": (env.handle, states, None, env.visited, env.visited_count,
                                  env.visited_status, None),
                "visited_mask": (env.visited, env.visited_count, hashes, parents),
                "target_visited": (torch.zeros(2, dtype=torch.int32, device=device),
                                   torch.zeros(2, dtype=torch.uint8, device=device), parents,
                                   torch.tensor([0, 1, 2], dtype=torch.int32, device=device), None,
                                   torch.tensor([0, 1, 2], dtype=torch.int32, device=device), None,
                                   hashes, None, env.visited, env.visited_count, None),
                "expand_flat": (table_handle, states, None, 64),
                "expand": (table_handle, states, None, 32),
                "applicable_schemas": (table_handle, states),
                "applicable_objects": (table_handle, states, schema, None, 0),
                "is_applicable": (table_handle, states, schema, binding),
                "match_first": (template_handle, states),
                "her_relabel_lifted": (template_handle, states.unsqueeze(0),
                                       torch.zeros((1, 2), dtype=torch.bool, device=device),
                                       None, states.clone().unsqueeze(0), 2),
                "bounded_actions": (encoder_handle, states, None, 256),
                "encode_flat": (encoder_handle, states, *([None] * 16), 2, 256, 1024, 256, 0),
                "literal_slots": (encoder_handle, predicates, objects),
                "literal_masks": (encoder_handle, predicates, objects, positive, parents, 2),
            }
            assert set(args) == registered, (registered - args.keys(), args.keys() - registered)
            for name, inputs in args.items():
                if device == "cpu" and name in cuda_only:
                    continue
                op = getattr(rt, name, None) or getattr(_ops, name, None) or getattr(_masks, name)
                assert isinstance(op, torch.library.CustomOpDef), name
                if compile_ops:
                    from torch._subclasses.fake_tensor import FakeTensorMode

                    torch.compiler.reset()
                    with FakeTensorMode() as mode:
                        fake_inputs = tuple(mode.from_tensor(x) if isinstance(x, torch.Tensor) else x for x in inputs)
                        fake = op(*fake_inputs)
                    op(*fake_inputs)
                    if name == "applicable_schemas":
                        op(table=inputs[0], states=fake_inputs[1])
                    meta_inputs = tuple(torch.empty_like(x, device="meta") if isinstance(x, torch.Tensor) else x
                                        for x in inputs)
                    meta = op(*meta_inputs)
                    result = torch.compile(op, fullgraph=True, backend="aot_eager")(*inputs)
                    if name in {"applicable_schemas", "her_relabel_lifted"}:
                        checked = torch.library.opcheck(op, inputs)
                        assert set(checked.values()) == {"SUCCESS"}, checked
                    fake = () if fake is None else (fake,) if isinstance(fake, torch.Tensor) else fake
                    meta = () if meta is None else (meta,) if isinstance(meta, torch.Tensor) else meta
                    actual = () if result is None else (result,) if isinstance(result, torch.Tensor) else result
                    assert [(x.shape, x.dtype, x.device) for x in fake] == [
                        (x.shape, x.dtype, x.device) for x in actual
                    ], name
                    assert [(x.shape, x.dtype) for x in meta] == [(x.shape, x.dtype) for x in actual], name
                else:
                    result = op(*inputs)
                assert result is None or isinstance(result, (torch.Tensor, tuple, list)), name
                check_imports()
            # Exercise the environment and framework-level entry points as well as the registered ops.
            env.reset()
            env.refresh()
            env.random_actions()
            env.step()
            env.clear_visited()
            rl.visited_mask(env.visited, env.visited_count, hashes, parents)
            seen = torch.zeros_like(states)
            rl.novelty_update(seen, states)
            check_imports()
            if device != "cpu":
                torch.cuda.synchronize()
            env.close()
    finally:
        for handle in handles:
            rt.unregister(handle)
    check_imports()


def _run_ops(*, compile_ops=False):
    pytest.importorskip("torch", exc_type=ImportError)
    source = inspect.getsource(_exercise_ops) + f"\n_exercise_ops(compile_ops={compile_ops!r})\n"
    result = run_code(source, timeout=None, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_eager_ops_leave_compiler_modules_unloaded():
    _run_ops()


@pytest.mark.needs_gil
def test_op_entry_points_support_fake_tensors_and_fullgraph_compile():
    _run_ops(compile_ops=True)
