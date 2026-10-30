"""Custom-op entry points preserve dispatcher mutation, tracing and registration semantics."""

import sys
from typing import Optional

import pytest

from conftest import text_task


@pytest.fixture
def state_pair():
    import numpy as np

    torch = pytest.importorskip("torch")
    from mymyr.rl import torch as rt

    task = text_task("gripper__prob05")
    handle = rt.register(task)
    initial = torch.tensor(np.asarray(task.encode([task.initial_state])).view(np.int64))
    for state in task.successor_states(task.initial_state):
        other = torch.tensor(np.asarray(task.encode([state])).view(np.int64))
        if not torch.equal(rt.applicable_schemas(handle, initial)[0], rt.applicable_schemas(handle, other)[0]):
            break
    else:
        pytest.fail("expected a successor with different applicable schemas")
    try:
        yield rt.applicable_schemas, handle, initial, other
    finally:
        rt.unregister(handle)


@pytest.mark.needs_gil
@pytest.mark.parametrize("tensor_arg", ["states", "prev_states", "done"])
@pytest.mark.parametrize("tensor_kind", ["fake", "meta"])
def test_her_relabel_lifted_non_data_arguments_match_dispatcher(tensor_arg, tensor_kind, monkeypatch):
    import numpy as np

    torch = pytest.importorskip("torch")
    from torch._subclasses.fake_tensor import FakeTensorMode

    from mymyr import rl
    from mymyr.rl import torch as rt

    task = text_task("gripper__prob05")
    handle = rt.register(rl.lifted_goal_templates(task))
    states = torch.tensor(np.asarray(task.encode([task.initial_state])).view(np.int64)).unsqueeze(0)
    args = dict(templates=handle, states=states, done=torch.zeros((1, 1), dtype=torch.bool),
                task_ids=None, prev_states=states.clone(), max_segments=2)
    if tensor_kind == "fake":
        args[tensor_arg] = FakeTensorMode(allow_non_fake_inputs=True).from_tensor(args[tensor_arg])
    else:
        args[tensor_arg] = torch.empty_like(args[tensor_arg], device="meta")

    def reject_direct(*args, **kwargs):
        pytest.fail("a non-data argument reached the direct implementation")

    monkeypatch.setattr(rt.her_relabel_lifted, "_init_fn", reject_direct)
    try:
        expected = rt.her_relabel_lifted._opoverload(**args)
        actual = rt.her_relabel_lifted(**args)
        assert len(actual) == len(expected) == 14
        assert [(x.shape, x.dtype, x.device, type(x)) for x in actual] == [
            (x.shape, x.dtype, x.device, type(x)) for x in expected
        ]
    finally:
        rt.unregister(handle)


@pytest.mark.needs_gil
@pytest.mark.parametrize("keywords", [False, True])
def test_refresh_invalidates_saved_goal_mask(keywords):
    torch = pytest.importorskip("torch")
    from mymyr.rl import torch as rt
    from mymyr.rl.torch import _ops

    env = rt.BatchedEnv(text_task("gripper__prob05"), 1)
    try:
        for call in (_ops.refresh, _ops.refresh._opoverload):
            env.at_goal.fill_(True)
            loss = torch.ones(1, requires_grad=True).masked_fill(env.at_goal, 0).sum()
            mutated = [x for x in (env.counts, env.views, env.count, env.at_goal) if x is not None]
            versions = [x._version for x in mutated]
            args = dict(env=env.handle, states=env.states, task_ids=None, counts=env.counts, views=env.views,
                        goal_pos=None, goal_neg=None, count=env.count, at_goal=env.at_goal)
            call(**args) if keywords else call(*args.values())
            assert not env.at_goal.item()
            assert [x._version - version for x, version in zip(mutated, versions)] == [1] * len(mutated)
            with pytest.raises(RuntimeError, match="modified by an inplace operation"):
                loss.backward()
    finally:
        env.close()


@pytest.mark.needs_gil
def test_preallocated_step_invalidates_saved_output():
    torch = pytest.importorskip("torch")
    from mymyr.rl import torch as rt

    env = rt.BatchedEnv(text_task("gripper__prob05"), 1)
    try:
        out = env.step()
        out.terminated.fill_(True)
        loss = torch.ones(1, requires_grad=True).masked_fill(out.terminated, 0).sum()
        outputs = tuple(out)[:8]
        versions = [tensor._version for tensor in outputs]
        env._step_into(out, None, None, True)
        assert [tensor._version - version for tensor, version in zip(outputs, versions)] == [1] * len(outputs)
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            loss.backward()
    finally:
        env.close()


@pytest.mark.needs_gil
@pytest.mark.parametrize("entry", ["eager", "dispatcher"])
@pytest.mark.parametrize("keywords", [False, True])
def test_failed_mutation_invalidates_saved_tensor(entry, keywords):
    import ctypes

    torch = pytest.importorskip("torch")
    from torch import Tensor

    from mymyr.rl.torch._ops import _op

    versions = []

    @_op("mymyr_test::failing_mutation", mutates_args=("x",))
    def mutate(x: Tensor) -> None:
        versions.append(x._version)
        ctypes.c_bool.from_address(x.data_ptr()).value = False
        raise ValueError("failure after pointer write")

    mask = torch.ones(1, dtype=torch.bool, device="cpu")
    loss = torch.ones(1, requires_grad=True).masked_fill(mask, 0).sum()
    version = mask._version
    call = mutate if entry == "eager" else mutate._opoverload
    with pytest.raises(ValueError, match="failure after pointer write"):
        call(x=mask) if keywords else call(mask)
    assert not mask.item()
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()
    assert versions == [version + 1]
    assert mask._version == version + 1


@pytest.mark.needs_gil
@pytest.mark.parametrize("optional", [False, True])
def test_declared_optional_and_keyword_mutations_increment_versions(optional):
    import ctypes

    torch = pytest.importorskip("torch")
    from torch import Tensor

    from mymyr.rl.torch._ops import _op

    @_op("mymyr_test::optional_mutations", mutates_args=("x", "y", "z"))
    def mutate(x: Tensor, y: Optional[Tensor] = None, z: Optional[Tensor] = None) -> None:
        for tensor in (x, y, z):
            if tensor is not None:
                ctypes.c_bool.from_address(tensor.data_ptr()).value = False

    for call in (mutate, mutate._opoverload):
        x, y, z = [torch.ones(1, dtype=torch.bool, device="cpu") for _ in range(3)]
        loss = torch.ones(1, requires_grad=True).masked_fill(x, 0).sum()
        call(x, y=y if optional else None, z=z if optional else None)
        assert [t._version for t in (x, y, z)] == [1, int(optional), int(optional)]
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            loss.backward()


@pytest.mark.needs_gil
def test_dispatch_key_exclusions_preserve_mutation_versions():
    import ctypes

    torch = pytest.importorskip("torch")
    from torch import Tensor

    from mymyr.rl.torch._ops import _op

    @_op("mymyr_test::excluded_mutation", mutates_args=("x",))
    def mutate(x: Tensor) -> None:
        ctypes.c_bool.from_address(x.data_ptr()).value = False

    for call in (mutate, mutate._opoverload):
        tensor = torch.ones(1, dtype=torch.bool, device="cpu")
        with torch._C._AutoDispatchBelowADInplaceOrView():
            call(tensor)
        assert not tensor.item()
        assert tensor._version == 0


@pytest.mark.needs_gil
@pytest.mark.parametrize("tracing_mode", ["real", "fake", "symbolic"])
def test_make_fx_retains_custom_op_and_uses_new_states(state_pair, tracing_mode):
    import torch
    from torch.fx.experimental.proxy_tensor import make_fx

    op, handle, initial, other = state_pair
    for call in (op, op._opoverload):
        graph = make_fx(lambda states: call(handle, states)[0], tracing_mode=tracing_mode)(initial)
        assert op._opoverload in [node.target for node in graph.graph.nodes]
        for states in (initial, other):
            assert torch.equal(graph(states), op._opoverload(handle, states)[0])


@pytest.mark.needs_gil
def test_torchscript_trace_retains_custom_op_and_uses_new_states(state_pair):
    import torch

    op, handle, initial, other = state_pair
    for call in (op, op._opoverload):
        if sys.version_info >= (3, 14):
            with pytest.warns(FutureWarning, match="torch.jit.trace.*not supported in Python 3.14"):
                traced = torch.jit.trace(lambda states: call(handle, states)[0], initial, check_trace=False)
        else:
            traced = torch.jit.trace(lambda states: call(handle, states)[0], initial, check_trace=False)
        assert "mymyr::applicable_schemas" in str(traced.graph)
        for states in (initial, other):
            assert torch.equal(traced(states), op._opoverload(handle, states)[0])


@pytest.mark.needs_gil
@pytest.mark.parametrize("transform", ["functionalize", "vmap", "grad"])
def test_function_transforms_match_dispatcher(state_pair, transform, monkeypatch):
    import torch

    op, handle, initial, other = state_pair

    def reject_direct(*args, **kwargs):
        pytest.fail("a transformed call reached the direct implementation")

    monkeypatch.setattr(op, "_init_fn", reject_direct)
    for states in (initial, other):
        results = []
        for call in (op, op._opoverload):
            if transform == "functionalize":
                result = torch.func.functionalize(lambda s: call(handle, s)[0])(states)
            elif transform == "vmap":
                enabled = torch._C._debug_only_are_vmap_fallback_warnings_enabled()
                torch._C._debug_only_display_vmap_fallback_warnings(True)
                try:
                    with pytest.warns(UserWarning, match="batching rule.*mymyr::applicable_schemas"):
                        result = torch.vmap(lambda s: call(handle, s)[0])(states.unsqueeze(0))
                finally:
                    torch._C._debug_only_display_vmap_fallback_warnings(enabled)
            else:
                result = torch.func.grad(lambda scale: (call(handle, states)[0] * scale).sum())(torch.ones(()))
            results.append(result)
        assert torch.equal(*results)


@pytest.mark.needs_gil
def test_functional_tensor_outside_transform_matches_dispatcher(state_pair):
    import torch

    op, handle, initial, other = state_pair
    for states in (initial, other):
        wrapped = torch._to_functional_tensor(states)
        assert type(wrapped) is torch.Tensor
        actual = op(handle, wrapped)[0]
        expected = op._opoverload(handle, wrapped)[0]
        assert torch.equal(torch._from_functional_tensor(actual), torch._from_functional_tensor(expected))


@pytest.mark.needs_gil
def test_dispatch_mode_observes_custom_op_with_keyword_tensors(state_pair):
    from torch.utils._python_dispatch import TorchDispatchMode

    op, handle, initial, _ = state_pair

    class Observe(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.seen = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.seen.append(func)
            return func(*args, **(kwargs or {}))

    for call in (op, op._opoverload):
        with Observe() as mode:
            call(table=handle, states=initial)
        assert mode.seen == [op._opoverload]


@pytest.mark.needs_gil
def test_tensor_subclass_intercepts_custom_op(state_pair):
    import torch

    op, handle, initial, _ = state_pair

    class Intercept(torch.Tensor):
        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            assert func is op._opoverload
            raise RuntimeError("intercepted custom op")

    wrapped = torch.Tensor._make_subclass(Intercept, initial)
    for call in (op, op._opoverload):
        with pytest.raises(RuntimeError, match="intercepted custom op"):
            call(handle, wrapped)


@pytest.mark.needs_gil
def test_torch_function_mode_intercepts_custom_op(state_pair):
    from torch.overrides import TorchFunctionMode

    op, handle, initial, _ = state_pair

    class Intercept(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            assert func is op._opoverload
            raise RuntimeError("intercepted custom op")

    for call in (op, op._opoverload):
        with Intercept(), pytest.raises(RuntimeError, match="intercepted custom op"):
            call(handle, initial)


@pytest.mark.needs_gil
@pytest.mark.parametrize("registration", ["kernel", "autograd", "vmap"])
def test_later_registrations_match_dispatcher_and_compilation(registration):
    torch = pytest.importorskip("torch")
    from torch import Tensor

    from mymyr.rl.torch._ops import _op

    @_op(f"mymyr_test::registered_{registration}", mutates_args=())
    def op(x: Tensor) -> Tensor:
        return x * 2

    @op.register_fake
    def fake(x):
        return torch.empty_like(x)

    assert op(torch.ones(2)).tolist() == [2, 2]
    if registration == "kernel":
        @torch.library.register_kernel(op, "cpu")
        def kernel(x):
            return x + 7
    elif registration == "autograd":
        torch.library.register_autograd(op, lambda ctx, grad: grad * 9)
    else:
        @torch.library.register_vmap(op)
        def vmap(info, in_dims, x):
            return x + 6, in_dims[0]

    compiled = torch.compile(op, fullgraph=True, backend="aot_eager")
    for call in (op, op._opoverload, compiled):
        x = torch.ones(2, requires_grad=registration == "autograd")
        if registration == "vmap":
            # Trace the transform so the compiled graph contains the registered batching behavior.
            mapped = torch.compile(torch.vmap(op), fullgraph=True, backend="aot_eager") if call is compiled else (
                torch.vmap(call))
            assert mapped(x).tolist() == [7, 7]
        else:
            result = call(x)
            assert result.tolist() == ([8, 8] if registration == "kernel" else [2, 2])
            if registration == "autograd":
                result.sum().backward()
                assert x.grad.tolist() == [9, 9]


@pytest.mark.needs_gil
def test_unregistered_backward_matches_dispatcher():
    torch = pytest.importorskip("torch")
    from torch import Tensor

    from mymyr.rl.torch._ops import _op

    @_op("mymyr_test::missing_backward", mutates_args=())
    def op(x: Tensor) -> Tensor:
        return x * 2

    for call in (op, op._opoverload):
        result = call(torch.ones(1, requires_grad=True))
        with pytest.raises(RuntimeError, match="no autograd formula was registered"):
            result.sum().backward()
