#!/usr/bin/env python3
"""Check an installed CUDA build of mymyr: that it is the given wheel, its version, the GPU code it carries, its JAX
targets and, with torch imported first, that torch and mymyr exchange arrays.

usage: check_cuda_install.py --version VERSION --archs "86 89 90" [--wheel PATH] [--torch VERSION --task PATH]
                             [--gpu]

The installed mymyr must have the version VERSION (in its metadata and mymyr.__version__), a CUDA backend whose
build_info() lists SASS (sm_<n>) for every architecture in --archs, and the XLA FFI targets of mymyr.rl.jax and
mymyr.encoding.jax for the CPU and for CUDA.

--wheel  mymyr must be imported from this interpreter's virtual environment (not from a base interpreter's site
         packages), and its package files must be the wheel's: their sha256 is compared with the wheel's RECORD.
--torch  torch is imported before mymyr, in the same process; its version must start with VERSION (e.g. 2.14) and it
         must be a CUDA 13.0 build. The task file (Task.from_text) is expanded from a torch tensor and the result
         compared with the expansion of a NumPy array: on the CPU, and with --gpu also from a torch CUDA tensor
         (DLPack into mymyr) and from a mymyr device array read by torch (DLPack into torch).
--gpu    a CUDA device must be visible, and the build must have SASS for the architecture of device 0.

Prints the build info, where mymyr was imported from and, with --gpu, the device.
"""

import argparse
import base64
import csv
import hashlib
import importlib
import importlib.metadata
import io
import sys
import zipfile
from pathlib import Path


def wheel_errors(wheel: Path, version: str) -> list[str]:
    """mymyr comes from this virtual environment and its files are the wheel's."""
    import mymyr

    errors = []
    name_version = wheel.name.split("-")[1] if wheel.name.count("-") >= 4 else ""
    if name_version != version:
        errors.append(f"the wheel {wheel.name} is not version {version}")
    prefix = Path(sys.prefix).resolve()
    if prefix == Path(sys.base_prefix).resolve():
        errors.append(f"{sys.executable} is not in a virtual environment")
    origin = Path(mymyr.__file__).resolve()
    if not origin.is_relative_to(prefix):
        errors.append(f"mymyr was imported from {origin}, outside the environment {prefix}")
    package = origin.parent
    with zipfile.ZipFile(wheel) as archive:
        record = next(n for n in archive.namelist() if n.endswith(".dist-info/RECORD"))
        rows = list(csv.reader(io.TextIOWrapper(archive.open(record), encoding="utf-8")))
    checked = 0
    for path, digest, _ in rows:
        if not path.startswith("mymyr/") or not path.endswith((".py", ".so")):
            continue
        algorithm, _, expected = digest.partition("=")
        installed = package.parent / path
        if not installed.is_file():
            errors.append(f"{path} of the wheel is not installed under {package.parent}")
            continue
        actual = base64.urlsafe_b64encode(hashlib.new(algorithm, installed.read_bytes()).digest()).rstrip(b"=")
        if actual.decode() != expected:
            errors.append(f"{installed} differs from the wheel's {path}")
        checked += 1
    core = importlib.import_module("mymyr._core")
    if not Path(core.__file__).resolve().is_relative_to(package):
        errors.append(f"mymyr._core was imported from {core.__file__}, outside {package}")
    if checked == 0:
        errors.append("the wheel's RECORD lists no mymyr modules")
    print(f"{checked} files of {wheel.name} installed in {prefix}")
    return errors


def torch_errors(torch, task_path: Path, gpu: bool) -> list[str]:
    """torch and mymyr exchange state words: the expansions agree with the NumPy one."""
    import numpy as np

    import mymyr
    from mymyr import rl

    task = mymyr.Task.from_text(str(task_path), atoms="dense")
    states = [task.initial_state, *task.successor_states(task.initial_state)]
    words = np.zeros((len(states), task.words), dtype=np.uint64)
    for i, state in enumerate(states):
        words[i, : state.num_words] = state.words
    want = rl.expand(task, words)
    errors = []

    def compare(label, got, succ):
        if got.total != want.total or not np.array_equal(succ, np.asarray(want.succ).view(np.uint64)):
            errors.append(f"{label}: {got.total} successors differ from the NumPy expansion's {want.total}")

    tensor = torch.from_numpy(words.view(np.int64))
    got = rl.expand(task, tensor)
    compare("expand of a torch CPU tensor", got, torch.from_dlpack(got.succ).numpy().view(np.uint64))
    if gpu:
        import mymyr.cuda

        ctx = mymyr.cuda.Context(0, max_bytes=1 << 30)
        got = rl.expand(task, tensor.cuda(), ctx=ctx)
        if not isinstance(got.succ, torch.Tensor) or not got.succ.is_cuda:
            errors.append(f"expand of a torch CUDA tensor gave {type(got.succ).__name__}, not a torch CUDA tensor")
        else:
            compare("expand of a torch CUDA tensor", got, got.succ.cpu().numpy().view(np.uint64))
        got = rl.expand(task, mymyr.cuda.to_memory(words, "device", ctx))
        succ = torch.from_dlpack(got.succ)
        if not succ.is_cuda:
            errors.append("torch.from_dlpack of a mymyr device array is not on the GPU")
        compare("torch.from_dlpack of a device expansion", got, succ.cpu().numpy().view(np.uint64))
        ctx.synchronize()
        torch.cuda.synchronize()
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}): {want.total} successors through DLPack"
          f"{' on the CPU and GPU 0' if gpu else ' on the CPU'}")
    return errors


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True)
    parser.add_argument("--archs", required=True, help="space-separated compute capabilities, e.g. '86 89 90'")
    parser.add_argument("--wheel", type=Path, help="the wheel mymyr must have been installed from")
    parser.add_argument("--torch", help="import torch first; its version must start with this, e.g. 2.14")
    parser.add_argument("--task", type=Path, help="a task file for the torch exchange (with --torch)")
    parser.add_argument("--gpu", action="store_true", help="also check device 0")
    args = parser.parse_args(argv[1:])
    if (args.torch is None) != (args.task is None):
        parser.error("--torch and --task go together")

    errors = []
    torch = None
    if args.torch is not None:
        if "mymyr" in sys.modules:
            errors.append("mymyr was imported before torch")
        import torch

        if not torch.__version__.startswith(f"{args.torch}."):
            errors.append(f"torch {torch.__version__}, expected {args.torch}.*")
        if not (torch.version.cuda or "").startswith("13.0"):
            errors.append(f"torch {torch.__version__} is built for CUDA {torch.version.cuda}, expected 13.0")

    import mymyr
    import mymyr.cuda
    from mymyr._core import _encoding_jax, _rl_jax

    versions = (mymyr.__version__, importlib.metadata.version("mymyr"))
    if versions != (args.version, args.version):
        errors.append(f"version {versions[0]} (metadata {versions[1]}), expected {args.version}")
    info = mymyr.build_info()
    print(info)
    print(f"from {mymyr.__file__}")
    if args.wheel is not None:
        errors += wheel_errors(args.wheel, args.version)
    _, _, cuda = info.rstrip(")").partition(", CUDA ")
    code = set(cuda.partition(", ")[2].split())
    if not code:
        errors.append("build_info() names no CUDA toolkit and GPU code")
    missing = [arch for arch in args.archs.split() if f"sm_{arch}" not in code]
    if missing:
        errors.append(f"no SASS for {', '.join(f'sm_{a}' for a in missing)} (the build has {' '.join(sorted(code))})")
    for name, module in (("mymyr.rl.jax", _rl_jax), ("mymyr.encoding.jax", _encoding_jax)):
        platforms = {target[1] for target in module.ffi_targets()}
        if platforms != {"cpu", "CUDA"}:
            errors.append(f"the XLA FFI targets of {name} are for {sorted(platforms) or 'no platform'}, "
                          "expected cpu and CUDA")
    gpu = False
    if args.gpu:
        if not mymyr.cuda.available():
            errors.append("no CUDA device is visible")
        else:
            capability = mymyr.cuda.Context(0).compute_capability
            print(f"device 0: compute capability {capability // 10}.{capability % 10}")
            if f"sm_{capability}" not in code:
                errors.append(f"no SASS for device 0 (sm_{capability})")
            elif torch is not None and not torch.cuda.is_available():
                errors.append("torch sees no CUDA device")
            else:
                gpu = True
    if torch is not None:
        try:
            errors += torch_errors(torch, args.task, gpu)
        except Exception as e:  # reported with the other errors
            errors.append(f"the exchange with torch failed: {type(e).__name__}: {e}")

    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
