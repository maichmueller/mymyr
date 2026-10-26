#!/usr/bin/env python3
"""Check an installed CUDA build of mymyr: its version, the GPU code it carries and its JAX targets.

usage: check_cuda_install.py --version VERSION --archs "89 90" [--gpu]

The installed mymyr must have the version VERSION (in its metadata and mymyr.__version__), a CUDA backend whose
build_info() lists SASS (sm_<n>) for every architecture in --archs, and the XLA FFI targets of mymyr.rl.jax and
mymyr.encoding.jax for the CPU and for CUDA. With --gpu a CUDA device must be visible, and the build must have SASS
for the architecture of device 0. Prints the build info and, with --gpu, the device.
"""

import argparse
import importlib.metadata
import sys


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", required=True)
    parser.add_argument("--archs", required=True, help="space-separated compute capabilities, e.g. '89 90'")
    parser.add_argument("--gpu", action="store_true", help="also check device 0")
    args = parser.parse_args(argv[1:])

    import mymyr
    import mymyr.cuda
    from mymyr._core import _encoding_jax, _rl_jax

    errors = []
    versions = (mymyr.__version__, importlib.metadata.version("mymyr"))
    if versions != (args.version, args.version):
        errors.append(f"version {versions[0]} (metadata {versions[1]}), expected {args.version}")
    info = mymyr.build_info()
    print(info)
    print(f"from {mymyr.__file__}")
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
    if args.gpu:
        if not mymyr.cuda.available():
            errors.append("no CUDA device is visible")
        else:
            capability = mymyr.cuda.Context(0).compute_capability
            print(f"device 0: compute capability {capability // 10}.{capability % 10}")
            if f"sm_{capability}" not in code:
                errors.append(f"no SASS for device 0 (sm_{capability})")

    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
