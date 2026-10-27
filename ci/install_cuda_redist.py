#!/usr/bin/env python3
"""Install the CUDA compiler components listed in a manifest from NVIDIA's redistributable archives.

usage: install_cuda_redist.py MANIFEST PREFIX

MANIFEST (ci/cuda_redist.json) names each component's archive under base_url with its SHA256 and size; every
archive is checked against both before it is unpacked. PREFIX receives the toolkit layout (bin/nvcc, include/,
lib/, nvvm/, lib64 -> lib) and each component's license under licenses/<component>/. Running it again with the
same manifest does nothing, so PREFIX can be cached; a changed manifest replaces PREFIX.
"""

import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

LICENSE_FILES = {"LICENSE", "LICENSE.txt", "EULA.txt"}


def fetch(url: str, dest: Path, sha256: str, size: int) -> None:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as source, dest.open("wb") as out:
        while chunk := source.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    if dest.stat().st_size != size:
        raise RuntimeError(f"{url}: {dest.stat().st_size} bytes, expected {size}")
    if digest.hexdigest() != sha256:
        raise RuntimeError(f"{url}: SHA256 {digest.hexdigest()}, expected {sha256}")


def unpack(archive: Path, component: str, staging: Path, scratch: Path) -> None:
    target = scratch / component
    target.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
    roots = list(target.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise RuntimeError(f"{archive.name}: expected one top-level directory")
    licenses = staging / "licenses" / component
    licenses.mkdir(parents=True)
    for item in roots[0].iterdir():
        if item.name in LICENSE_FILES:
            shutil.copy2(item, licenses / item.name)
        elif item.is_dir():
            shutil.copytree(item, staging / item.name, symlinks=True, dirs_exist_ok=True)
        else:
            shutil.copy2(item, staging / item.name, follow_symlinks=False)
    shutil.rmtree(target)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    manifest_path = Path(argv[1])
    prefix = Path(argv[2]).absolute()
    text = manifest_path.read_bytes()
    stamp = prefix / f".installed-{hashlib.sha256(text).hexdigest()[:16]}"
    if stamp.exists():
        print(f"CUDA components already installed in {prefix}")
        return 0
    manifest = json.loads(text)

    prefix.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(prefix, ignore_errors=True)
    with tempfile.TemporaryDirectory(dir=prefix.parent, prefix=f".{prefix.name}-") as tmp:
        work = Path(tmp)
        staging = work / "prefix"
        staging.mkdir()
        (work / "unpack").mkdir()
        for component, entry in manifest["components"].items():
            archive = work / Path(entry["path"]).name
            fetch(manifest["base_url"] + entry["path"], archive, entry["sha256"], int(entry["size"]))
            unpack(archive, component, staging, work / "unpack")
            archive.unlink()
            print(f"installed {component} {entry['version']}")
        # the archives use lib/, where nvcc's profile looks for lib64/
        if (staging / "lib").is_dir() and not (staging / "lib64").exists():
            (staging / "lib64").symlink_to("lib", target_is_directory=True)
        shutil.copy2(manifest_path, staging / "redist.json")
        (staging / stamp.name).touch()
        staging.rename(prefix)
    print(f"CUDA {manifest['release']} components installed in {prefix}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
