#!/usr/bin/env python3
"""Check that a release tag matches the package version in pyproject.toml.

usage: check_release_version.py TAG [ARTIFACT_DIR]

The tag must be "v" followed by the version in pyproject.toml. If ARTIFACT_DIR is given, every wheel
and source distribution in it must carry that same version in its filename and metadata.
The artifact directory must contain the complete wheel matrix and one source distribution. When running under
GitHub Actions the version and a pre-release flag are written to the step outputs.
"""

import os
import re
import sys
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRERELEASE = re.compile(r"\d(a|b|rc)\d+|\.dev\d*")
WHEEL_ABIS = {"cp314", "cp314t", "cp315", "cp315t"}
WHEEL_PLATFORMS = {"manylinux_2_34_x86_64", "macosx_15_0_arm64"}


def check_metadata(path: Path, version: str) -> list[str]:
    try:
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as archive:
                entries = [name for name in archive.namelist()
                           if name.endswith(".dist-info/METADATA") and name.count("/") == 1]
                if len(entries) != 1:
                    raise ValueError("expected one .dist-info/METADATA")
                metadata = archive.read(entries[0])
        else:
            with tarfile.open(path, "r:gz") as archive:
                root = path.name.removesuffix(".tar.gz")
                stream = archive.extractfile(f"{root}/PKG-INFO")
                if stream is None:
                    raise ValueError("missing PKG-INFO")
                metadata = stream.read()
                stream = archive.extractfile(f"{root}/pyproject.toml")
                if stream is None:
                    raise ValueError("missing pyproject.toml")
                source_version = tomllib.loads(stream.read().decode())["project"]["version"]
                if source_version != version:
                    raise ValueError(f"source pyproject.toml has version {source_version}, expected {version}")
        message = BytesParser().parsebytes(metadata)
        if message.get_all("Name") != ["mymyr"] or message.get_all("Version") != [version]:
            raise ValueError(f"metadata must have Name: mymyr and Version: {version}")
    except (OSError, ValueError, KeyError, tarfile.TarError, zipfile.BadZipFile) as error:
        return [f"{path.name}: {error}"]
    return []


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 2
    tag = argv[1].removeprefix("refs/tags/")
    with open(ROOT / "pyproject.toml", "rb") as f:
        version = tomllib.load(f)["project"]["version"]

    errors = []
    if tag != f"v{version}":
        errors.append(f"tag '{tag}' does not match the package version: pyproject.toml has version "
                      f"'{version}', so the tag must be 'v{version}'")

    if len(argv) == 3:
        artifacts = sorted(Path(argv[2]).iterdir())
        if not artifacts:
            errors.append(f"no wheels or source distributions found in {argv[2]}")
        wheels = set()
        sdists = 0
        for p in artifacts:
            if not p.is_file() or not (p.suffix == ".whl" or p.name.endswith(".tar.gz")):
                errors.append(f"unexpected artifact: {p.name}")
                continue
            errors.extend(check_metadata(p, version))
            m = re.fullmatch(r"mymyr-([^-]+)(-.*\.whl|\.tar\.gz)", p.name)
            if m is None:
                errors.append(f"unexpected artifact name: {p.name}")
            elif m.group(1) != version:
                errors.append(f"{p.name} carries version {m.group(1)}, expected {version}")
            if p.suffix == ".whl":
                parts = p.stem.split("-")
                if len(parts) != 5:
                    errors.append(f"unexpected wheel name: {p.name}")
                    continue
                _, _, interpreter, abi, platforms = parts
                if interpreter != abi.removesuffix("t"):
                    errors.append(f"unexpected interpreter/ABI pair: {p.name}")
                for platform in platforms.split("."):
                    key = (abi, platform)
                    if key in wheels:
                        errors.append(f"duplicate wheel for {abi} on {platform}")
                    wheels.add(key)
            else:
                sdists += 1
        expected = {(abi, platform) for abi in WHEEL_ABIS for platform in WHEEL_PLATFORMS}
        if wheels != expected:
            errors.append(f"wheel matrix mismatch: missing {sorted(expected - wheels)}, "
                          f"unexpected {sorted(wheels - expected)}")
        if sdists != 1:
            errors.append(f"expected one source distribution, found {sdists}")
        print(f"checked {len(artifacts)} artifacts against version {version}")

    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"release version {version} (tag {tag})")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"version={version}\n")
            f.write(f"prerelease={'true' if PRERELEASE.search(version) else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
