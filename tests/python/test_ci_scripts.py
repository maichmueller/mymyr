import base64
import fnmatch
import hashlib
import io
import json
import os
import re
import runpy
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import tomllib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "ci" / "check_ctest_log.py"
COUNTS = json.loads((ROOT / "ci" / "expected_counts.json").read_text())


def ctest_log(total: int, skipped: int) -> str:
    skipped_lines = "".join(
        f"  {i} - skipped_{i} (Skipped)\n" for i in range(1, skipped + 1)
    )
    return f"100% tests passed, 0 tests failed out of {total}\n\n{skipped_lines}"


def check_log(
    tmp_path: Path, text: str, *, sanitized: bool = False, platform: str | None = None
) -> subprocess.CompletedProcess:
    log = tmp_path / "ctest.log"
    log.write_text(text)
    command = [sys.executable, str(CHECKER)]
    if sanitized:
        command.append("--sanitized")
    if platform is not None:
        command += ["--platform", platform]
    command.append(str(log))
    return subprocess.run(command, capture_output=True, text=True, check=False)


def test_ctest_log_passes_with_cpu_counts(tmp_path: Path) -> None:
    result = check_log(
        tmp_path,
        ctest_log(COUNTS["ctest_min_total"], COUNTS["ctest_max_skipped"]),
        platform="linux",
    )

    assert result.returncode == 0
    assert (
        f"{COUNTS['ctest_min_total']} tests, 0 failed, "
        f"{COUNTS['ctest_max_skipped']} skipped"
    ) in result.stdout


def test_ctest_log_fails_below_cpu_counts(tmp_path: Path) -> None:
    min_total = COUNTS["ctest_min_total"]
    max_skipped = COUNTS["ctest_max_skipped"]
    result = check_log(tmp_path, ctest_log(min_total - 1, max_skipped + 1), platform="linux")

    assert result.returncode == 1
    assert f"only {min_total - 1} tests were registered (expected at least {min_total})" in result.stderr
    assert f"{max_skipped + 1} tests skipped (expected at most {max_skipped})" in result.stderr


def test_ctest_log_passes_with_macos_counts(tmp_path: Path) -> None:
    total = COUNTS["ctest_min_total"] - COUNTS["ctest_linux_only_tests"]
    result = check_log(tmp_path, ctest_log(total, COUNTS["ctest_max_skipped"]), platform="macos")

    assert result.returncode == 0
    assert f"{total} tests, 0 failed" in result.stdout


def test_ctest_log_fails_below_macos_counts(tmp_path: Path) -> None:
    total = COUNTS["ctest_min_total"] - COUNTS["ctest_linux_only_tests"]
    result = check_log(tmp_path, ctest_log(total - 1, 0), platform="macos")

    assert result.returncode == 1
    assert f"only {total - 1} tests were registered (expected at least {total})" in result.stderr


def test_macos_log_fails_the_linux_check(tmp_path: Path) -> None:
    total = COUNTS["ctest_min_total"] - COUNTS["ctest_linux_only_tests"]
    result = check_log(tmp_path, ctest_log(total, 0), platform="linux")

    assert result.returncode == 1
    assert f"expected at least {COUNTS['ctest_min_total']}" in result.stderr


def test_ctest_log_passes_with_sanitized_counts(tmp_path: Path) -> None:
    result = check_log(
        tmp_path,
        ctest_log(COUNTS["ctest_sanitized_min_total"], COUNTS["ctest_sanitized_max_skipped"]),
        sanitized=True,
    )

    assert result.returncode == 0
    assert (
        f"{COUNTS['ctest_sanitized_min_total']} tests, 0 failed, "
        f"{COUNTS['ctest_sanitized_max_skipped']} skipped"
    ) in result.stdout


def test_ctest_log_fails_above_sanitized_skip_count(tmp_path: Path) -> None:
    max_skipped = COUNTS["ctest_sanitized_max_skipped"]
    result = check_log(
        tmp_path,
        ctest_log(COUNTS["ctest_sanitized_min_total"], max_skipped + 1),
        sanitized=True,
    )

    assert result.returncode == 1
    assert f"{max_skipped + 1} tests skipped (expected at most {max_skipped})" in result.stderr


# ------------------------------------------------------------------------------------------------ package versions


def version_tree(tmp_path: Path) -> Path:
    """A copy of the version script next to a pyproject.toml of version 1.2.3, which the script edits."""
    (tmp_path / "ci").mkdir()
    shutil.copy2(ROOT / "ci" / "set_rehearsal_version.py", tmp_path / "ci")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "mymyr"\nversion = "1.2.3"\n')
    return tmp_path


def set_version(tree: Path, *args: str) -> subprocess.CompletedProcess:
    output = tree / "github_output"
    output.unlink(missing_ok=True)
    env = {"PATH": "", "GITHUB_OUTPUT": str(output)}
    return subprocess.run([sys.executable, str(tree / "ci" / "set_rehearsal_version.py"), *args],
                          capture_output=True, text=True, check=False, env=env)


def project_version(tree: Path) -> str:
    return tomllib.loads((tree / "pyproject.toml").read_text())["project"]["version"]


def test_local_label_gives_a_local_version(tmp_path: Path) -> None:
    tree = version_tree(tmp_path)
    base = "1.2.3"
    result = set_version(tree, "--local", "cu13")

    assert result.returncode == 0, result.stderr
    assert project_version(tree) == f"{base}+cu13"
    assert f"package version {base}+cu13" in result.stdout
    assert (tree / "github_output").read_text() == f"version={base}+cu13\nprerelease=false\n"


def test_local_label_takes_dot_separated_parts(tmp_path: Path) -> None:
    tree = version_tree(tmp_path)
    base = "1.2.3"
    result = set_version(tree, "--local", "cu13.sm89")

    assert result.returncode == 0, result.stderr
    assert project_version(tree) == f"{base}+cu13.sm89"


def test_invalid_local_labels_leave_the_version(tmp_path: Path) -> None:
    tree = version_tree(tmp_path)
    before = (tree / "pyproject.toml").read_text()
    for label in ("CU13", "cu-13", "cu13.", "", "+cu13"):
        result = set_version(tree, "--local", label)
        assert result.returncode == 1, label
        assert "expected a local label" in result.stderr
    assert (tree / "pyproject.toml").read_text() == before


def test_rehearsal_version_from_the_run(tmp_path: Path) -> None:
    tree = version_tree(tmp_path)
    base = "1.2.3"
    result = set_version(tree, "123", "2")

    assert result.returncode == 0, result.stderr
    assert project_version(tree) == f"{base}.dev123000002"
    assert (tree / "github_output").read_text() == f"version={base}.dev123000002\nprerelease=true\n"


# ------------------------------------------------------------------------------------------------ CUDA components

INSTALLER = ROOT / "ci" / "install_cuda_redist.py"


def redist(tmp_path: Path, files: dict[str, bytes], *, sha256: str | None = None) -> Path:
    """A manifest of one component whose archive holds files under one top-level directory, as NVIDIA's do."""
    archives = tmp_path / "archives"
    path = "tool/linux-x86_64/tool-linux-x86_64-1.0-archive.tar.xz"
    (archives / path).parent.mkdir(parents=True)
    with tarfile.open(archives / path, "w:xz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(f"tool-linux-x86_64-1.0-archive/{name}")
            info.size = len(data)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(data))
    data = (archives / path).read_bytes()
    manifest = {
        "release": "1.0",
        "base_url": archives.as_uri() + "/",
        "components": {"tool": {"version": "1.0", "path": path,
                                 "sha256": sha256 or hashlib.sha256(data).hexdigest(), "size": len(data)}},
    }
    (tmp_path / "redist.json").write_text(json.dumps(manifest))
    return tmp_path / "redist.json"


def install(manifest: Path, prefix: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(INSTALLER), str(manifest), str(prefix)],
                          capture_output=True, text=True, check=False)


def test_cuda_components_install_once(tmp_path: Path) -> None:
    manifest = redist(tmp_path, {"bin/nvcc": b"#!/bin/sh\n", "lib/libcudart_static.a": b"a", "LICENSE": b"terms"})
    prefix = tmp_path / "toolchain" / "cuda"
    result = install(manifest, prefix)

    assert result.returncode == 0, result.stderr
    assert (prefix / "bin" / "nvcc").read_bytes() == b"#!/bin/sh\n"
    assert (prefix / "lib64").resolve() == (prefix / "lib").resolve()
    assert (prefix / "lib64" / "libcudart_static.a").is_file()
    assert (prefix / "licenses" / "tool" / "LICENSE").read_bytes() == b"terms"
    assert not (prefix / "LICENSE").exists()

    again = install(manifest, prefix)
    assert again.returncode == 0, again.stderr
    assert "already installed" in again.stdout


def test_cuda_components_with_a_wrong_checksum_are_refused(tmp_path: Path) -> None:
    manifest = redist(tmp_path, {"bin/nvcc": b"x"}, sha256="0" * 64)
    prefix = tmp_path / "toolchain" / "cuda"
    result = install(manifest, prefix)

    assert result.returncode != 0
    assert "SHA256" in result.stderr
    assert not prefix.exists()
    assert list((tmp_path / "toolchain").iterdir()) == []


def test_cuda_manifest_pins_every_compiler_component() -> None:
    manifest = json.loads((ROOT / "ci" / "cuda_redist.json").read_text())

    assert manifest["base_url"].startswith("https://developer.download.nvidia.com/")
    assert {"cuda_nvcc", "cuda_crt", "cuda_cudart", "cccl", "libnvvm", "cuda_cuobjdump"} <= set(manifest["components"])
    for name, entry in manifest["components"].items():
        assert entry["path"].startswith(f"{name}/linux-x86_64/"), name
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), name
        assert entry["size"] > 0, name


# ------------------------------------------------------------------------------------------------ CUDA wheels

ATTACH = ROOT / "ci" / "attach_cuda_wheels.sh"
CHECK_INSTALL = ROOT / "ci" / "check_cuda_install.py"


def fake_wheels(directory: Path, version: str, abis: list[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for abi in abis:
        (directory / f"mymyr-{version}-{abi.rstrip('t')}-{abi}-manylinux_2_34_x86_64.whl").write_bytes(b"")
    return directory


def attach_dry_run(wheels: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(ATTACH), "--dry-run", "--dir", str(wheels), "--python", sys.executable,
                           *args, "v1.2.3"], capture_output=True, text=True, check=False)


def test_attach_refuses_an_incomplete_set_of_wheels(tmp_path: Path) -> None:
    result = attach_dry_run(fake_wheels(tmp_path / "wheels", "1.2.3+cu13", ["cp314t"]))

    assert result.returncode == 1
    assert "0 wheels for cp315t" in result.stderr
    assert "not the complete set for cp314t, cp315t; nothing tested or attached" in result.stderr
    assert "== mymyr-" not in result.stdout


def test_attach_refuses_a_directory_without_wheels(tmp_path: Path) -> None:
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    result = attach_dry_run(wheels)

    assert result.returncode == 1
    assert f"no mymyr-*.whl in {wheels}; nothing tested or attached" in result.stderr


def test_attach_refuses_wheels_of_other_abis_and_versions(tmp_path: Path) -> None:
    wheels = fake_wheels(tmp_path / "wheels", "1.2.3+cu13", ["cp314t", "cp315t", "cp313t"])
    fake_wheels(wheels, "1.2.2+cu13", ["cp314t"])
    result = attach_dry_run(wheels)

    assert result.returncode == 1
    assert "is for cp313t, which is not in --abis" in result.stderr
    assert "mymyr-1.2.2+cu13-cp314-cp314t-manylinux_2_34_x86_64.whl is not mymyr-1.2.3+cu13-" in result.stderr
    assert "2 wheels for cp314t" in result.stderr
    assert "== mymyr-" not in result.stdout


def test_attach_takes_the_abis_it_is_given(tmp_path: Path) -> None:
    result = attach_dry_run(fake_wheels(tmp_path / "wheels", "1.2.3+cu13", ["cp315t"]), "--abis", "cp314t")

    assert result.returncode == 1
    assert "0 wheels for cp314t" in result.stderr
    assert "is for cp315t, which is not in --abis" in result.stderr


def test_attach_needs_a_directory_or_a_run(tmp_path: Path) -> None:
    result = subprocess.run(["bash", str(ATTACH), "--dry-run", "--python", sys.executable, "v1.2.3"],
                            capture_output=True, text=True, check=False)

    assert result.returncode == 2
    assert "pass the wheels' directory (--dir) or the CUDA wheel workflow run that built them (--run)" in result.stderr


def test_cuda_wheel_defaults_agree() -> None:
    workflow = (ROOT / ".github" / "workflows" / "cuda-wheels.yml").read_text()
    attach = ATTACH.read_text()
    build = (ROOT / "ci" / "cuda_wheel.sh").read_text()
    local = LOCAL_WHEELS.read_text()
    archs = re.findall(r"^ +default: ([0-9a-z;-]+-real)$", workflow, re.MULTILINE)
    abis = re.findall(r"^ +default: (cp3[0-9 cpt]+)$", workflow, re.MULTILINE)

    assert archs == ["86-real;89-real;90-real"]
    assert abis == ["cp314t cp315t"]
    assert f'archs="{archs[0]}"' in build
    assert f'archs="{archs[0]}"' in local
    assert 'archs="{}"'.format(archs[0].replace("-real", "").replace(";", " ")) in attach
    assert f'abis="{abis[0]}"' in attach
    assert f'abis="{abis[0]}"' in local


def test_releases_do_not_build_the_cuda_wheels_on_github() -> None:
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    workflow = (ROOT / ".github" / "workflows" / "cuda-wheels.yml").read_text()

    assert "cuda-wheels.yml" not in re.sub(r"^ *#.*$", "", release, flags=re.MULTILINE)
    assert "workflow_call" not in workflow
    assert "workflow_dispatch" in workflow


LOCAL_WHEELS = ROOT / "ci" / "local_cuda_wheels.sh"
IMAGE = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["cibuildwheel"]["linux"]["manylinux-x86_64-image"]


def local_wheels(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(LOCAL_WHEELS), "--toolchain", str(tmp_path / "toolchain"), "--work",
                           str(tmp_path / "work"), "--out", str(tmp_path / "out"), "--dry-run", *args],
                          capture_output=True, text=True, check=False)


def local_commands(result: subprocess.CompletedProcess) -> tuple[str, list[list[str]]]:
    """The summary line and the container commands of a dry run, the toolchain's first."""
    assert result.returncode == 0, result.stderr
    summary, *commands = result.stdout.splitlines()
    return summary, [shlex.split(command) for command in commands]


def podman(tmp_path: Path, work: str, gb: int) -> list[str]:
    return ["podman", "run", "--rm", "--memory", f"{gb}g", "--volume", f"{ROOT}:/src",
            "--volume", f"{tmp_path / 'toolchain'}:/opt/mymyr-toolchain",
            "--volume", f"{tmp_path / 'work' / work}:/work", "--volume", f"{tmp_path / 'out'}:/out", IMAGE, "bash", "/src/ci/cuda_wheel.sh", "--work", "/work"]


def test_local_wheels_derive_the_jobs_from_the_memory_and_the_cores(tmp_path: Path) -> None:
    summary, commands = local_commands(local_wheels(tmp_path, "--cpus", "64", "--memory-gb", "120",
                                                    "--engine", "podman"))

    # 7 GB for the largest kernels, 3 per other compile and 2 for the rest: 7 + 3 * 17 + 2 = 60 GB per ABI
    assert summary == ("== 64 cores and 120 GB: 2 of 2 ABIs at a time, each with --jobs 18 --cuda-jobs 17 in 60 GB;"
                       " the toolchain with --jobs 39")
    assert commands == [
        [*podman(tmp_path, "toolchain", 120), "--stage", "toolchain", "--jobs", "39"],
        *([*podman(tmp_path, abi, 60), "--python", abi, "--archs", "86-real;89-real;90-real", "--out", "/out",
           "--jobs", "18", "--cuda-jobs", "17"] for abi in ["cp314t", "cp315t"]),
    ]


def test_local_wheels_build_one_abi_at_a_time_when_two_would_get_fewer_than_8_jobs(tmp_path: Path) -> None:
    summary, _ = local_commands(local_wheels(tmp_path, "--cpus", "16", "--memory-gb", "500"))
    assert "2 of 2 ABIs at a time, each with --jobs 8 --cuda-jobs 7 in 250 GB" in summary

    summary, _ = local_commands(local_wheels(tmp_path, "--cpus", "12", "--memory-gb", "500"))
    assert "1 of 2 ABIs at a time, each with --jobs 12 --cuda-jobs 11 in 500 GB" in summary

    summary, _ = local_commands(local_wheels(tmp_path, "--cpus", "64", "--memory-gb", "30"))
    assert "1 of 2 ABIs at a time, each with --jobs 8 --cuda-jobs 7 in 30 GB" in summary


def test_local_wheels_take_the_numbers_they_are_given(tmp_path: Path) -> None:
    summary, commands = local_commands(local_wheels(
        tmp_path, "--cpus", "64", "--memory-gb", "120", "--parallel", "1", "--jobs", "5", "--cuda-jobs", "2",
        "--python", "cp314", "--archs", "80-real", "--image", "example.org/image:1", "--engine", "docker"))

    assert "1 of 1 ABIs at a time, each with --jobs 5 --cuda-jobs 2 in 120 GB" in summary
    expected = podman(tmp_path, "cp314", 120)
    expected[0] = "docker"
    expected[expected.index(IMAGE)] = "example.org/image:1"
    assert commands[1] == [*expected, "--python", "cp314", "--archs", "80-real", "--out", "/out", "--jobs", "5",
                           "--cuda-jobs", "2"]


def test_local_wheels_with_apptainer_bind_the_same_directories(tmp_path: Path) -> None:
    _, commands = local_commands(local_wheels(tmp_path, "--cpus", "8", "--memory-gb", "16", "--engine", "apptainer",
                                              "--python", "cp315t"))

    binds = (f"{ROOT}:/src,{tmp_path / 'toolchain'}:/opt/mymyr-toolchain,{tmp_path / 'work' / 'cp315t'}:/work,"
             f"{tmp_path / 'out'}:/out")
    assert commands[1][:6] == ["apptainer", "exec", "--containall", "--bind", binds, f"docker://{IMAGE}"]
    assert "--memory" not in commands[1]


@pytest.mark.parametrize(("args", "status", "message"), [
    (["--memory-gb", "8"], 1, "8 GB per build (8 GB for 1 at a time) is less than the 9 GB that one build needs"),
    (["--memory-gb", "16", "--parallel", "2"], 1, "8 GB per build (16 GB for 2 at a time) is less than the 9 GB"),
    (["--python", "cp314t py3"], 2, "expected CPython ABIs such as cp314t, got 'py3'"),
    (["--python", " "], 2, "--python names no ABI"),
    (["--jobs", "0"], 2, "must be positive integers"),
    (["--cpus", "many"], 2, "must be positive integers"),
    (["--engine", "lxc"], 2, "--engine takes podman, docker or apptainer, not 'lxc'"),
    (["--archs"], 2, "usage: ci/local_cuda_wheels.sh"),
    (["--stage", "all"], 2, "usage: ci/local_cuda_wheels.sh"),
])
def test_local_wheels_refuse_bad_arguments(tmp_path: Path, args: list[str], status: int, message: str) -> None:
    result = local_wheels(tmp_path, "--cpus", "8", "--memory-gb", "64", "--engine", "podman", *args)

    assert result.returncode == status
    assert message in result.stderr
    assert result.stdout == ""


def test_local_wheels_need_a_toolchain_and_a_work_directory(tmp_path: Path) -> None:
    result = subprocess.run(["bash", str(LOCAL_WHEELS), "--work", str(tmp_path), "--dry-run"],
                            capture_output=True, text=True, check=False)

    assert result.returncode == 2
    assert "usage: ci/local_cuda_wheels.sh --toolchain DIR --work DIR" in result.stderr


def pins(path: Path) -> dict[str, list[tuple[str, str]]]:
    """name -> [(version, environment marker)] of a hashed requirements file; every pin must carry a hash."""
    out: dict[str, list[tuple[str, str]]] = {}
    for entry in re.split(r"\n(?=[a-z])", path.read_text()):
        match = re.match(r"([a-z0-9._-]+)==([^ ;\\]+)(?: ; ([^\\]+))?", entry)
        if match:
            assert "--hash=sha256:" in entry, match[1]
            out.setdefault(match[1], []).append((match[2], (match[3] or "").strip()))
    return out


def satisfies(version: str, specifiers: str) -> bool:
    def key(v: str) -> tuple[int, ...]:
        return tuple(int(part) for part in re.match(r"[0-9.]+", v)[0].strip(".").split("."))

    checks = {">=": lambda a, b: a >= b, "<": lambda a, b: a < b, "==": lambda a, b: a == b}
    for spec in filter(None, specifiers.split(",")):
        op, bound = re.fullmatch(r"(>=|<|==)\s*([0-9.]+)", spec.strip()).groups()
        if not checks[op](key(version), key(bound)):
            return False
    return True


def test_cuda_wheel_requirements_are_pinned_within_the_project_ranges() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    build = pins(ROOT / "ci" / "cuda_wheel_build_requirements.txt")
    test = pins(ROOT / "ci" / "cuda_wheel_test_requirements.txt")
    required = [*project["build-system"]["requires"], "cmake>=4.4"]
    tested = [*project["project"]["optional-dependencies"]["test"], *project["project"]["dependencies"]]

    for requirements, lock in ((required, build), (tested, test)):
        for requirement in requirements:
            name, specifiers = re.fullmatch(r"([a-z0-9._-]+)(.*)", requirement).groups()
            assert name in lock, name
            for version, _ in lock[name]:
                assert satisfies(version, specifiers), (name, version, specifiers)
    assert build["nanobind"] == test["nanobind"]
    assert "triton" not in test
    assert [v for v, marker in test["torch"] if marker == "sys_platform != 'darwin'"] == ["2.14.1+cu130"]


def installed_copy(tmp_path: Path, version: str) -> tuple[Path, Path, Path]:
    """A wheel of a stand-in mymyr, and a fresh virtual environment (without pip) and its site packages."""
    files = {"mymyr/__init__.py": f"__version__ = {version!r}\n".encode(), "mymyr/_core.py": b"BUILD = 'wheel'\n"}
    record = "".join(
        f"{name},sha256={base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()},{len(data)}\n"
        for name, data in files.items()
    ) + f"mymyr-{version}.dist-info/RECORD,,\n"
    wheel = tmp_path / f"mymyr-{version}-cp314-cp314t-manylinux_2_34_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr(f"mymyr-{version}.dist-info/RECORD", record)
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    purelib = "import sysconfig; print(sysconfig.get_path('purelib'))"
    site = subprocess.run([str(venv / "bin" / "python"), "-c", purelib],
                          capture_output=True, text=True, check=True).stdout.strip()
    return wheel, venv, Path(site)


def wheel_errors(venv: Path, wheel: Path, version: str, **env: str) -> list[str]:
    """check_cuda_install.wheel_errors, run by the environment's interpreter."""
    code = ("import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "from check_cuda_install import wheel_errors; "
            "[print('error:', e) for e in wheel_errors(Path(sys.argv[2]), sys.argv[3])]")
    result = subprocess.run([str(venv / "bin" / "python"), "-c", code, str(CHECK_INSTALL.parent), str(wheel), version],
                            capture_output=True, text=True, check=True, env={"PATH": "", **env})
    return [line for line in result.stdout.splitlines() if line.startswith("error: ")]


def unpack(wheel: Path, into: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(into)


def test_the_wheel_check_accepts_the_wheels_own_files(tmp_path: Path) -> None:
    wheel, venv, site = installed_copy(tmp_path, "1.2.3+cu13")
    unpack(wheel, site)

    assert wheel_errors(venv, wheel, "1.2.3+cu13") == []


def test_the_wheel_check_refuses_a_copy_outside_the_environment(tmp_path: Path) -> None:
    # a mymyr of the same version that a base interpreter's site packages would provide
    wheel, venv, _ = installed_copy(tmp_path, "1.2.3+cu13")
    base = tmp_path / "base-site"
    unpack(wheel, base)
    errors = wheel_errors(venv, wheel, "1.2.3+cu13", PYTHONPATH=str(base))

    assert any("outside the environment" in e for e in errors), errors


def test_the_wheel_check_refuses_another_build_of_the_same_version(tmp_path: Path) -> None:
    wheel, venv, site = installed_copy(tmp_path, "1.2.3+cu13")
    unpack(wheel, site)
    (site / "mymyr" / "_core.py").write_text("BUILD = 'another'\n")
    errors = wheel_errors(venv, wheel, "1.2.3+cu13")

    assert any("_core.py differs from the wheel's mymyr/_core.py" in e for e in errors), errors
    assert any("is not version 1.2.4" in e for e in wheel_errors(venv, wheel, "1.2.4"))


RUN_UNTIL = ROOT / "ci" / "run_until.py"
proc = pytest.mark.skipif(not os.path.isdir("/proc/self"),
                          reason="ci/run_until.py finds the build's Ninja processes through /proc, as on Linux")


def fake_ninja(tmp_path: Path) -> Path:
    """An executable named ninja (a script run by its interpreter directly keeps that process name) that records its
    start and its SIGINT in the file given as its argument, and ends by itself after 30 seconds."""
    ninja = tmp_path / "bin" / "ninja"
    ninja.parent.mkdir()
    ninja.write_text('#!/bin/bash\ntrap \'echo interrupted > "$1"; exit 130\' INT\n'
                     'echo started > "$1"\nfor _ in $(seq 600); do sleep 0.05; done\n')
    ninja.chmod(0o755)
    return ninja


def wait_for(path: Path, text: str) -> None:
    for _ in range(200):
        if path.is_file() and path.read_text().strip() == text:
            return
        time.sleep(0.05)
    raise AssertionError(f"{path} never read {text!r}")


def run_until(deadline: float, *command: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(RUN_UNTIL), str(deadline), "--", *command],
                          capture_output=True, text=True, check=False, timeout=60)


# A background job of a non-interactive shell starts with SIGINT ignored, and so would the build without run_until
# resetting it: an ignored signal stays ignored across exec, and bash cannot trap a signal ignored on entry.
@proc
@pytest.mark.parametrize("sigint", ["inherited", "ignored"])
def test_run_until_stops_the_builds_ninja_at_the_deadline(tmp_path: Path, monkeypatch, capsys, sigint: str) -> None:
    ninja = fake_ninja(tmp_path)
    marker = tmp_path / "build"
    main = runpy.run_path(str(RUN_UNTIL))["main"]

    def expired_deadline() -> float:
        wait_for(marker, "started")
        return 1.0

    monkeypatch.setitem(main.__globals__, "time", SimpleNamespace(time=expired_deadline, sleep=time.sleep))
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN) if sigint == "ignored" else None
    try:
        # The child installs its signal handler before the clock reaches the deadline.
        # The build tool fails when its Ninja is interrupted, as CMake does.
        status = main([str(RUN_UNTIL), "0", "--", "bash", "-c", '"$0" "$1"; exit 1', str(ninja), str(marker)])
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)

    assert status == 75
    assert marker.read_text().strip() == "interrupted"
    assert "stopping the build at the deadline" in capsys.readouterr().err


@proc
def test_run_until_passes_on_the_status_of_a_build_that_ends_in_time(tmp_path: Path) -> None:
    assert run_until(time.time() + 60, "bash", "-c", "exit 0").returncode == 0
    assert run_until(time.time() + 60, "bash", "-c", "exit 4").returncode == 4


def test_run_until_refuses_to_start_without_proc(tmp_path: Path, capsys) -> None:
    marker = tmp_path / "build"
    main = runpy.run_path(str(RUN_UNTIL))["main"]
    main.__globals__["PROC"] = str(tmp_path / "proc")

    assert main([str(RUN_UNTIL), "0", "--", "touch", str(marker)]) == 2
    assert not marker.exists()
    assert f"through {tmp_path / 'proc'}, which this system does not have" in capsys.readouterr().err


@proc
def test_run_until_leaves_other_ninjas_alone(tmp_path: Path) -> None:
    ninja = fake_ninja(tmp_path)
    other = tmp_path / "other"
    outsider = subprocess.Popen([str(ninja), str(other)])
    try:
        wait_for(other, "started")
        result = run_until(time.time() + 1, "bash", "-c", "sleep 2; exit 0")

        assert result.returncode == 0, result.stderr
        assert other.read_text().strip() == "started"
    finally:
        outsider.kill()
        outsider.wait()


BUILD_STATE = ROOT / "ci" / "build_state.sh"
# the script drives GNU tar, coreutils and findutils, as on the Linux runners
gnu_tools = pytest.mark.skipif(sys.platform != "linux", reason="ci/build_state.sh needs GNU tar, coreutils and findutils")


def build_state(tmp_path: Path, *args: object) -> subprocess.CompletedProcess:
    """ci/build_state.sh with a zstd that passes the data through: the tests check what the script does with the
    archives, the dry run of the workflow the compression."""
    stub = tmp_path / "bin" / "zstd"
    if not stub.exists():
        stub.parent.mkdir()
        stub.write_text("#!/bin/sh\nexec cat\n")
        stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{stub.parent}:{os.environ['PATH']}"}
    return subprocess.run(["bash", str(BUILD_STATE), *map(str, args)], capture_output=True, text=True, env=env,
                          check=False, timeout=60)


def snapshot(*roots: Path) -> dict[str, tuple]:
    """Every entry under the roots: type, mode, size, modification second, link target and content hash (of the
    files under 100 MB; larger ones are sparse)."""
    out = {}
    for root in roots:
        for path in [root, *sorted(root.rglob("*"))]:
            st = path.lstat()
            kind = "link" if path.is_symlink() else "dir" if path.is_dir() else "file"
            small = kind == "file" and st.st_size < 10**8
            content = hashlib.sha256(path.read_bytes()).hexdigest() if small else None
            target = os.readlink(path) if kind == "link" else None
            out[str(path)] = (kind, st.st_mode, st.st_size if kind == "file" else None, int(st.st_mtime), target,
                              content)
    return out


def build_tree(root: Path) -> None:
    (root / "obj" / "deep").mkdir(parents=True)
    (root / "empty").mkdir()
    (root / "obj" / "deep" / "kernel.o").write_bytes(os.urandom(200_000))
    (root / "build.ninja").write_text("rule cc\n" * 1000)
    (root / "tool").write_text("#!/bin/sh\n")
    (root / "tool").chmod(0o755)
    (root / "latest").symlink_to("obj/deep/kernel.o")
    # sub-second modification times, as a build leaves them
    os.utime(root / "obj" / "deep" / "kernel.o", ns=(1_700_000_000_123_456_789, 1_700_000_000_987_654_321))
    os.utime(root / "obj", ns=(1_700_000_100_500_000_000, 1_700_000_100_500_000_000))


def manifest(tmp_path: Path, tree_bytes: int) -> Path:
    path = tmp_path / "manifest"
    path.write_text(f"part-bytes 1000\narchive 0 1000 {tree_bytes} {tmp_path / 'restored' / 'work'}\n")
    return path


@gnu_tools
def test_build_state_round_trip_keeps_files_modes_links_and_modification_seconds(tmp_path: Path) -> None:
    work, toolchain = tmp_path / "runner" / "work", tmp_path / "workspace" / "toolchain"
    build_tree(work)
    (toolchain / "bin").mkdir(parents=True)
    (toolchain / "bin" / "nvcc").write_text("nvcc")
    before = snapshot(work, toolchain)

    packed = build_state(tmp_path, "pack", "--part-size", "64K", "--state", tmp_path / "state", work, toolchain)

    assert packed.returncode == 0, packed.stderr
    state = Path(packed.stdout.strip())
    assert state == tmp_path / "state"
    assert not work.exists() and not toolchain.exists()
    lines = (state / "manifest").read_text().splitlines()
    assert lines[0] == "part-bytes 65536"
    assert [line.split()[0:2] + line.split()[4:] for line in lines[1:]] == [
        ["archive", "0", str(work)], ["archive", "1", str(toolchain)]]
    parts = sorted(p.name for p in state.glob("0.tar.zst.*"))
    assert len(parts) >= 3 and parts[:3] == ["0.tar.zst.000", "0.tar.zst.001", "0.tar.zst.002"]
    assert int(lines[1].split()[2]) == sum(p.stat().st_size for p in state.glob("0.tar.zst.*"))

    unpacked = build_state(tmp_path, "unpack", state)

    assert unpacked.returncode == 0, unpacked.stderr
    assert snapshot(work, toolchain) == before
    assert not state.exists()


@gnu_tools
def test_build_state_packs_only_where_the_largest_file_fits(tmp_path: Path) -> None:
    work = tmp_path / "work"
    build_tree(work)
    # a sparse file larger than the free space: packing next to it would need its whole size at once
    huge = work / "huge.a"
    try:
        with huge.open("wb") as stream:
            stream.truncate(shutil.disk_usage(tmp_path).free + 10**12)
    except OSError:
        pytest.skip("the file system takes no sparse file larger than its free space")
    before = snapshot(work)

    result = build_state(tmp_path, "pack", "--state", tmp_path / "state", work)

    assert result.returncode == 1
    assert "error: no room for the build state" in result.stderr and "free, needs" in result.stderr
    assert snapshot(work) == before
    assert not (tmp_path / "state").exists()


@gnu_tools
def test_build_state_room_names_the_first_directory_that_fits(tmp_path: Path) -> None:
    first, second = tmp_path / "mnt" / "state", tmp_path / "temp" / "state"

    fits = build_state(tmp_path, "room", manifest(tmp_path, 5000), first, second)

    assert fits.returncode == 0, fits.stderr
    assert fits.stdout.strip() == str(first)

    lacks = build_state(tmp_path, "room", manifest(tmp_path, shutil.disk_usage(tmp_path).free + 10**12), first, second)

    assert lacks.returncode == 1
    assert "error: no room to receive and unpack the build state" in lacks.stderr
    assert lacks.stderr.count("free, needs") == 2


@gnu_tools
def test_build_state_unpack_without_room_keeps_the_parts(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    manifest(tmp_path, shutil.disk_usage(tmp_path).free + 10**12).rename(state / "manifest")
    (state / "0.tar.zst.000").write_bytes(b"part")

    result = build_state(tmp_path, "unpack", state)

    assert result.returncode == 1
    assert "error: no room to unpack the build state" in result.stderr
    assert (state / "0.tar.zst.000").read_bytes() == b"part"
    assert not (tmp_path / "restored").exists()


@gnu_tools
def test_build_state_space_fails_with_the_room_a_job_needs(tmp_path: Path) -> None:
    assert build_state(tmp_path, "space", tmp_path, 0).returncode == 0

    result = build_state(tmp_path, "space", tmp_path / "not-yet", 10**9)

    assert result.returncode == 1
    assert f"error: {tmp_path} has" in result.stderr and "this job needs at least 1000000000 GB" in result.stderr


def cuda_wheel_jobs() -> dict[str, str]:
    workflow = (ROOT / ".github" / "workflows" / "cuda-wheels.yml").read_text()
    body = workflow.split("\njobs:\n", 1)[1]
    return dict(re.findall(r"^  ([a-z-]+):\n(.*?)(?=^  [a-z-]+:\n|\Z)", body, re.MULTILINE | re.DOTALL))


def test_cuda_wheel_jobs_stop_the_build_in_time_to_hand_it_on_or_test_it() -> None:
    jobs = cuda_wheel_jobs()
    room = {"compile": 45, "resume": 45, "wheel": 60}
    for name, minutes in room.items():
        timeout = int(re.search(r"timeout-minutes: (\d+)", jobs[name])[1])
        deadline = re.findall(r"--deadline \$\(\(JOB_START \+ (\d+) \* 60\)\)", jobs[name])
        assert len(deadline) == 1, name
        assert timeout - int(deadline[0]) >= minutes, name
        assert "--minutes" not in jobs[name], name
        assert jobs[name].index("JOB_START=$(date +%s)") < jobs[name].index("--deadline"), name


def test_cuda_build_state_artifacts_are_not_downloaded_as_wheels() -> None:
    jobs = cuda_wheel_jobs()
    actions = ROOT / ".github" / "actions"
    texts = [*jobs.values(), *(p.read_text() for p in actions.glob("*-cuda-build/action.yml"))]
    names = {n for text in texts for n in re.findall(r"^ +(?:name|artifact): (cuda-.+)$", text, re.MULTILINE)}
    # attach_cuda_wheels.sh downloads the artifacts that match cuda-wheel-*
    assert "--pattern 'cuda-wheel-*'" in ATTACH.read_text()
    assert {n for n in names if fnmatch.fnmatch(n, "cuda-wheel-*")} == {"cuda-wheel-${{ matrix.python }}"}
    assert "cuda-build-state-${{ matrix.python }}" in names
    assert "cuda-build-state-${{ matrix.python }}-manifest" in names
