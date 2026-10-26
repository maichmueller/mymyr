import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path


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
