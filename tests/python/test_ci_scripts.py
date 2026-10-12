import json
import subprocess
import sys
from pathlib import Path

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


NO_REFERENCE_CHECK = ROOT / "ci" / "check_no_mimir.py"
REFERENCE_NAME = "mi" + "mir"


def run_reference_check(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(NO_REFERENCE_CHECK), str(root)], capture_output=True, text=True, check=False
    )


def test_the_tree_names_no_reference_implementation_outside_the_parity_area() -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    result = run_reference_check(ROOT)

    assert result.returncode == 0, result.stdout + result.stderr


def test_reference_names_fail_outside_the_parity_area(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.cpp").write_text(f"// as in {REFERENCE_NAME.capitalize()}'s search\n")

    result = run_reference_check(tmp_path)

    assert result.returncode == 1
    assert "src/a.cpp:1" in result.stdout


def test_reference_names_pass_in_the_allowed_places(tmp_path: Path) -> None:
    for rel in ("tests/parity/a.py", "bench/parity/b.cpp", "LICENSE"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(f"{REFERENCE_NAME}\n")
    (tmp_path / "README.md").write_text(
        f"A rewrite of the [{REFERENCE_NAME}](https://github.com/maichmueller/{REFERENCE_NAME}) core.\n"
        f"These are the terms of {REFERENCE_NAME}.\n"
    )

    result = run_reference_check(tmp_path)

    assert result.returncode == 0, result.stdout
