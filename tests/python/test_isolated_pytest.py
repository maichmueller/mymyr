"""Isolated tests retain pytest configuration and outcomes from any working directory."""

import pathlib
import shutil

import pytest

from python_children import run_code

HERE = pathlib.Path(__file__).resolve().parent


@pytest.mark.parametrize("working_directory", ["suite", "empty"])
def test_isolated_tests_keep_configuration_and_outcomes(working_directory, tmp_path):
    suite = tmp_path / "suite"
    tests = suite / "tests"
    tests.mkdir(parents=True)
    empty = tmp_path / "empty"
    empty.mkdir()
    shutil.copyfile(HERE / "conftest.py", tests / "conftest.py")
    (suite / "conftest.py").write_text("raise AssertionError('outside the conftest boundary')\n")
    ini = suite / "selected.ini"
    ini.write_text("[pytest]\naddopts = --strict-markers\nmarkers = configured\nempty_parameter_set_mark = xfail\n")
    module = tests / "test_selected.py"
    module.write_text(
        "import os, pathlib, warnings, pytest\n"
        "pytestmark = [pytest.mark.needs_gil, pytest.mark.configured]\n"
        "@pytest.mark.parametrize('value', ['first::case', 'skip'])\n"
        "def test_configuration(value, pytestconfig):\n"
        "    assert os.environ.get('MYMYR_TEST_ISOLATED_REPORT')\n"
        f"    assert pytestconfig.rootpath == pathlib.Path({str(suite)!r})\n"
        f"    assert pytestconfig.inipath == pathlib.Path({str(ini)!r})\n"
        "    assert pytestconfig.getoption('importmode') == 'importlib'\n"
        "    assert pytestconfig.getini('empty_parameter_set_mark') == 'skip'\n"
        "    with pytest.raises(UserWarning):\n"
        "        warnings.warn('warnings are errors')\n"
        "    if value == 'skip':\n"
        "        pytest.skip('child skip reason')\n"
        "def test_failure():\n"
        "    assert False, 'child assertion detail'\n"
        "def test_unselected():\n"
        "    assert False, 'unselected test ran'\n"
    )
    result = run_code(
        "import pytest; sys.exit(pytest.main(sys.argv[1:]))",
        "-q", "-rs", "-p", "no:cacheprovider", "-W", "error", "-c", str(ini),
        "--rootdir", str(suite), "--confcutdir", str(tests), "--import-mode", "importlib",
        "-o", "empty_parameter_set_mark=skip",
        f"{module}::test_configuration", f"{module}::test_failure",
        cwd=suite if working_directory == "suite" else empty, capture_output=True, check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 failed, 1 passed, 1 skipped" in result.stdout, result.stdout + result.stderr
    assert "child assertion detail" in result.stdout
    assert "child skip reason" in result.stdout
    assert "unselected test ran" not in result.stdout
