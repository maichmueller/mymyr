"""Child programs retain interpreter options and the package being tested."""

import json
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).parent


@pytest.mark.parametrize("flags", [[], ["-S"], ["-I"], ["-E", "-X", "utf8=1"]])
def test_child_interpreter_configuration(flags, tmp_path):
    # a stand-in package on the parent's path: without site processing (-S) nothing else supplies a mymyr, and with
    # it an editable install may take precedence, for the parent and the child alike
    package = tmp_path / "packages" / "mymyr"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "_core.py").write_text("")
    child = tmp_path / "inspect_child.py"
    child.write_text(
        "import json, mymyr, sys\n"
        "print(json.dumps({'no_site': sys.flags.no_site, 'isolated': sys.flags.isolated, "
        "'ignore_environment': sys.flags.ignore_environment, 'xoptions': sys._xoptions, 'path': sys.path, "
        "'package': mymyr.__file__}))\n"
    )
    # an intermediate parent with the flags under test runs the child through the helper
    parent = (
        f"import json, sys; sys.path[:0] = [{str(HERE)!r}, {str(package.parent)!r}]\n"
        "from python_children import run_python\n"
        f"result = run_python({str(child)!r}, cwd={str(tmp_path)!r}, timeout=45, capture_output=True)\n"
        "import mymyr\n"
        "print(json.dumps([json.loads(result.stdout), sys.path, mymyr.__file__]))\n"
    )
    result = subprocess.run([sys.executable, *flags, "-c", parent], check=True, capture_output=True, text=True)
    observed, parent_path, parent_package = json.loads(result.stdout)
    assert observed["no_site"] == int("-S" in flags)
    assert observed["isolated"] == int("-I" in flags)
    assert observed["ignore_environment"] == int("-I" in flags or "-E" in flags)
    if "-X" in flags:
        assert observed["xoptions"]["utf8"] == "1"
    assert [pathlib.Path(path or ".").resolve() for path in observed["path"]] == [
        pathlib.Path(path or ".").resolve() for path in parent_path
    ]
    assert observed["package"] == parent_package
