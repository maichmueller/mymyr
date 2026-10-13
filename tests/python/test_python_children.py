"""Child programs retain interpreter options and the extension being tested."""

import json
import pathlib
import subprocess
import sys

import pytest
from python_children import module_bootstrap

HELPER = pathlib.Path(__file__).with_name("python_children.py")


@pytest.mark.parametrize("flags", [[], ["-S"], ["-I"], ["-E", "-X", "utf8=1"]])
def test_child_interpreter_configuration(flags, tmp_path):
    child = tmp_path / "inspect_child.py"
    child.write_text(
        "import json, sys\n"
        "print(json.dumps({'no_site': sys.flags.no_site, 'isolated': sys.flags.isolated, "
        "'ignore_environment': sys.flags.ignore_environment, 'xoptions': sys._xoptions, 'path': sys.path}))\n"
    )
    bootstrap = module_bootstrap() + (
        f"helper = runpy.run_path({str(HELPER)!r})\n"
        f"result = helper['run_python']({str(child)!r}, cwd={str(tmp_path)!r}, timeout=45, capture_output=True)\n"
        "print(result.stdout, end='')\n"
    )
    result = subprocess.run([sys.executable, *flags, "-c", bootstrap], check=True, capture_output=True, text=True)
    observed = json.loads(result.stdout)
    assert observed["no_site"] == int("-S" in flags)
    assert observed["isolated"] == int("-I" in flags)
    assert observed["ignore_environment"] == int("-I" in flags or "-E" in flags)
    if "-X" in flags:
        assert observed["xoptions"]["utf8"] == "1"
    assert observed["path"] == [str(pathlib.Path(path or ".").resolve()) for path in sys.path]
