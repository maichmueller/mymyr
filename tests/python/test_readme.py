"""Run the README's Python example without repository task files."""

import pathlib
import re

import mymyr
import pytest
from python_children import run_python

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_readme_example(tmp_path):
    if not hasattr(mymyr, "Domain"):
        pytest.skip("built without the PDDL front end")
    blocks = re.findall(r"^```python\n(.*?)^```", (ROOT / "README.md").read_text(), re.MULTILINE | re.DOTALL)
    assert blocks, "README has no Python example"
    script = tmp_path / "example.py"
    script.write_text("\n".join(blocks))
    run_python(script, cwd=tmp_path)
