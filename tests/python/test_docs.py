"""Execute runnable Python examples from the user documentation."""

import importlib.util
import json
import pathlib
import re
import mymyr
import pytest
from python_children import requires_frontend, run_python

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
_FENCE = re.compile(r"^```([^\n]*)\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)


def _cuda_available():
    try:
        import mymyr.cuda as cuda
    except ImportError:
        return False
    return cuda.available()


def python_blocks(path):
    blocks = []
    for match in _FENCE.finditer(path.read_text()):
        info = match.group(1).split()
        if info and info[0] == "python":
            blocks.append((set(info[1:]), match.group(2)))
    return blocks


PAGES = [path for path in sorted(DOCS.glob("*.md")) if python_blocks(path)]


@pytest.mark.parametrize("page", PAGES, ids=lambda path: path.stem)
def test_documentation_python_blocks(page, tmp_path):
    blocks = python_blocks(page)
    if not hasattr(mymyr, "Domain") and any(requires_frontend(source) for _, source in blocks):
        pytest.skip("built without the PDDL front end")
    has_cuda = _cuda_available()
    runnable, skipped = [], 0
    for options, source in blocks:
        # "needs-<module>": the block imports an optional package
        missing = [o for o in options if o.startswith("needs-") and importlib.util.find_spec(o[6:]) is None]
        if ("skip-if-no-cuda" in options and not has_cuda) or missing:
            skipped += 1
        else:
            runnable.append(source)

    if runnable:
        statements = [
            f"exec(compile({json.dumps(source)}, {json.dumps(f'{page.name}:block-{i + 1}')}, 'exec'), namespace)"
            for i, source in enumerate(runnable)
        ]
        script = tmp_path / f"{page.stem}.py"
        script.write_text("namespace = {'__name__': '__main__'}\n" + "\n".join(statements) + "\n")
        run_python(script, cwd=ROOT, timeout=None if has_cuda else 45)

    print(f"docs {page.name}: {len(runnable)} blocks run, {skipped} skipped")
