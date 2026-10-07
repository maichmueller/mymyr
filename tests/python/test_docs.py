"""Execute runnable Python examples from the user documentation."""

import json
import pathlib
import re
import subprocess
import sys

import pytest

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
    has_cuda = _cuda_available()
    runnable, skipped = [], 0
    for options, source in blocks:
        if "skip-if-no-cuda" in options and not has_cuda:
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
        subprocess.run([sys.executable, "-W", "error", str(script)], cwd=ROOT, check=True,
                       timeout=None if has_cuda else 45)

    print(f"docs {page.name}: {len(runnable)} blocks run, {skipped} skipped")
