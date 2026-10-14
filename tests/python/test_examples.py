"""Run the user-facing example scripts as isolated Python programs."""

import importlib.util
import pathlib
import mymyr
import pytest
from python_children import requires_frontend, run_python

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
REQUIREMENTS = {
    "rl_jax_cuda.py": ("cuda", "jax"),
    "rl_torch_cuda.py": ("cuda", "torch"),
    "cuda_brfs.py": ("cuda",),
    "rl_gym.py": ("gymnasium",),
}
SCRIPTS = sorted(EXAMPLES.glob("*.py"))


def _cuda_available():
    try:
        import mymyr.cuda as cuda
    except ImportError:
        return False
    return cuda.available()


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
def test_example_script(script):
    if not hasattr(mymyr, "Domain") and requires_frontend(script.read_text()):
        pytest.skip("built without the PDDL front end")
    requirements = REQUIREMENTS.get(script.name, ())
    missing = [name for name in requirements if name != "cuda" and importlib.util.find_spec(name) is None]
    if missing:
        print(f"example {script.name}: skipped (missing {', '.join(missing)})")
        pytest.skip(f"needs optional package(s): {', '.join(missing)}")
    if "cuda" in requirements and not _cuda_available():
        print(f"example {script.name}: skipped (no visible CUDA device or CUDA-enabled build)")
        pytest.skip("needs a visible CUDA device and a CUDA-enabled mymyr build")

    timeout = None if "cuda" in requirements else 180 if requirements else 45
    run_python(script, cwd=ROOT, timeout=timeout)
    print(f"example {script.name}: ran")
