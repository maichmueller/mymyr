"""A real policy update through native encoding and multi-instance planning dynamics."""

import runpy

import pytest

import mymyr
from conftest import ROOT

pytestmark = pytest.mark.skipif(not hasattr(mymyr, "Domain"), reason="built without the loki front end")


def test_small_training_beats_random_on_larger_held_out_instances():
    example = runpy.run_path(str(ROOT / "examples/rl_learning.py"))
    report = example["run"](training_episodes=400, evaluation_episodes=32, seed=7)
    assert all(visits > 0 for visits in report["training_instance_visits"])
    assert min(report["held_out_objects"]) > max(report["training_objects"])
    assert report["learned_greedy"]["success_rate"] == 1.0
    assert report["learned_greedy"]["success_rate"] > report["random"]["success_rate"]
    assert report["learned_greedy"]["truncations"] == 0
