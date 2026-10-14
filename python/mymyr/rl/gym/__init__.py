"""Gymnasium environments over mymyr tasks (needs gymnasium 1.1 or later: ``pip install 'mymyr[gym]'``).

- :class:`Env`: one environment (a ``gymnasium.Env``) over a Task, a ``mymyr.rl.TaskTable``, a
  ``mymyr.rl.TaskSuite`` or a PDDL domain and its problems, with pddlgym-style symbolic observations (sets of mymyr
  GroundAtoms, :class:`SymbolicSpace`), relational graphs of ``mymyr.encoding`` or atom bits, and actions as an index
  into the state's applicable actions (``Discrete`` with ``info["action_mask"]``, the default) or as ground actions
  (:class:`GroundActionSpace`)::

      env = mymyr.rl.gym.Env(domain="blocks/domain.pddl", problems="blocks/", max_steps=100)
      obs, info = env.reset(seed=0)
      obs.atoms                                   # frozenset of GroundAtoms; obs.goal: the GroundCondition
      action = env.action_space.sample(mask=info["action_mask"])
      obs, reward, terminated, truncated, info = env.step(action)

- :class:`VectorEnv`: N environments as a ``gymnasium.vector.VectorEnv`` stepped by one native
  ``mymyr.rl.CpuEnvPool``, with Gymnasium's autoreset modes.
- Registration: importing this module registers ``mymyr/PDDL-v0`` (``gymnasium.make("mymyr/PDDL-v0",
  domain=..., problems=...)``; ``gymnasium.make_vec`` builds the VectorEnv); :func:`register_pddl` registers one
  domain's problems as ``mymyr/<name>-v0``.

Transitions follow the RL contract of ``mymyr.rl`` exactly (they are the CpuEnvPool's steps): rewards, terminations,
truncation, absorbing goals and dead ends, solved resets, instance sampling.
"""

try:
    import gymnasium
    from gymnasium.vector import AutoresetMode as _AutoresetMode  # noqa: F401 (gymnasium 1.1 or later)
except ImportError as error:
    raise ImportError(
        "mymyr.rl.gym needs gymnasium 1.1 or later, which the extra 'gym' installs: pip install 'mymyr[gym]'"
    ) from error

import os
from typing import Any, Optional

from ._core import GroundActionSpace, SymbolicObservation, SymbolicSpace, _problem_arg
from ._env import Env
from ._vector import VectorEnv

__all__ = ["Env", "GroundActionSpace", "SymbolicObservation", "SymbolicSpace", "VectorEnv", "register_pddl"]

_ENTRY = "mymyr.rl.gym:Env"
_VECTOR_ENTRY = "mymyr.rl.gym:VectorEnv"
PDDL_ENV_ID = "mymyr/PDDL-v0"


def register_pddl(
    domain: Any, problems: Any, *, name: Optional[str] = None, namespace: str = "mymyr", version: int = 0,
    max_episode_steps: Optional[int] = None, **kwargs: Any,
) -> str:  # fmt: skip
    """Registers a PDDL domain and its problems with Gymnasium and returns the id ``<namespace>/<name>-v<version>``.

    ``problems`` is a problem file, a directory of them (sorted by path), a glob or a list, as
    ``TaskTable.from_pddl`` takes it; ``name`` defaults to the domain file's name without ``.pddl``, or its
    directory's name for a file named ``domain.pddl``. ``kwargs`` are :class:`Env` / :class:`VectorEnv` keywords
    (``observation``, ``action``, ``max_steps``, ...). ``gymnasium.make(id)`` then builds an :class:`Env`,
    ``gymnasium.make_vec(id, num_envs=N)`` a :class:`VectorEnv`; ``max_episode_steps`` adds Gymnasium's TimeLimit
    (``max_steps=`` truncates natively instead)."""
    domain = os.fspath(domain)
    if name is None:
        stem = os.path.splitext(os.path.basename(domain))[0]
        name = os.path.basename(os.path.dirname(os.path.abspath(domain))) if stem == "domain" else stem
    env_id = f"{namespace}/{name}-v{int(version)}"
    gymnasium.register(
        env_id, entry_point=_ENTRY, vector_entry_point=_VECTOR_ENTRY, max_episode_steps=max_episode_steps,
        kwargs={"domain": domain, "problems": _problem_arg(problems), **kwargs},
    )  # fmt: skip
    return env_id


if PDDL_ENV_ID not in gymnasium.registry:
    gymnasium.register(PDDL_ENV_ID, entry_point=_ENTRY, vector_entry_point=_VECTOR_ENTRY)
