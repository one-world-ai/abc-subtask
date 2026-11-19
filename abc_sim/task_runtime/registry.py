"""Registry for Python-side task runtimes."""

from __future__ import annotations

from typing import TypeAlias

import mujoco

from abc_sim.task_runtime.base import TaskRuntime
from abc_sim.task_runtime.ball_tray import BallTrayBalancingRuntime
from abc_sim.task_runtime.conveyor_pick import ConveyorPickRuntime
from abc_sim.task_runtime.multi_drawer import MultiDrawerSearchRuntime
from abc_sim.task_runtime.real2gym_erase import Real2GymEraseRuntime
from abc_sim.task_runtime.wipe_whiteboard import WipeWhiteboardRuntime
from abc_sim.task_registry import resolve_env_task_name
from abc_sim.task_specs import SimTaskSpec


TaskRuntimeFactory: TypeAlias = type[TaskRuntime]

_TASK_RUNTIMES: dict[str, TaskRuntimeFactory] = {
    "ball_tray_balancing": BallTrayBalancingRuntime,
    "conveyor_pick": ConveyorPickRuntime,
    "multi_drawer_search": MultiDrawerSearchRuntime,
    "whiteboard": WipeWhiteboardRuntime,
    "r2g_erase_the_whiteboard": Real2GymEraseRuntime,
}

# Real2Gym variants: stage prompts (and the whiteboard's erasing) per env task
import functools as _functools  # noqa: E402

from abc_sim.real2gym_catalog import variants as _r2g_variants  # noqa: E402
from abc_sim.task_runtime.real2gym_prompts import Real2GymPromptRuntime  # noqa: E402

_TASK_RUNTIMES.update({name: _functools.partial(Real2GymPromptRuntime, v["task"], v["variant"], v["object"],
                                                members=v.get("members"), group=v.get("group"))
                       for name, v in _r2g_variants(include_legacy=True).items()})


def make_task_runtime(
    task: str | SimTaskSpec | None,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> TaskRuntime | None:
    """Instantiate and bind the task runtime for a task, if one is registered."""

    env_task = resolve_env_task_name(task)
    if env_task is None:
        return None

    runtime_cls = _TASK_RUNTIMES.get(env_task)
    if runtime_cls is None:
        return None

    runtime = runtime_cls()
    runtime.bind(model, data)
    return runtime

# Real2Gym tool envs (abc_sim/real2gym_tool_envs.py): one fixed instruction per env
from abc_sim.real2gym_tool_envs import FixedPromptRuntime as _FixedPrompt, tool_variants as _r2g_tool_variants  # noqa: E402

_TASK_RUNTIMES.update({name: _functools.partial(_FixedPrompt, v["prompt"]) for name, v in _r2g_tool_variants().items()})
