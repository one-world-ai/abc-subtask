"""Subtask catalogue, discovered from the task modules themselves.

Every ``abc_sim/task_eval/<short>_subtask.py`` defines one or more ``SubtaskEvaluator`` subclasses that set ``TASK``
(the parent task, e.g. ``put_plastic_bottles_in_bin``), ``ENV_TASK`` (its scene) and ``NAME`` (the evaluator key).
Each key of the class's ``STAGE_PROMPTS`` becomes one task spec ``<TASK>__<stage>`` with that prompt.

Adding a task therefore touches only its own modules: no shared registry or spec list is edited, so per-task
branches merge without conflicts. Discovery runs once, on first use (``task_specs`` and ``registry`` call it
lazily), and fails loudly on a duplicate evaluator name, spec name or prompt.
"""

from __future__ import annotations

import functools
import importlib
import pkgutil

from abc_sim.task_eval.stagekit.subtask import SubtaskEvaluator
from abc_sim.task_specs import SimTaskSpec

MODULE_SUFFIX = "_subtask"


@functools.cache
def subtask_evaluators() -> dict[str, type[SubtaskEvaluator]]:
    """``NAME`` → evaluator class for every subtask evaluator in ``abc_sim.task_eval``."""
    import abc_sim.task_eval as pkg

    found: dict[str, type[SubtaskEvaluator]] = {}
    for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda i: i.name):
        if not info.name.endswith(MODULE_SUFFIX):
            continue
        module = importlib.import_module(f"{pkg.__name__}.{info.name}")
        for obj in vars(module).values():
            if not (isinstance(obj, type) and issubclass(obj, SubtaskEvaluator) and obj.__module__ == module.__name__
                    and obj.TASK):
                continue
            missing = [a for a in ("ENV_TASK", "NAME") if not getattr(obj, a)]
            if missing:
                raise TypeError(f"{obj.__qualname__} sets TASK but not {', '.join(missing)}")
            if obj.NAME in found and found[obj.NAME] is not obj:
                raise ValueError(f"evaluator name {obj.NAME!r} used by {found[obj.NAME].__qualname__} and "
                                 f"{obj.__qualname__}")
            found[obj.NAME] = obj
    return found


def specs_for(cls: type[SubtaskEvaluator]) -> tuple[SimTaskSpec, ...]:
    """One spec per stage of ``cls`` (in ``STAGE_PROMPTS`` order)."""
    module = cls.__module__.rsplit(".", 1)[-1]
    return tuple(
        SimTaskSpec(
            name=f"{cls.TASK}__{stage}",
            env_task=cls.ENV_TASK,
            prompt=prompt,
            description=f"{cls.TASK} subtask '{stage}' for single-stage RL: start with env.reset_to_stage "
                        f"({module}.py).",
            evaluator_name=cls.NAME,
            evaluator_options=(("stage", stage),),
        )
        for stage, prompt in cls.STAGE_PROMPTS.items()
    )


@functools.cache
def subtask_specs() -> tuple[SimTaskSpec, ...]:
    """Specs of every discovered subtask stage."""
    specs = tuple(s for cls in subtask_evaluators().values() for s in specs_for(cls))
    names = [s.name for s in specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate subtask spec names: {dupes}")
    return specs


def stage_spec(task: str, stage: str) -> SimTaskSpec:
    """The spec of ``<task>__<stage>``."""
    for spec in subtask_specs():
        if spec.name == f"{task}__{stage}":
            return spec
    stages = [s.name.split("__", 1)[1] for s in subtask_specs() if s.name.startswith(f"{task}__")]
    raise KeyError(f"no subtask {task}__{stage}; {task} has stages {stages}")
