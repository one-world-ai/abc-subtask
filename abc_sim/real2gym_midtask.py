"""Staged mid-task starts for Real2Gym objects that are locked where they start (fix-unlock).

Some goal objects cannot be grasped and moved alone at reset: they lie under another object (bowl_4 under the yellow
floral bowl, nested baskets), stand between neighbours (inner rack plates) or sit behind a closed door (the shoes).
In the source demo they are handled after the objects in the way were moved to their goals and the door was opened.
A mid-task start reproduces that point of the demo: after the normal reset (layout jitter + settle) the target's
predecessors (the objects the demo handles before it) are placed at the poses the demo had them in when the target's
stage starts, relative to their goal container (or, for goals without a container, relative to the target's own
jittered cluster), joints the demo moved before then (a door) are set to the demo's value, and the scene settles again.
The target and every object the demo handles later stay where the jittered layout put them. Deterministic in the seed
(no random draws: the staged poses follow from the layout of the seed). Every reference the evaluator takes at reset
(start heights, off-goal positions) is taken after staging; the staged objects are not goal objects of the env.

Env ids: the per-object env name + ``__midtask``:  r2g_<task>__pick_<object>__midtask, r2g_<task>__obj_<object>__midtask
(env = abc_sim.make_env(task=...); env.reset(seed=s)). The reset metadata ``r2g_midtask`` lists what was staged and how.
Data: real2gym_midtask.json (tools/real2gym_midtask_build.py, from the release demos). Physics: the randomizer
randomization/tasks/real2gym_midtask.py. Validation: tools/real2gym_midtask_check.py.

Only json / functools / inspect at import time (real2gym_catalog wraps its variants() with ``with_midtask``).
"""
from __future__ import annotations

import functools
import inspect
import json
from pathlib import Path
from typing import Any

MIDTASK_FILE = Path(__file__).resolve().parent / "real2gym_midtask.json"
SUFFIX = "__midtask"
_CACHE: dict[str, Any] = {}


def load_midtask() -> dict[str, dict[str, Any]]:
    """{task: {target: {frame, why_locked, staged: [{name, goal, pos, quat, anchor, anchor_pos, anchor_quat, ...}],
    joints: {joint: value}}}}; {} if the data file is missing."""
    if "data" not in _CACHE:
        try:
            _CACHE["data"] = json.loads(MIDTASK_FILE.read_text())["tasks"]
        except (OSError, ValueError, KeyError):
            _CACHE["data"] = {}
    return _CACHE["data"]


def midtask_record(task: str, target: str) -> dict[str, Any] | None:
    return load_midtask().get(task, {}).get(target)


def is_midtask(env_task: str) -> bool:
    return env_task.endswith(SUFFIX)


def midtask_variants(base: dict[str, dict[str, Any]], prefix: str = "r2g_", sep: str = "__") -> dict[str, dict[str, Any]]:
    """Mid-task env entries derived from the per-object entries ``base`` (env name -> variants() entry): for each
    target of real2gym_midtask.json and kind pick / obj, a copy of r2g_<task>__<kind>_<target> under that name + SUFFIX
    with the instance fixed (object = members = [target]), ``midtask`` = what is staged, offered (the start reason the
    phrase table gives is what the staging removes)."""
    out = {}
    for task, targets in load_midtask().items():
        for target, rec in targets.items():
            for kind in rec.get("kinds", ("pick", "obj")):  # pick only where the start blocks the grasp, not just the goal
                name = f"{prefix}{task}{sep}{kind}_{target}"
                v = base.get(name)
                if v is None:
                    continue
                v = dict(v)
                staged = [s["name"] for s in rec["staged"]]
                what = ", ".join(staged) if staged else "nothing"
                joints = ", ".join(f"{j} = {q:.2f}" for j, q in rec["joints"].items())
                v.update(object=target, members=[target], legacy=False, offered=True, not_offered_reason="",
                         midtask={"target": target, "frame": rec["frame"], "staged": staged, "joints": dict(rec["joints"]),
                                  "why_locked": rec["why_locked"]},
                         title=(v.get("title") or name) + f" [mid-task start: {what} at goal" + (f"; {joints}" if joints else "") + "]")
                v.pop("class_env", None)
                out[name + SUFFIX] = v
    return out


def with_midtask(variants_fn):
    """Wrap real2gym_catalog.variants so its listing includes the mid-task envs (every registry reads it)."""
    legacy_kw = "include_legacy" in inspect.signature(variants_fn).parameters

    @functools.wraps(variants_fn)
    def variants(*args, **kwargs):
        out = variants_fn(*args, **kwargs)
        full = variants_fn(include_legacy=True) if legacy_kw else out
        out.update(midtask_variants(full))
        return out
    variants.__wrapped_midtask__ = True
    return variants


__all__ = ["MIDTASK_FILE", "SUFFIX", "is_midtask", "load_midtask", "midtask_record", "midtask_variants", "with_midtask"]
