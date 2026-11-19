"""Prompts for the Real2Gym task variants (abc_sim/real2gym_catalog.py: __staged, __obj_<o>, __pick_<o>).

"pick" / "obj" (subtask envs): the prompt is the instruction of the spec resolved at reset from the settled layout
(abc_sim/real2gym_spec.py: arm, qualifier, object phrase, goal clause), the one the check enforces; it stays the same
for the whole episode. The spec comes from the randomization metadata (``r2g_spec``, written by the randomizer); if it
is not there it is resolved here from the reset state and written into the metadata, so the evaluator reads the same.

"staged": the prompt is one of the source episode's training-label sentences (real2gym_prompts.json): for the current
object its grasp sentence until it is held (a finger touches it and it is >= 1 cm above where it started), then its
place sentence; it walks the objects in the example's completion order, skipping objects whose goal is met, moving an
object to the back of the order when it was not grasped within ``stall_steps`` control steps, and ends with the joint
(lid / door) sentences. Re-evaluated every ``every`` physics steps (one control step = 17); abc_client re-reads the
env prompt at every inference.
The whiteboard task's ink erasing (real2gym_erase.py) runs here too.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from abc_sim.real2gym_catalog import load_prompts, object_stages, task_config
from abc_sim.real2gym_spec import fallback_prompt, resolve_spec


class Real2GymPromptRuntime:
    def __init__(self, task: str, variant: str, obj: str | None = None, every: int = 51, stall_steps: int = 600,
                 members=None, group: str | None = None) -> None:
        self.stall_steps = int(stall_steps)  # control steps on one object without grasping it before moving on
        self.task, self.variant, self.obj, self.every = task, variant, obj, int(every)
        self.members = [obj] if obj is not None else list(members or [])  # subtask envs: the class (legacy: the instance)
        self.group = group
        stages = object_stages(task)
        self.stages = [o for o in stages if obj is None or o["name"] == obj]
        self.joints = [j for j in load_prompts().get(task, {}).get("joints", []) if j.get("text") and not j.get("met_at_start")]
        self._prompt = self.stages[0]["grasp_text"] if self.stages else task_config(task)["prompt"]
        self.spec = None  # subtask envs: the spec of the current episode
        if variant in ("pick", "obj") and self.members:
            self._prompt = fallback_prompt(task, variant, self.members)
        self._eraser = None
        if task == "erase_the_whiteboard":
            from abc_sim.task_runtime.real2gym_erase import Real2GymEraseRuntime
            self._eraser = Real2GymEraseRuntime()
        self._n = 0
        self._stage = ("start", None)

    # ----------------------------------------------------------------------------------- runtime protocol
    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._eraser is not None:
            self._eraser.bind(model, data)

    def after_reset(self, model: mujoco.MjModel, data: mujoco.MjData, *, randomization: Any = None) -> None:
        from abc_sim.task_eval.real2gym_tasks import Real2GymEvaluator
        if self._eraser is not None:
            self._eraser.after_reset(model, data, randomization=randomization)
        self._n = 0
        if self.variant in ("pick", "obj") and self.members:
            meta = getattr(randomization, "metadata", None)
            spec = meta.get("r2g_spec") if isinstance(meta, dict) else None
            if not (isinstance(spec, dict) and spec.get("task") == self.task and spec.get("variant") == self.variant
                    and spec.get("object") in self.members):
                spec = resolve_spec(model, data, self.task, self.variant, self.obj, members=self.members,
                                    group=self.group, seed=getattr(randomization, "seed", None))
                if isinstance(meta, dict):
                    meta["r2g_spec"] = spec
            self.spec = spec
            self._prompt, self._stage = spec["instruction"], (self.variant, spec["object"])
            self._ready = False  # no stage machinery
            return
        self._check = Real2GymEvaluator(model=model, spec=None, task=self.task)
        self._ready = getattr(self._check, "_ready", False)
        self._rest = {}
        if self._ready:
            for o in self.stages:
                self._rest[o["name"]] = float(data.xipos[self._check._body[o["name"]]][2])
        self._n = 0
        self._order = [o["name"] for o in self.stages]
        self._since = {}  # object -> physics step its current grasp attempt started
        self._update(model, data)

    def before_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        pass

    def after_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._eraser is not None:
            self._eraser.after_step(model, data)
        self._n += 1
        if self._n % self.every == 0:
            self._update(model, data)

    def observation_prompt(self) -> str:
        return self._prompt

    def debug_state(self) -> dict[str, Any]:
        out = {"prompt": self._prompt, "stage": list(self._stage)}
        if self.spec is not None:
            out["spec"] = dict(self.spec)
        if self._eraser is not None:
            out.update(self._eraser.debug_state())
        return out

    # ----------------------------------------------------------------------------------- stages
    def _held(self, data, name) -> bool:
        from abc_sim.task_eval.openwam_tasks import _touching
        c = self._check
        lifted = float(data.xipos[c._body[name]][2]) - self._rest.get(name, 0.0) >= 0.01
        return bool(lifted and _touching(data, c._geoms[name], c._fingers))

    def _update(self, model, data) -> None:
        if not self._ready or not self.stages or self.variant in ("pick", "obj"):
            return
        met = {r["object"]: r["met"] for r in self._check.goal_detail(data)}
        by = {o["name"]: o for o in self.stages}
        todo = [n for n in self._order if not met.get(n)]
        if todo:
            n = todo[0]
            held = self._held(data, n)
            if held:
                self._since.pop(n, None)
            else:
                start = self._since.setdefault(n, self._n)
                if len(todo) > 1 and (self._n - start) >= self.stall_steps * 17:
                    # stalled on this object: try the next one, come back to it later
                    self._order.remove(n)
                    self._order.append(n)
                    self._since.pop(n, None)
                    n = [x for x in self._order if not met.get(x)][0]
                    self._since[n] = self._n
                    held = self._held(data, n)
            o = by[n]
            self._prompt = o["place_text"] if held else o["grasp_text"]
            self._stage = ("place" if held else "grasp", n)
            return
        if self.variant == "staged":
            for j in self.joints:
                g = next((g for g in self._check.cfg["goals"] if g["kind"] == "joint" and g["joint"] == j["joint"]), None)
                if g is None:
                    continue
                adr = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j["joint"])])
                if not (g["lo"] <= float(data.qpos[adr]) <= g["hi"]):
                    self._prompt, self._stage = j["text"], ("joint", j["joint"])
                    return
        self._stage = ("done", None)  # keep the last sentence


__all__ = ["Real2GymPromptRuntime"]
