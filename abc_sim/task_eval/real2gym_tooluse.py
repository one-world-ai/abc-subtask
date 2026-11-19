"""Checks of the Real2Gym tool envs (abc_sim/real2gym_tool_envs.py): pick up the tool; move an item with the tool.

tool_pick  success: both fingers of the designated arm's gripper touch the tool, the tool's centre of mass is >= LIFT_M
           above its reset height and it touches nothing but the robot, continuously for HOLD_S. Progress 0.3 (pinched)
           + 0.7 x lift fraction while pinched. failed: the tool left the table or was pushed > 8 cm while not pinched.
tool_use   transport (latched): the item clamped by both tool jaws, the tool pinched by the designated arm, no finger on
           the item, the item >= CARRY_M above its reset height and touching nothing but the tool, continuously for
           CARRY_S. Success: transport happened AND now the item meets its catalog goal, touches neither the tool nor a
           finger and is at rest, continuously for HOLD_S. An item moved into the goal by hand or pushed there never
           pays (an item a finger ever touched is out). The env names a class of look-alikes ("a round pink cookie"):
           any member counts (``members``). Progress: 0.2 tool pinched, 0.5 once transported, 0.9 transported and goal
           met (not yet released / held). failed: an item left the table, or fingers touched every member.
Time is the sim time (data.time); every reference (reset heights) is taken at the first evaluation after reset.
"""
from __future__ import annotations

import mujoco
import numpy as np

from abc_sim.real2gym_catalog import task_config
from abc_sim.real2gym_tool_envs import TOOL_ENVS, tool_members
from abc_sim.real2gym_tooluse import ToolUse, _subtree
from abc_sim.task_eval.real2gym_tasks import Real2GymEvaluator

LIFT_M, CARRY_M = 0.03, 0.02
HOLD_S, CARRY_S = 0.5, 0.3


class Real2GymToolEvaluator(Real2GymEvaluator):
    def __init__(self, *, model, spec, task: str, variant: str, object: str | None = None, members=(), **kw) -> None:
        self.tool_variant = variant
        self.tool_cfg = TOOL_ENVS[task]
        # tool_use: the item and its look-alikes (the instruction names the class with "a"): any of them may be moved
        self._members = (list(members) or tool_members(task, object)) if variant == "tool_use" else []
        # the item's own goal (variant "obj" of the parent: that goal only, no constraints)
        super().__init__(model=model, spec=spec, task=task, variant="obj" if variant == "tool_use" else None,
                         object=object, **kw)

    def reset(self, *, nworld: int = 1) -> None:
        super().reset(nworld=nworld)
        self._t_ref = None
        self._since = {}
        self._transported: set[str] = set()
        self._finger_touched: set[str] = set()

    def _bind(self, model: mujoco.MjModel) -> None:
        super()._bind(model)
        if not getattr(self, "_ready", False):
            return
        self._tu = ToolUse(model, self.task)
        self._robot = set().union(*self._tu.sides.values())
        for g in range(model.ngeom):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_rootid[int(model.geom_bodyid[g])])) or ""
            if n.startswith(("left_", "right_")) or "arm" in n or "station" in n:
                self._robot.add(g)
        self._mgeoms = {m: _subtree(model, m) for m in self._members}
        self._goal_of = {}
        for m in self._members:
            for g in task_config(self.task)["goals"]:
                if m in g.get("objects", []) and g["kind"] == "in":
                    self._goal_of[m] = g

    def _only_touches(self, data, geoms: set[int], allowed: set[int]) -> bool:
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist > 1e-3:
                continue
            a, b = int(c.geom1), int(c.geom2)
            if (a in geoms) == (b in geoms):
                continue
            if (b if a in geoms else a) not in allowed:
                return False
        return True

    def _lasting(self, key: str, cond: bool, t: float, need: float) -> bool:
        if not cond:
            self._since.pop(key, None)
            return False
        t0 = self._since.setdefault(key, t)
        return t - t0 >= need - 1e-9

    def _score_inner(self, data, *, with_velocity):
        tu, arm, t = self._tu, self.tool_cfg["arm"], float(data.time)
        tool_b = tu.body
        if self._t_ref is None:
            self._t_ref = {"tool": data.xipos[tool_b].copy()}
            for m in self._members:
                self._t_ref[m] = data.xipos[self._body[m]].copy()
        held = tu.held(data, arm)
        if self.tool_variant == "tool_pick":
            lift = float(data.xipos[tool_b][2] - self._t_ref["tool"][2])
            free = self._only_touches(data, tu.geoms, self._robot)
            ok = self._lasting("pick", held and lift >= LIFT_M and free, t, HOLD_S)
            moved = float(np.linalg.norm(data.xipos[tool_b][:2] - self._t_ref["tool"][:2]))
            failed = bool(data.xipos[tool_b][2] < 0.70 or (moved > 0.08 and not held))
            prog = (0.3 + 0.7 * float(np.clip(lift / LIFT_M, 0.0, 1.0))) if held else 0.0
            return (1.0 if ok else min(prog, 0.99)), ok, {"items_done": float(ok), "items": 1, "tool_held": held,
                                                          "tool_lift_m": lift, "lifted_free": free, "failed": failed,
                                                          "held": held, "at_rest": True, "constraints_ok": True}
        best, best_prog, out = False, 0.2 * held, {}
        for m in self._members:
            ig, b = self._mgeoms[m], self._body[m]
            lift = float(data.xipos[b][2] - self._t_ref[m][2])
            clamped = tu.clamps(data, ig)
            finger = tu.fingers(data, ig)
            if finger:
                self._finger_touched.add(m)
            carried = held and clamped and not finger and lift >= CARRY_M and self._only_touches(data, ig, tu.geoms)
            if self._lasting("carry:" + m, carried, t, CARRY_S):
                self._transported.add(m)
            goal_met = bool(self._goal_in(data, dict(self._goal_of[m], objects=[m]))[0])
            jid = self._objs[m]["free_joint"]
            dof = int(self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)])
            speed = float(np.linalg.norm(data.qvel[dof: dof + 3])) if with_velocity else 0.0
            released = not tu.touches(data, ig) and not finger
            done = (m in self._transported and m not in self._finger_touched and goal_met and released
                    and speed <= self.rest_speed)
            if self._lasting("done:" + m, done, t, HOLD_S):
                best = True
            if m in self._transported and m not in self._finger_touched:
                best_prog = max(best_prog, 0.9 if goal_met else 0.5)
            out[m] = {"clamped": clamped, "finger": finger, "goal_met": goal_met}
        failed = bool(any(data.xipos[self._body[m]][2] < 0.70 for m in self._members)
                      or (self._finger_touched and self._finger_touched >= set(self._members)))
        return (1.0 if best else min(best_prog, 0.99)), best, {
            "items_done": float(best), "items": 1, "tool_held": held, "transported": len(self._transported),
            "finger_touched": len(self._finger_touched), "clamped": any(v["clamped"] for v in out.values()),
            "goal_met": any(v["goal_met"] for v in out.values()), "failed": failed, "held": held, "at_rest": True,
            "constraints_ok": True}


def with_tool_evaluator(base_factory):
    """The "real2gym" evaluator factory, dispatching the tool variants to Real2GymToolEvaluator."""
    def factory(*, model, spec, **kw):
        if kw.get("variant") in ("tool_pick", "tool_use"):
            return Real2GymToolEvaluator(model=model, spec=spec, **kw)
        return base_factory(model=model, spec=spec, **kw)
    return factory


__all__ = ["Real2GymToolEvaluator", "with_tool_evaluator"]
