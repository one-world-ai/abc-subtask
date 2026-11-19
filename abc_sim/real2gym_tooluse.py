"""Tool use in the Real2Gym tasks whose demos move the items with tongs (pack_the_chocolate, pack_the_cookies,
place_the_bread, serve_the_lunch_box): which body is the tool, its two jaws, and the contact facts the checks need.

  held(arm)      both fingers of that arm's gripper touch the tool (a pinch of the tool, not a push)
  clamps(item)   both jaws of the tool touch the item
  fingers(item)  a gripper finger touches the item

All three are read from the contacts of one MjData (dist <= 1 mm), so they work on recorded qpos after mj_forward.
"""

from __future__ import annotations

import mujoco

# tool body, its two jaw bodies (children of the tool body), the arm that holds it in the demo, label wording
TOOLS: dict[str, dict] = {
    "pack_the_chocolate": {"tool": "tweezers", "jaws": ("tweezer_a", "tweezer_b"), "arm": "right", "phrase": "metal tongs"},
    "pack_the_cookies": {"tool": "tweezers", "jaws": ("tweezer_a", "tweezer_b"), "arm": "left", "phrase": "metal tongs"},
    "place_the_bread": {"tool": "food_tongs", "jaws": ("food_tongs_arm_-1", "food_tongs_arm_1"), "arm": "right",
                        "phrase": "metal tongs"},
    "serve_the_lunch_box": {"tool": "tongs", "jaws": ("tongs_a", "tongs_b"), "arm": "left", "phrase": "clamp"},
}


def _subtree(model: mujoco.MjModel, name: str) -> set[int]:
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if root < 0:
        raise ValueError(f"body {name!r} not found")
    bodies = {root}
    for b in range(model.nbody):
        if model.body_parentid[b] in bodies and b != 0:
            bodies.add(b)
    return {g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies}


def _touch(data: mujoco.MjData, a: set[int], b: set[int], margin: float = 1e-3) -> bool:
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist > margin:
            continue
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in a and g2 in b) or (g1 in b and g2 in a):
            return True
    return False


class ToolUse:
    def __init__(self, model: mujoco.MjModel, task: str) -> None:
        cfg = TOOLS[task]
        self.cfg, self.model = cfg, model
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cfg["tool"])
        self.geoms = _subtree(model, cfg["tool"])
        self.jaws = tuple(_subtree(model, j) for j in cfg["jaws"])
        self.hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
                       and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[j])) or "") in cfg["jaws"]]
        self.hinge_qadr = [int(model.jnt_qposadr[j]) for j in self.hinges]
        self.sides: dict[tuple[str, str], set[int]] = {}
        for g in range(model.ngeom):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
            if not (n.startswith("left_") or n.startswith("right_")):
                continue
            if not ("finger" in n or "_lf_" in n or "_rf_" in n):
                continue
            arm = "left" if n.startswith("left_") else "right"
            side = "lf" if ("_lf_" in n or "left_finger" in n) else "rf"
            self.sides.setdefault((arm, side), set()).add(g)
        self.fingers_all = set().union(*self.sides.values()) if self.sides else set()

    def held(self, data: mujoco.MjData, arm: str) -> bool:
        return _touch(data, self.geoms, self.sides.get((arm, "lf"), set())) and \
            _touch(data, self.geoms, self.sides.get((arm, "rf"), set()))

    def held_by(self, data: mujoco.MjData) -> str | None:
        for arm in ("left", "right"):
            if self.held(data, arm):
                return arm
        return None

    def clamps(self, data: mujoco.MjData, item_geoms: set[int]) -> bool:
        return _touch(data, item_geoms, self.jaws[0]) and _touch(data, item_geoms, self.jaws[1])

    def touches(self, data: mujoco.MjData, item_geoms: set[int]) -> bool:
        return _touch(data, item_geoms, self.geoms)

    def fingers(self, data: mujoco.MjData, item_geoms: set[int]) -> bool:
        return _touch(data, item_geoms, self.fingers_all)

    def opening(self, data: mujoco.MjData) -> list[float]:
        return [float(data.qpos[a]) for a in self.hinge_qadr]


__all__ = ["TOOLS", "ToolUse"]
