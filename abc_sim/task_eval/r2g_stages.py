"""Physics-defined stages of the Real2Gym "put it in a container" tasks: reach, pick, carry, place.

One module for every Real2Gym task whose goal kind is ``in`` (abc_sim/real2gym_catalog.py): the per-task differences
(objects, containers, box margins, tilt range) are catalog data. All measurements come from the task's own
``Real2GymEvaluator`` (task_eval/real2gym_tasks.py): pad contacts per arm, "touches nothing but the gripper", the
container box and the ``in`` goal. Stage table and thresholds: subtask_rewards/r2g/stage_table.md.

Per-object arrays (``R2GScene.observe``); ``arm`` is the arm an object must be handled with (-1: either; the
episode's target gets the arm its instruction names):
  pin (m, 2)   both pads of that arm on it          pad (m, 2)  a pad of that arm on it
  dist (m, 2)  pinch point of that arm -> centre of mass
  lift         centre of mass above its height at the scene's start
  blocked      touches something other than the gripper(s) of its arm (table, container, robot, another object)
  gap          horizontal distance from the centre of mass to its container's footprint (0 inside)
  floor_h      centre of mass above the container floor (container frame)
  in_goal      the catalog's ``in`` goal holds;  resting: touches something that is not a finger
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from abc_sim.real2gym_catalog import task_config
from abc_sim.task_eval.real2gym_tasks import Real2GymEvaluator
from abc_sim.task_eval.stagekit.robot import ARMS
from abc_sim.task_eval.stagekit.scene import ObjectScene
from abc_sim.task_eval.stagekit.tracker import StageTracker

STAGES = ("reach", "pick", "carry", "place", "done")
R, P, C, PL, DONE = range(5)
FALLEN = len(STAGES)
LABELS = STAGES + ("fallen",)
# Generic prompts (unique lookup keys, they name no object): the episode's instruction is rendered from the spec.
STAGE_PROMPTS = {
    "reach": "container subtask: move the named arm to the target object",
    "pick": "container subtask: grasp the target object and lift it",
    "carry": "container subtask: carry the target object over its container",
    "place": "container subtask: put the target object in its container and release it",
}
# Finger-pad centre in the grasp-site frame (openwam-sim-eval abc_tasks/tools/scripted_rollouts.py: measured).
PINCH = np.array([0.044, 0.0, -0.005])


@dataclass(frozen=True)
class StageConfig:
    reach_m: float = 0.05           # pinch point to centre of mass (stage table: the site is ~5 cm off at a grasp)
    reach_exit_m: float = 0.07
    lift_m: float = 0.03            # Real2GymEvaluator.lift_m
    pinch_s: float = 0.2            # Real2GymEvaluator.pinch_s: pinch time the final check needs
    rest_speed: float = 0.05        # Real2GymEvaluator.rest_speed
    push_m: float = 0.08            # Real2GymEvaluator pick check: pushed away without a grasp
    table_z: float = 0.70           # Real2GymEvaluator pick check: below this the object is off the table
    over_margin_m: float = 0.02     # centre of mass this far inside the container's footprint
    over_exit_m: float = 0.02       # ... and it stays "over" until this far outside the margin line
    floor_slack_m: float = 0.01
    container_moved_m: float = 0.05     # bottles' values
    container_tilt_rad: float = float(np.deg2rad(20))
    hold: int = 5
    eps: dict = field(default_factory=lambda: dict(reach=0.005, pick=0.05, carry=0.005, place=0.005))
    span: dict = field(default_factory=lambda: dict(reach=0.45, pick=1.0, carry=0.40, place=0.20))


def stage_objects(task: str) -> tuple[list[str], dict[str, dict]]:
    """Goal objects of the task's ``in`` goals (catalog order) and each one's goal."""
    names, goal_of = [], {}
    for g in task_config(task)["goals"]:
        if g["kind"] != "in":
            continue
        for n in g.get("objects", []):
            if n not in goal_of:
                names.append(n)
                goal_of[n] = g
    return names, goal_of


class R2GScene(ObjectScene):
    """The task's ``in``-goal objects that have a free joint in this model, and their containers."""

    def __init__(self, model: mujoco.MjModel, task: str, cfg: StageConfig | None = None) -> None:
        self.task, self.cfg = task, cfg or StageConfig()
        self.full = Real2GymEvaluator(model=model, spec=None, task=task)  # geometry, contacts and goals
        f = self.full
        self.ready = bool(getattr(f, "_ready", False))
        if not self.ready:  # make_env builds the evaluator before the task's scene is loaded
            self.model, self.names, self.joints, self.bodies = model, [], [], []
            self.qpos_adr = self.dof_adr = np.zeros(0, np.int64)
            self.containers, self.goal_of = [], {}
            self.arm = np.zeros(0, np.int64)
            return
        names, self.goal_of = stage_objects(task)
        objects = []
        for n in names:
            fj = f._objs[n]["free_joint"]
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fj) if fj else -1
            if j >= 0:
                objects.append((n, j))
        super().__init__(model, objects)
        self.bodies = [f._body[n] for n in self.names]  # the library's body (its inertial frame is the reference)
        self.containers = sorted({c for n in self.names for c in self.goal_of[n]["containers"]})
        self.cont_body = [f._body[c] for c in self.containers]
        self.arm = np.full(len(self.names), -1, dtype=np.int64)
        self.inner = None  # the episode's obj-variant evaluator (final check, pinch time); set by the subtask
        self.target = -1

    # rest: one row [x, y, z, zx, zy, zz] per object (centre of mass, body z axis), then per container (origin, z
    # axis), then one row whose first entry is the target's pinch time so far (it is not part of the sim state).
    def heights(self, data: mujoco.MjData) -> np.ndarray:
        rows = [np.concatenate([data.xipos[b], data.xmat[b].reshape(3, 3)[:, 2]]) for b in self.bodies]
        rows += [np.concatenate([data.xpos[b], data.xmat[b].reshape(3, 3)[:, 2]]) for b in self.cont_body]
        return np.vstack(rows + [np.zeros(6)]).astype(np.float64)

    def pinch_points(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray([data.site_xpos[s] + data.site_xmat[s].reshape(3, 3) @ PINCH for s in self.sites])

    def _box(self, data, k: int) -> tuple[float, float]:
        """(horizontal gap to the footprint shrunk by over_margin_m, height above the floor) for the nearest
        container of object k's goal."""
        f, n = self.full, self.names[k]
        best = (np.inf, 0.0)
        mg = self.cfg.over_margin_m
        for c in self.goal_of[n]["containers"]:
            ext = f._objs[c]["extent"]
            p = f._local(data, c, n)
            dx = max(ext[0] + mg - p[0], 0.0, p[0] - (ext[1] - mg))
            dy = max(ext[2] + mg - p[1], 0.0, p[1] - (ext[3] - mg))
            gap = float(np.hypot(dx, dy))
            if gap < best[0]:
                best = (gap, float(p[2] - ext[4]))
        return best

    def observe(self, data: mujoco.MjData, rest: np.ndarray) -> dict[str, np.ndarray]:
        f, m = self.full, len(self.names)
        rest = np.asarray(rest, np.float64).reshape(-1, 6)
        touch = f._pad_touch(data)
        pts = self.pinch_points(data)
        com = np.asarray([data.xipos[b] for b in self.bodies], np.float64).reshape(m, 3)
        pin, pad = np.zeros((m, 2), bool), np.zeros((m, 2), bool)
        blocked, resting = np.zeros(m, bool), np.zeros(m, bool)
        gap, floor_h, speed = np.zeros(m), np.zeros(m), np.zeros(m)
        for k, n in enumerate(self.names):
            for a, arm in enumerate(ARMS):
                pin[k, a] = f._pin(touch, arm, n)
                pad[k, a] = n in touch[(arm, "lf")] or n in touch[(arm, "rf")]
            arms = (ARMS[self.arm[k]],) if self.arm[k] >= 0 else ARMS
            blocked[k] = f._blocked(data, n, touch, arms)
            resting[k] = f._resting(data, n)
            gap[k], floor_h[k] = self._box(data, k)
            d = int(self.dof_adr[k])
            speed[k] = float(np.linalg.norm(data.qvel[d:d + 3]))
        in_goal = np.zeros(m, bool)
        index = {n: k for k, n in enumerate(self.names)}
        for g in task_config(self.task)["goals"]:
            if g["kind"] == "in":
                for n, ok in zip(g["objects"], f._goal_in(data, g)):
                    if n in index:
                        in_goal[index[n]] = bool(ok)
        # the target's final condition is the obj env's own check (placed, released, at rest, after a real grasp)
        done = in_goal & ~pad.any(axis=1) & (speed <= self.cfg.rest_speed)
        pinch_s, stock = 0.0, {}
        if self.inner is not None and 0 <= self.target < m:
            res = self.inner.evaluate_model_data(self.model, data)
            stock = {key: (v[0] if isinstance(v, np.ndarray) and v.shape else v) for key, v in res.metrics.items()}
            stock["success"] = bool(res.success[0])
            pinch_s = float(stock.get("pinch_s", 0.0))
            done[self.target] = bool(float(stock.get("items_done", 0.0)) > 0 and not stock.get("held")
                                     and stock.get("at_rest") and not stock.get("sim_fault"))
        moved = tilt = 0.0
        for i, b in enumerate(self.cont_body):
            r0 = rest[m + i]
            moved = max(moved, float(np.linalg.norm(data.xpos[b][:2] - r0[:2])))
            z = data.xmat[b].reshape(3, 3)[:, 2]
            tilt = max(tilt, float(np.arccos(np.clip(z @ r0[3:6], -1.0, 1.0))))
        dist = np.linalg.norm(com[:, None, :] - pts[None, :, :], axis=-1)
        return dict(com=com, xy=com[:, :2], lift=com[:, 2] - rest[:m, 2], pin=pin, pad=pad, dist=dist,
                    blocked=blocked, resting=resting, gap=gap, floor_h=floor_h, speed=speed, in_goal=in_goal,
                    done=done, arm=self.arm.copy(), touched=pad.any(axis=1), pinch_s=pinch_s, stock=stock,
                    container_moved=moved, container_tilt=tilt)


def by_arm(obs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pin, pad, dist) per object for the arm it must be handled with (either arm where none is required)."""
    a = obs["arm"]
    idx = np.arange(len(a))
    sel = np.maximum(a, 0)
    pin = np.where(a >= 0, obs["pin"][idx, sel], obs["pin"].any(axis=1))
    pad = np.where(a >= 0, obs["pad"][idx, sel], obs["pad"].any(axis=1))
    dist = np.where(a >= 0, obs["dist"][idx, sel], obs["dist"].min(axis=1))
    return pin, pad, dist


def classify(obs: dict, prev: np.ndarray, cfg: StageConfig) -> np.ndarray:
    pin, pad, dist = by_arm(obs)
    out = np.full(prev.shape, R, dtype=np.int64)
    near = (dist <= np.where(prev == P, cfg.reach_exit_m, cfg.reach_m)) | pad
    out[near] = P
    held_before = (prev == C) | (prev == PL)
    # lifted clear: the pick check's condition; once carried, brushing something does not undo it
    out[pin & (obs["lift"] >= cfg.lift_m) & (~obs["blocked"] | held_before)] = C
    over = (obs["gap"] <= np.where(prev == PL, cfg.over_exit_m, 0.0)) & (obs["floor_h"] >= -cfg.floor_slack_m)
    clear = pin & (obs["lift"] >= cfg.lift_m) & ~obs["blocked"]
    # over the container while pinched (entered from a clear hold; lowering into the container keeps it), or in
    # the container with a finger still on it
    out[over & pin & (clear | held_before)] = PL
    out[over & obs["touched"] & (prev == PL)] = PL
    at_rest = obs["speed"] <= cfg.rest_speed
    free = ~obs["touched"] & (~at_rest | ~obs["resting"])  # released and still falling / rolling: keeps its stage
    out[free] = prev[free]
    out[obs["done"]] = DONE
    out[(obs["com"][:, 2] < cfg.table_z) & ~obs["in_goal"]] = FALLEN
    return out


def progress(stage: int, obs: dict) -> np.ndarray:
    pin, _, dist = by_arm(obs)
    if stage == R:
        return -dist
    if stage == P:
        return 0.3 * pin + 0.7 * np.clip(obs["lift"] / 0.03, 0.0, 1.0) * pin
    if stage == C:
        return -np.minimum(obs["gap"], 1.0)
    if stage == PL:
        return -np.clip(obs["floor_h"], 0.0, 0.20) + 0.05 * ~obs["touched"]
    raise ValueError(stage)


class R2GTracker(StageTracker):
    def __init__(self, scene: R2GScene, cfg: StageConfig | None = None) -> None:
        super().__init__(scene, classify, cfg or scene.cfg, start_label=R)
