"""Single-stage subtasks (reach, pick, carry, place) of the Real2Gym "put it in a container" tasks.

Env ids ``r2g_<task>__<stage>``; one evaluator class per task (generated from ``R2G_STAGE_TASKS``), all the
same code. One episode = one target object and the arm its instruction names, both from the existing spec
(``real2gym_spec.resolve_spec`` on the scene's start layout, exactly as ``r2g_<task>__obj_<class>``). The place
stage's goal is the obj env's own check (``Real2GymEvaluator``, variant "obj").

Use: ``env.reset_to_stage(seed, row)`` with a saved start, or ``env.reset(seed=s)`` for a reach episode on the
plain scene (target drawn from the seed's layout). Stage table: subtask_rewards/r2g/stage_table.md.
"""
from __future__ import annotations

import zlib

import mujoco
import numpy as np

from abc_sim.real2gym_spec import resolve_spec
from abc_sim.task_eval.r2g_stages import (C, DONE, FALLEN, LABELS, PL, STAGE_PROMPTS, STAGES, R2GScene, R2GTracker,
                                          StageConfig, by_arm, progress, stage_objects)
from abc_sim.task_eval.real2gym_tasks import Real2GymEvaluator
from abc_sim.task_eval.stagekit.robot import ARMS
from abc_sim.task_eval.stagekit.subtask import SubtaskEvaluator

# Tasks with stage envs, and the goal objects left out of the target pool (a scripted top-down pinch cannot lift
# them: subtask_rewards/r2g/STATUS.md).
R2G_STAGE_TASKS: dict[str, tuple[str, ...]] = {
    "put_the_screwdriver_in_the_bin": (),
    "place_and_organize_the_fake_fruits_in_the_fruit_bowl": (),
    "organize_the_makeup": (),
    "organize_the_condiment_bottles": ("bottle03_grinder",),  # no collision-free hold found in 24 of 24 scenes
    "sort_the_hair_cutting_tools": (),
    "sort_the_tools_into_containers": (),
    "sort_the_stationery_into_containers": (),
    "place_and_organize_the_beverage_onto_the_shelf": (),
    "place_and_organize_the_canned_foods_onto_the_counter": (),
    "put_the_wine_bottle_on_the_wine_rack": (),
}


class R2GSubtaskEvaluator(SubtaskEvaluator):
    STAGES, LABELS, FALLEN, STAGE_PROMPTS = STAGES, LABELS, FALLEN, STAGE_PROMPTS
    TARGET_KEY = "object"
    R2G_TASK = ""
    EXCLUDE: tuple[str, ...] = ()
    stage_env = False  # env.reset(seed) is the plain scene (a reach episode); saved starts go through reset_to_stage

    # -- hooks ---------------------------------------------------------------------------------------------------
    def make_config(self):
        return StageConfig()

    def make_scene(self, model):
        return R2GScene(model, self.R2G_TASK, self.cfg)

    def make_tracker(self, scene):
        return R2GTracker(scene, self.cfg)

    def progress(self, stage, obs):
        return progress(stage, obs)

    def _bind(self, model):
        self._layout: np.ndarray | None = getattr(self, "_layout", None)
        self._inner_cache: dict[str, Real2GymEvaluator] = {}
        self.r2g_spec: dict = {}
        super()._bind(model)

    def configure_from_randomization(self, model, randomization):
        if model is not self.model or not self.scene.ready:
            self._bind(model)
        self._layout = None  # a new scene: the next evaluation is its start layout
        self.scene.inner, self.scene.target = None, -1
        self.scene.arm[:] = -1

    # -- the episode's spec --------------------------------------------------------------------------------------
    def _layout_data(self) -> mujoco.MjData:
        d = mujoco.MjData(self.model)
        d.qpos[:] = self._layout
        mujoco.mj_forward(self.model, d)
        return d

    def spec_for(self, k: int) -> dict:
        """The obj env's spec for object k on the scene's start layout."""
        f = self.scene.full
        return resolve_spec(self.model, self._layout_data(), self.R2G_TASK, "obj", self.scene.names[k],
                            names=list(f._objs), extents={n: o["extent"] for n, o in f._objs.items()})

    def candidates(self) -> list[int]:
        """Objects that can be this scene's target: in the pool, spec valid, goal not met at the start."""
        d = self._layout_data()
        obs = self.scene.observe(d, self.scene.heights(d))
        out = []
        for k, n in enumerate(self.scene.names):
            if n in self.EXCLUDE or obs["in_goal"][k]:
                continue
            sp = self.spec_for(k)
            if sp.get("valid") and sp.get("arm") in ARMS:
                out.append(k)
        return out

    def pick_target(self, obs):
        cand = self.candidates()
        if not cand:
            raise RuntimeError(f"{self.R2G_TASK}: no valid target in this layout")
        key = zlib.crc32(np.round(self._layout, 6).tobytes())  # deterministic in the seed's layout
        return int(cand[int(np.random.default_rng(key).integers(len(cand)))])

    def begin(self, model, data, target, rest_height=None, labels=None):
        if model is not self.model:
            self._bind(model)
        if self._layout is None:  # begun on the scene's start layout (plain reset)
            self._layout = np.array(data.qpos, copy=True)
        sc = self.scene
        k = int(target)
        name = sc.names[k]
        spec = self.spec_for(k)
        self.r2g_spec = spec
        layout = self._layout_data()
        inner = self._inner_cache.get(name)
        if inner is None:
            inner = self._inner_cache[name] = Real2GymEvaluator(model=model, spec=None, task=self.R2G_TASK,
                                                                variant="obj", object=name)
        inner.reset(nworld=1)
        inner.set_spec(spec, data=layout)
        rest = sc.heights(layout) if rest_height is None else np.asarray(rest_height, np.float64).reshape(-1, 6)
        pinch0 = float(rest[-1, 0])
        if pinch0 > 0:
            inner._pinch_t[name] = pinch0
        sc.arm[:] = -1
        sc.arm[k] = ARMS.index(spec["arm"]) if spec.get("arm") in ARMS else -1
        sc.inner, sc.target = inner, k
        super().begin(model, data, k, rest_height=rest, labels=labels)
        o = self.tracker.obs
        self._start_xy = o["xy"][k].copy()
        self._placed = o["in_goal"].copy()
        self._placed[k] = False
        self._was_touched = bool(o["touched"][k])
        self._released = False
        self.episode_spec = {"instruction": self.render_prompt(spec), "arm": spec.get("arm") or "",
                             "target": name, "title": f"{self.R2G_TASK}: {self.stage_name}",
                             "valid": bool(spec.get("valid"))}

    def reset(self, *, nworld: int = 1) -> None:
        super().reset(nworld=nworld)
        self._released = False
        self._was_touched = False

    def evaluate_model_data(self, model, data):
        if model is not self.model or not self.scene.ready:
            self._bind(model)
        if self.target is None and self._layout is None:
            self._layout = np.array(data.qpos, copy=True)
        return super().evaluate_model_data(model, data)

    def episode_state(self):
        st = super().episode_state()
        st["rest_height"][-1, 0] = float(self.tracker.obs.get("pinch_s", 0.0))  # carried to the next stage
        return st

    # -- instruction ---------------------------------------------------------------------------------------------
    def stage_prompt(self) -> str:
        """The instruction rendered for this episode's start (``begin``), else the stage's generic prompt."""
        return (getattr(self, "episode_spec", None) or {}).get("instruction") or self.STAGE_PROMPTS[self.stage_name]

    def render_prompt(self, spec: dict) -> str:
        """The stage's instruction from the spec's own phrases (objects by feature and position, never by name)."""
        full = spec.get("instruction") or ""
        head, sep, goal = full.partition(" and grasp it, then ")
        if not sep or " to " not in head:
            return full or self.STAGE_PROMPTS[self.stage_name]
        arm = spec.get("arm") or "nearer"
        where = head.split(" to ", 1)[1]  # the object with its position at the start: true for reach and pick
        clause = (spec.get("position") or {}).get("clause") or ""
        obj = where[: -len(clause)].rstrip() if clause and where.endswith(clause) else where
        goal = goal.rstrip(".")
        container = ""
        for word in (" into ", " onto ", " in ", " on "):
            if word in goal:
                container = goal.rsplit(word, 1)[1]
                break
        named = goal.replace(" it ", f" {obj} ", 1) if " it " in goal else f"{goal} ({obj})"
        return {"reach": head + ".",
                "pick": f"Grasp {where} with the {arm} arm and lift it.",
                "carry": (f"Carry {obj} in the {arm} gripper over {container}." if container else
                          f"Carry {obj} in the {arm} gripper to where it goes."),
                "place": f"With the {arm} arm, {named}."}[self.stage_name]

    def goal_reached(self, k, obs, raw_label):
        # place: the obj check's own condition at this evaluation, so the 0.5 s hold restarts exactly when the
        # check's does (a released object keeps its stage label while it still moves, which would bridge a wobble)
        if self.stage == PL:
            return bool(obs["done"][k])
        return super().goal_reached(k, obs, raw_label)

    # -- failures ------------------------------------------------------------------------------------------------
    def step_failure(self, k, obs, label):
        cfg = self.cfg
        if obs["stock"].get("sim_fault"):
            return "sim fault"
        if obs["container_moved"] > cfg.container_moved_m or obs["container_tilt"] > cfg.container_tilt_rad:
            return "container moved or tipped"
        at_rest = obs["speed"] <= cfg.rest_speed
        if np.any(self._placed & ~obs["in_goal"] & at_rest & ~obs["touched"]):
            return "knocked a placed object out"
        touched = bool(obs["touched"][k])
        pin_any = bool(obs["pin"][k].any())
        if self.stage < C:
            if not pin_any and float(np.linalg.norm(obs["xy"][k] - self._start_xy)) > cfg.push_m:
                return "pushed"
        elif self._was_touched and not touched:
            if obs["gap"][k] > cfg.over_exit_m and not obs["in_goal"][k]:
                return "released outside the container"
            self._released = True
        if (self._released and not touched and obs["resting"][k] and at_rest[k] and not obs["in_goal"][k]):
            return "landed outside the container"
        self._was_touched = touched
        return ""

    def extra_metrics(self, k, obs):
        pin, pad, dist = by_arm(obs)
        stock = obs["stock"]
        return dict(lift_m=float(obs["lift"][k]), pinned=bool(pin[k]), dist_m=float(dist[k]),
                    blocked=bool(obs["blocked"][k]), gap_m=float(obs["gap"][k]), floor_h_m=float(obs["floor_h"][k]),
                    in_goal=bool(obs["in_goal"][k]), touched=bool(obs["touched"][k]), speed=float(obs["speed"][k]),
                    pinch_s=float(obs["pinch_s"]), container_moved_m=float(obs["container_moved"]),
                    container_tilt_rad=float(obs["container_tilt"]),
                    wrong_arm=bool(stock.get("wrong_arm", False)), stock_success=bool(stock.get("success", False)),
                    stock_ever_success=bool(stock.get("ever_success", False)),
                    sim_fault=bool(stock.get("sim_fault", False)))

    def build_start(self, env, seed, rng=None):
        from abc_sim.task_eval.r2g_starts import build_start
        return build_start(env, seed, self.stage_name, rng=rng)

    def evaluate_qpos_batch(self, qpos_batch):
        raise RuntimeError("Real2Gym subtasks need evaluate_model_data (contacts and velocities)")


def _make(task: str, exclude: tuple[str, ...]) -> type[R2GSubtaskEvaluator]:
    cls = type(f"R2G_{task}", (R2GSubtaskEvaluator,), {
        "TASK": f"r2g_{task}", "ENV_TASK": f"r2g_{task}", "NAME": f"r2g_subtask__{task}", "R2G_TASK": task,
        "EXCLUDE": tuple(exclude), "__module__": __name__,
        # prompts are lookup keys: one set per task
        "STAGE_PROMPTS": {s: f"{p} ({task.replace('_', ' ')})" for s, p in STAGE_PROMPTS.items()}})
    return cls


for _task, _exclude in R2G_STAGE_TASKS.items():  # registration: abc_sim/task_eval/subtasks.py finds these classes
    globals()[f"R2G_{_task}"] = _make(_task, _exclude)

__all__ = ["R2GSubtaskEvaluator", "R2G_STAGE_TASKS", "stage_objects", "DONE", "PL"]
