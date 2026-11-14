"""Success checks for the OpenWAM-added tasks (see randomization/tasks/openwam_tasks.py).

drawer_open / drawer_close   the named drawer's slide joint past a threshold, the other drawers left closed
fruit_on_plate               the named fruit resting on the plate (contact with the plate, not with the table, inside
                             the rim), no other fruit on the plate, nothing held by a gripper
plates_stack                 every plate nested on the one below (one plate on the table, the rest on plates, centred),
                             all level, at rest, released
plates_unstack               every plate flat on the table on its own (touching the table, no plate touching another),
                             at rest, released

Evaluators read the randomization metadata at reset (configure_from_randomization). The contact-based checks use
``evaluate_model_data``; ``evaluate_qpos_batch`` recomputes contacts from qpos (no velocities: "at rest" is skipped).
"""

from __future__ import annotations

from typing import Any, Mapping

import mujoco
import numpy as np

from abc_sim.task_eval.base import TaskEvalResult
from abc_sim.task_specs import SimTaskSpec

_DRAWER_JOINTS = ("joint_top_drawer_low", "joint_middle_drawer_low", "joint_bottom_drawer_low")


def _meta(randomization: object) -> Mapping[str, Any]:
    meta = getattr(randomization, "metadata", None)
    if meta is None and isinstance(randomization, Mapping):
        meta = randomization.get("metadata")
    return meta if isinstance(meta, Mapping) else {}


def _subtree_geoms(model: mujoco.MjModel, body_name: str) -> set[int]:
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if root < 0:
        raise ValueError(f"body {body_name!r} not found")
    bodies = {root}
    for b in range(model.nbody):  # bodies are ordered parent-first
        if model.body_parentid[b] in bodies and b != 0:
            bodies.add(b)
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies}


def _finger_geoms(model: mujoco.MjModel) -> set[int]:
    out = set()
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
        if ("finger" in name or "_lf_" in name or "_rf_" in name) and (name.startswith("left_") or name.startswith("right_")):
            out.add(g)
    return out


def _table_geoms(model: mujoco.MjModel) -> set[int]:
    out = set()
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_plane")
    if g >= 0:
        out.add(g)
    try:
        out |= _subtree_geoms(model, "play_table")
    except ValueError:
        pass
    return out


def _touching(data: mujoco.MjData, a: set[int], b: set[int], margin: float = 1e-3) -> bool:
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist > margin:
            continue
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in a and g2 in b) or (g1 in b and g2 in a):
            return True
    return False


def _free_qpos_adr(model: mujoco.MjModel, joint: str) -> tuple[int, int]:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if jid < 0 or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"free joint {joint!r} not found")
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def _tilt_deg(quat: np.ndarray) -> float:
    """Angle between the body's z axis and world z."""
    w, x, y, z = (float(v) for v in quat)
    zz = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(zz / max(np.linalg.norm(quat) ** 2, 1e-9), -1.0, 1.0))))


class _Base:
    def __init__(self, *, model: mujoco.MjModel, spec: SimTaskSpec, **kwargs: Any) -> None:
        self.spec = spec
        self.options = kwargs
        self._meta: Mapping[str, Any] = {}
        self._bind(model)
        self.reset(nworld=1)

    def _bind(self, model: mujoco.MjModel) -> None:
        self.model = model
        self._scratch = mujoco.MjData(model)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)
        self._max_reward = np.zeros((self._nworld,), dtype=np.float32)

    def configure_from_randomization(self, model: mujoco.MjModel, randomization: object) -> None:
        if model is not self.model:
            self._bind(model)
            self.reset(nworld=self._nworld)
        self._meta = _meta(randomization)
        self._configure()

    def _configure(self) -> None:
        pass

    def debug_spec(self) -> dict[str, object] | None:
        return None

    def _result(self, rewards: list[float], successes: list[bool], metrics: list[dict[str, Any]]) -> TaskEvalResult:
        reward = np.asarray(rewards, dtype=np.float32)
        success = np.asarray(successes, dtype=bool)
        if success.shape[0] != self._nworld:
            self.reset(nworld=success.shape[0])
        self._ever_success |= success
        self._max_reward = np.maximum(self._max_reward, reward)
        out: dict[str, Any] = {"goal_configured": bool(self._meta)}
        for key in (metrics[0] if metrics else {}):
            out[key] = np.asarray([m[key] for m in metrics]) if not isinstance(metrics[0][key], str) else metrics[0][key]
        out["max_reward_so_far"] = self._max_reward.copy()
        out["ever_success"] = self._ever_success.copy()
        return TaskEvalResult(reward=reward, success=success, metrics=out)

    def evaluate_model_data(self, model: mujoco.MjModel, data: mujoco.MjData) -> TaskEvalResult:
        if model is not self.model:
            self._bind(model)
        r, s, m = self._score(data, with_velocity=True)
        return self._result([r], [s], [m])

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float64)
        rs, ss, ms = [], [], []
        for q in qpos_batch:
            self._scratch.qpos[:] = q
            self._scratch.qvel[:] = 0.0
            mujoco.mj_forward(self.model, self._scratch)
            r, s, m = self._score(self._scratch, with_velocity=False)
            rs.append(r); ss.append(s); ms.append(m)
        return self._result(rs, ss, ms)

    def _score(self, data: mujoco.MjData, *, with_velocity: bool) -> tuple[float, bool, dict[str, Any]]:
        raise NotImplementedError


class DrawerOpenCloseEvaluator(_Base):
    """open: target drawer >= open_m (of its 0.2 m travel); close: target <= closed_m. Both: the other drawers stay
    below other_max_m (the wrong drawer opened is a failure)."""

    def __init__(self, *, model: mujoco.MjModel, spec: SimTaskSpec, mode: str = "open", open_m: float = 0.12,
                 closed_m: float = 0.015, other_max_m: float = 0.05) -> None:
        self.mode, self.open_m, self.closed_m, self.other_max_m = mode, float(open_m), float(closed_m), float(other_max_m)
        super().__init__(model=model, spec=spec)

    def _configure(self) -> None:
        self._adr = {j: int(self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in _DRAWER_JOINTS}

    def _bind(self, model: mujoco.MjModel) -> None:
        super()._bind(model)
        self._configure()

    def _score(self, data, *, with_velocity):
        target = self._meta.get("target_joint")
        if target is None:
            return 0.0, False, {"target_q_m": 0.0, "others_max_q_m": 0.0}
        q = {j: float(data.qpos[a]) for j, a in self._adr.items()}
        others = max(v for j, v in q.items() if j != target)
        others_ok = others <= self.other_max_m
        if self.mode == "open":
            done = q[target] >= self.open_m
            progress = np.clip(q[target] / self.open_m, 0.0, 1.0)
        else:
            start = float(self._meta.get("drawer_start", {}).get(target, 0.16))
            done = q[target] <= self.closed_m
            progress = np.clip((start - q[target]) / max(start - self.closed_m, 1e-6), 0.0, 1.0)
        success = bool(done and others_ok)
        reward = 1.0 if success else float(min(progress, 0.99) * (1.0 if others_ok else 0.5))
        return reward, success, {"target_q_m": q[target], "others_max_q_m": others, "others_closed": others_ok}


class FruitOnPlateEvaluator(_Base):
    """The named fruit on the plate, inside rim_frac of the plate radius, touching the plate and not the table, no other
    fruit on the plate, no fruit or plate held by a gripper, everything at rest (< rest_speed_mps)."""

    def __init__(self, *, model, spec, rim_frac: float = 0.85, rest_speed_mps: float = 0.05) -> None:
        self.rim_frac, self.rest_speed = float(rim_frac), float(rest_speed_mps)
        super().__init__(model=model, spec=spec)

    def _configure(self) -> None:
        m = self._meta
        if not m:
            return
        self._plate = _subtree_geoms(self.model, m["plate"])
        self._plate_adr = _free_qpos_adr(self.model, f'{m["plate"]}_joint')
        self._fruits = {o["name"]: (_subtree_geoms(self.model, o["name"]), _free_qpos_adr(self.model, o["joint"])) for o in m["objects"]}
        self._table = _table_geoms(self.model)
        self._fingers = _finger_geoms(self.model)
        self._radius = float(m["plate_radius"])
        # progress is measured from where the target starts (zero at reset): the drop in its distance to the plate
        st = getattr(self, "_object_states", None) or {}
        tgt, pl = st.get(f'{m["target_object"]}_joint'), st.get(f'{m["plate"]}_joint')
        self._d0 = float(np.linalg.norm(np.asarray(tgt["pos"][:2]) - np.asarray(pl["pos"][:2]))) if tgt and pl else None

    def configure_from_randomization(self, model, randomization) -> None:
        states = getattr(randomization, "object_states", None)
        if states is None and isinstance(randomization, Mapping):
            states = randomization.get("object_states")
        self._object_states = states if isinstance(states, Mapping) else {}
        super().configure_from_randomization(model, randomization)

    def _on_plate(self, data, name) -> bool:
        geoms, (qa, _) = self._fruits[name]
        d = float(np.linalg.norm(data.qpos[qa: qa + 2] - data.qpos[self._plate_adr[0]: self._plate_adr[0] + 2]))
        return d <= self.rim_frac * self._radius and _touching(data, geoms, self._plate) and not _touching(data, geoms, self._table)

    def _score(self, data, *, with_velocity):
        m = self._meta
        if not m:
            return 0.0, False, {}
        target = m["target_object"]
        on = {n: self._on_plate(data, n) for n in self._fruits}
        held = any(_touching(data, g, self._fingers) for g, _ in self._fruits.values()) or _touching(data, self._plate, self._fingers)
        speeds = [float(np.linalg.norm(data.qvel[da: da + 3])) for _, (_, da) in self._fruits.values()] if with_velocity else [0.0]
        at_rest = max(speeds) <= self.rest_speed
        wrong = sorted(n for n, v in on.items() if v and n != target)
        success = bool(on[target] and not wrong and not held and at_rest)
        qa = self._fruits[target][1][0]
        dist = float(np.linalg.norm(data.qpos[qa: qa + 2] - data.qpos[self._plate_adr[0]: self._plate_adr[0] + 2]))
        # progress: 0 at reset, 0.9 once the target is over the plate, 0.95 on the plate but not yet a success
        # (another fruit on it, still held, or moving), 1 on success; halved while another fruit is on the plate
        d0 = self._d0 if self._d0 is not None else dist
        goal = self.rim_frac * self._radius
        progress = 0.95 if on[target] else float(np.clip((d0 - dist - 0.02) / max(d0 - goal - 0.02, 1e-3), 0.0, 1.0)) * 0.9  # 2 cm deadband: settling is not progress
        reward = 1.0 if success else min(progress * (0.5 if wrong else 1.0), 0.99)
        return reward, success, {"target_on_plate": on[target], "other_fruit_on_plate": len(wrong), "held": held,
                                 "at_rest": at_rest, "target_plate_dist_m": dist}


class PlateStackEvaluator(_Base):
    """stack: one plate touches the table, every other plate touches a plate below it and is centred on the plate
    under it (xy within center_frac of the radius); unstack: every plate touches the table and no plate touches
    another. Both: all plates level (tilt < max_tilt_deg), released, at rest."""

    def __init__(self, *, model, spec, mode: str = "stack", center_frac: float = 0.35, max_tilt_deg: float = 20.0,
                 rest_speed_mps: float = 0.05) -> None:
        self.mode, self.center_frac, self.max_tilt, self.rest_speed = mode, float(center_frac), float(max_tilt_deg), float(rest_speed_mps)
        super().__init__(model=model, spec=spec)

    def _configure(self) -> None:
        m = self._meta
        if not m:
            return
        self._plates = [(p, _subtree_geoms(self.model, p), _free_qpos_adr(self.model, f"{p}_joint")) for p in m["plates"]]
        self._table = _table_geoms(self.model)
        self._fingers = _finger_geoms(self.model)
        self._radius = float(m["plate_radius"])

    def _score(self, data, *, with_velocity):
        if not self._meta:
            return 0.0, False, {}
        n = len(self._plates)
        on_table = [_touching(data, g, self._table) for _, g, _ in self._plates]
        pos = [data.qpos[qa: qa + 3].copy() for _, _, (qa, _) in self._plates]
        tilt = [_tilt_deg(data.qpos[qa + 3: qa + 7]) for _, _, (qa, _) in self._plates]
        level = max(tilt) <= self.max_tilt
        held = any(_touching(data, g, self._fingers) for _, g, _ in self._plates)
        at_rest = (max(float(np.linalg.norm(data.qvel[da: da + 3])) for _, _, (_, da) in self._plates) <= self.rest_speed) if with_velocity else True
        touch = [[i != j and _touching(data, self._plates[i][1], self._plates[j][1]) for j in range(n)] for i in range(n)]
        if self.mode == "stack":
            order = sorted(range(n), key=lambda i: pos[i][2])
            nested = 0
            for lo, hi in zip(order, order[1:]):
                centred = float(np.linalg.norm(pos[hi][:2] - pos[lo][:2])) <= self.center_frac * self._radius
                if centred and touch[hi][lo] and not on_table[hi]:
                    nested += 1
            done = nested == n - 1 and on_table[order[0]]
            progress = nested / max(n - 1, 1)
            metrics = {"plates_nested": nested, "plates": n}
        else:
            alone = sum(1 for i in range(n) if on_table[i] and not any(touch[i]))
            done = alone == n
            progress = alone / n
            metrics = {"plates_alone_on_table": alone, "plates": n}
        success = bool(done and level and not held and at_rest)
        reward = 1.0 if success else float(min(progress, 0.99))
        metrics.update(level=level, max_tilt_deg=max(tilt), held=held, at_rest=at_rest)
        return reward, success, metrics


class WipeWhiteboardEvaluator(_Base):
    """Fraction of marker segments erased (the runtime makes erased segments transparent). Success: at least
    erased_frac of them erased and the duster released (no finger contact). Progress = fraction erased (0 at reset)."""

    def __init__(self, *, model, spec, erased_frac: float = 0.9) -> None:
        self.erased_frac = float(erased_frac)
        super().__init__(model=model, spec=spec)

    def _configure(self) -> None:
        self._marks = [g for g in range(self.model.ngeom) if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("mark_")]
        self._duster = _subtree_geoms(self.model, "duster") if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "duster") >= 0 else set()
        self._fingers = _finger_geoms(self.model)

    def _bind(self, model) -> None:
        super()._bind(model)
        self._configure()

    def _score(self, data, *, with_velocity):
        if not self._marks:
            return 0.0, False, {}
        frac = float(np.mean([self.model.geom_rgba[g, 3] < 0.5 for g in self._marks]))
        held = _touching(data, self._duster, self._fingers)
        success = bool(frac >= self.erased_frac and not held)
        reward = 1.0 if success else min(frac, 0.99)
        return reward, success, {"erased_frac": frac, "marks": len(self._marks), "duster_held": held}


class HangHeadphonesEvaluator(_Base):
    """The headset hanging on the stand: touching the stand, not the table, released (no finger contact), at rest.
    Progress: how much of the headset's start distance to the hanging point (randomization metadata) is closed,
    up to 0.9; 0.95 once it touches the stand off the table but is still held or moving; 1 on success."""

    def __init__(self, *, model, spec, rest_speed_mps: float = 0.05) -> None:
        self.rest_speed = float(rest_speed_mps)
        super().__init__(model=model, spec=spec)

    def configure_from_randomization(self, model, randomization) -> None:
        states = getattr(randomization, "object_states", None)
        if states is None and isinstance(randomization, Mapping):
            states = randomization.get("object_states")
        self._start = np.asarray(states["headset_joint"]["pos"], float) if isinstance(states, Mapping) and "headset_joint" in states else None
        super().configure_from_randomization(model, randomization)

    def _configure(self) -> None:
        if not self._meta:
            return
        self._head = _subtree_geoms(self.model, "headset")
        self._stand = _subtree_geoms(self.model, "headset_stand")
        self._adr = _free_qpos_adr(self.model, "headset_joint")
        self._table = _table_geoms(self.model)
        self._fingers = _finger_geoms(self.model)
        self._hp = np.asarray(self._meta["hang_point"], float)

    def _score(self, data, *, with_velocity):
        if not self._meta:
            return 0.0, False, {}
        on_stand = _touching(data, self._head, self._stand)
        on_table = _touching(data, self._head, self._table)
        held = _touching(data, self._head, self._fingers)
        speed = float(np.linalg.norm(data.qvel[self._adr[1]: self._adr[1] + 3])) if with_velocity else 0.0
        at_rest = speed <= self.rest_speed
        pos = data.qpos[self._adr[0]: self._adr[0] + 3]
        dist = float(np.linalg.norm(pos - self._hp))
        hanging = on_stand and not on_table
        success = bool(hanging and not held and at_rest)
        d0 = float(np.linalg.norm(self._start - self._hp)) if getattr(self, "_start", None) is not None else dist
        progress = 0.95 if hanging else 0.9 * float(np.clip((d0 - dist - 0.02) / max(d0 - 0.05, 1e-3), 0.0, 1.0))  # 2 cm deadband
        reward = 1.0 if success else min(progress, 0.99)
        return reward, success, {"on_stand": on_stand, "on_table": on_table, "held": held, "at_rest": at_rest, "hang_point_dist_m": dist}


__all__ = ["DrawerOpenCloseEvaluator", "FruitOnPlateEvaluator", "PlateStackEvaluator", "WipeWhiteboardEvaluator", "HangHeadphonesEvaluator"]
