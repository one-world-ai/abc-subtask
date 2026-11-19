"""Success checks and progress for the Real2Gym tasks and their variants (abc_sim/real2gym_catalog.py).

Full task (variant None / "staged"):
Success: every goal met (``count`` goals: k of their objects), every constraint met, no goal object touching a gripper
finger, all goal objects at rest (< rest_speed_mps; skipped for qpos-only evaluation).
Progress: per goal the items met now beyond those met at reset (0 at reset), summed over goals and divided by the
items not met at reset; 0.99 at most until success. ``erased`` counts the erased fraction / frac.
An ``off`` item not already off at reset must also have moved >= off_move_m (3 cm) from its reset position: contact
alone flickers for a plug resting in its socket (counted as pulled with no motion at all).

Subtask envs ("pick", "obj"): instruction = enforced spec (abc_sim/real2gym_spec.py, SPEC_VERSION 2). The spec
{object, arm, targets, qualifier, position, instruction, valid} is resolved at reset from the settled layout (by the
randomizer, which puts it into the randomization metadata as ``r2g_spec``; the prompt runtime shows its instruction;
a class env draws its target instance ``object`` there) and the check enforces its clauses:
  pick  both finger pads of the DESIGNATED arm touch an accepted target, that target's centre of mass is >= lift_m
        (3 cm) above its own reset height, and it is lifted off what it rested on (the instruction's "lift it off the
        table" / "lift it out of the blue basket"): it touches nothing but the designated gripper (no fixed scene
        geometry, not the other arm or the arm's links, no other object except one the designated gripper pinches as
        well: a plate tilted up on one edge, or a bowl raised together with the bowl under it, is not lifted off) and,
        for "out of <container>" (spec.lift_out_of), its lowest point is above the container's rim (highest point
        of its collision geometry); all of
        it continuously for >= hold_s (0.5 s) of sim time. Progress 0.3 x pinched by the designated arm + 0.7 x lift
        fraction while pinched (0.99 while the hold runs). ``failed`` as before: the env's object pushed > 8 cm
        without a grasp, or off the table.
  obj   an accepted target meets the object's goal (the catalog goal, as before), the DESIGNATED arm pinched that
        target with both pads for >= pinch_s (0.2 s) in total while its goal was not met (grasp, then place: an object
        pushed or dropped into place by anything else does not count), it touches no finger and is at rest; all of
        it continuously for >= hold_s (0.5 s). Progress 0 until that target's goal is met after such a grasp (0.99
        until the hold is complete), 1 on success. ``failed`` as before (only the sim-fault latch).
  A spec that is not valid (target buried under a look-alike, object flagged in the phrase table) never succeeds.
Metrics that do not gate success: ``wrong_arm`` (the other arm pinched a target; latched), ``wrong_object`` (a goal
object of the task that is not a target was pinched and lifted >= lift_m; latched), ``disturbed`` (other free objects
now > 5 cm from their reset place or tilted > 30 deg from their reset pose), the spec (spec_arm, spec_targets, ...).
Hold and grasp times use the sim clock (data.time); qpos-only evaluation counts one control step per call.
"""

from __future__ import annotations

import mujoco
import numpy as np

from abc_sim import sim_guard
from abc_sim.randomization.tasks.real2gym_tasks import load_library
from abc_sim.real2gym_catalog import task_config
from abc_sim.real2gym_spec import resolve_spec, spec_metrics
from abc_sim.task_eval.openwam_tasks import _Base, _finger_geoms, _meta, _subtree_geoms, _table_geoms, _touching


def _tilt_deg(xmat) -> float:
    return float(np.degrees(np.arccos(np.clip(xmat.reshape(3, 3)[2, 2], -1.0, 1.0))))


def _tilt_ok(xmat, rng) -> bool:
    return rng is None or (rng[0] <= _tilt_deg(xmat) <= rng[1])


def variant_config(task: str, variant: str | None, obj: str | None, targets: list[str] | None = None) -> dict:
    """The task's goals for a variant: "obj" keeps object ``obj``'s goal only (no constraints), checked for each of the
    accepted ``targets`` (the spec's: ``obj`` and the look-alikes the instruction does not tell apart; default [obj]);
    others unchanged."""
    cfg = dict(task_config(task))
    if variant == "obj":
        goals = []
        for g in cfg["goals"]:
            if obj in g.get("objects", []) and g["kind"] in ("in", "off", "upright"):
                goals.append(dict(g, objects=list(targets or [obj]), count=None))
        cfg["goals"], cfg["constraints"] = goals, []
    return cfg


class Real2GymEvaluator(_Base):
    """Full task (variant None / "staged"), or a subtask env whose check enforces the instruction rendered at reset
    (module docstring): bring one object to its goal after grasping it with the designated arm ("obj"), or grasp one
    object with the designated arm, lift it and hold it ("pick"; metric ``failed`` when the env's object was pushed
    > 8 cm from its start without a grasp or fell off the table)."""

    SUBTASKS = ("pick", "obj")

    def __init__(self, *, model: mujoco.MjModel, spec, task: str, variant: str | None = None, object: str | None = None,
                 members=None, group: str | None = None,
                 rest_speed_mps: float = 0.05, lift_m: float = 0.03, off_move_m: float = 0.03, hold_s: float = 0.5,
                 pinch_s: float = 0.2, max_gap_s: float = 0.11, control_steps: int = 17, disturbed_m: float = 0.05,
                 disturbed_deg: float = 30.0) -> None:
        # object: the fixed target instance (legacy env) or None (class env: ``members``, one drawn per reset)
        self.task, self.variant, self.object = task, variant, object
        self.members = [object] if object is not None else list(members or [])
        self.group = group
        self._obj = self.members[0] if self.members else None  # the target instance of the current episode
        self.off_move_m = float(off_move_m)
        self._pos0 = None
        self.cfg = variant_config(task, variant, self._obj)
        self.rest_speed = float(rest_speed_mps)
        self.lift_m = float(lift_m)
        self.hold_s, self.pinch_s = float(hold_s), float(pinch_s)
        self.max_gap_s = float(max_gap_s)  # evaluations further apart than this do not count as continuous
        self.control_steps = int(control_steps)
        self.disturbed_m, self.disturbed_cos = float(disturbed_m), float(np.cos(np.radians(disturbed_deg)))
        self._baseline = None
        self._spec = None
        super().__init__(model=model, spec=spec)

    def reset(self, *, nworld: int = 1) -> None:
        super().reset(nworld=nworld)
        self._baseline = None
        self._pos0 = None
        self._fault, self._fault_score, self._last_reward = None, None, 0.0
        self._free_prev = None
        had_targets = bool(self._spec) and self.variant == "obj"
        self._spec, self._ref = None, None  # the spec and the reset poses it was resolved from
        self._hold, self._pinch_t = {}, {}  # target -> (t first, t last) of the running hold; -> seconds pinched
        self._t_prev, self._vclock = None, 0.0
        self._wrong_arm, self._wrong_object, self._disturbed_max = False, "", 0
        self._obj = self.members[0] if self.members else None
        if had_targets:  # back to the env's own object until the next spec
            self.cfg = variant_config(self.task, self.variant, self._obj)
            if getattr(self, "_ready", False):
                self._bind_goals(self.model)

    # ------------------------------------------------------------------ spec (subtask envs)
    def configure_from_randomization(self, model: mujoco.MjModel, randomization: object) -> None:
        if model is not self.model:
            self._bind(model)
            self.reset(nworld=self._nworld)
        self._meta = _meta(randomization)
        spec = self._meta.get("r2g_spec")
        if (self.variant in self.SUBTASKS and getattr(self, "_ready", False) and isinstance(spec, dict)
                and spec.get("task") == self.task and spec.get("variant") == self.variant
                and spec.get("object") in self.members):
            self.set_spec(spec, object_states=getattr(randomization, "object_states", None))

    def set_spec(self, spec: dict, *, data=None, object_states=None) -> None:
        """Adopt the spec resolved at reset and the reset poses its heights refer to (``object_states``: the
        randomization's settled free-joint poses; else ``data``: the state at hand)."""
        self._spec = dict(spec)
        self._obj = spec.get("object") or self._obj
        if self.variant == "obj":
            self.cfg = variant_config(self.task, self.variant, self._obj, [n for n in spec.get("targets", []) if n in self._objs])
            self._bind_goals(self.model)
        ref = self._ref_from_states(object_states) if object_states else None
        if not ref and data is not None:
            ref = self._ref_from_data(data)
        self._ref = ref or None

    def debug_spec(self) -> dict | None:
        return dict(self._spec) if self._spec else None

    def _ref_from_data(self, data) -> dict:
        return {n: (data.xipos[b].copy(), data.xmat[b].reshape(3, 3)[:, 2].copy()) for n, b in self._body.items()}

    def _ref_from_states(self, states) -> dict:
        """name -> (centre of mass, body z axis) from settled free-joint poses {joint: {pos, quat}}."""
        out = {}
        for n, o in self._objs.items():
            st = states.get(o["free_joint"]) if o["free_joint"] else None
            if not st:
                continue
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, np.asarray(st["quat"], float))
            R = R.reshape(3, 3)
            out[n] = (np.asarray(st["pos"], float) + R @ self.model.body_ipos[self._body[n]], R[:, 2].copy())
        return out

    def _bind(self, model: mujoco.MjModel) -> None:
        super()._bind(model)
        lib = load_library(self.task)
        self._objs = {o["name"]: o for o in lib["objects"]}
        bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        if any(bid(n) < 0 for n in self._objs):  # not this task's scene (e.g. before the first reload)
            self._ready = False
            return
        self._ready = True
        self.__dict__.pop("_pads", None)  # _pinched's per-model cache
        self._table = _table_geoms(model)
        self._fingers = _finger_geoms(model)
        # static scene geometry (welded to the world: table, mats, pads, shelves, fixed props) with collision: an
        # object resting on any of it outside its container counts as "off" (not only on the table top)
        self._static = {g for g in range(model.ngeom) if int(model.body_weldid[int(model.geom_bodyid[g])]) == 0
                        and (model.geom_contype[g] or model.geom_conaffinity[g])} | set(self._table)
        self._geoms = {n: _subtree_geoms(model, n) for n in self._objs}
        self._body = {n: bid(n) for n in self._objs}
        self._bind_goals(model)
        # one-pass contact lookups for the subtask checks: geom -> object, finger geom -> (arm, finger)
        self._geom_obj = {g: n for n, gs in self._geoms.items() for g in gs}
        self._pad_of = {}
        for g in self._fingers:
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
            self._pad_of[g] = ("left" if n.startswith("left_") else "right",
                               "lf" if "_lf_" in n or "left_finger" in n else "rf")
        # the robot: per arm the geoms of its gripper (link 6 and below: wrist camera, fingers) and of the whole arm
        self._grip_geoms, self._robot_geoms = {}, set()
        for arm in ("left", "right"):
            try:
                self._grip_geoms[arm] = _subtree_geoms(model, f"{arm}_link_6")
            except ValueError:
                self._grip_geoms[arm] = {g for g, (a, _) in self._pad_of.items() if a == arm}
            try:
                self._robot_geoms |= _subtree_geoms(model, f"{arm}_arm")
            except ValueError:
                self._robot_geoms |= self._grip_geoms[arm]
        self._zcache = {}  # geom -> mesh vertices (local frame) or box corners, for the "out of" heights
        self._free = [n for n, o in self._objs.items() if o["free_joint"]
                      and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, o["free_joint"]) >= 0]
        self._task_goal_objects = sorted({n for g in task_config(self.task)["goals"] for n in g.get("objects", [])
                                          if n in self._objs})
        self._jadr = {}
        for g in self.cfg["goals"]:
            if g["kind"] == "joint":
                self._jadr[g["joint"]] = int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, g["joint"])])
        self._cons = []
        for c in self.cfg.get("constraints", []):
            if c["kind"] == "joints_max":
                adr = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in c["joints"]]
                self._cons.append(("joints_max", adr, float(c["max"])))
        self._ink = np.array([g for g in range(model.ngeom)
                              if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("ink_")], dtype=int)

    def _bind_goals(self, model: mujoco.MjModel) -> None:
        """The objects the goals of ``self.cfg`` name (an obj env: the spec's targets) and their free-joint dofs."""
        self._goal_objects = sorted({n for g in self.cfg["goals"] for n in g.get("objects", [])})
        self._dof = {}
        for n in self._goal_objects:
            fj = self._objs[n]["free_joint"]
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, fj) if fj else -1
            self._dof[n] = int(model.jnt_dofadr[jid]) if jid >= 0 else None

    # ------------------------------------------------------------------ geometry
    def _local(self, data, cont, name):
        cb = self._body[cont]
        return data.xmat[cb].reshape(3, 3).T @ (data.xipos[self._body[name]] - data.xpos[cb])

    def _in_box(self, data, cont, name, *, margin=0.0, above=0.0, floor_slack=0.01) -> bool:
        ext = self._objs[cont]["extent"]
        p = self._local(data, cont, name)
        return bool(ext[0] + margin <= p[0] <= ext[1] - margin and ext[2] + margin <= p[1] <= ext[3] - margin
                    and ext[4] - floor_slack <= p[2] <= ext[5] + above)

    def _resting(self, data, name) -> bool:
        g = self._geoms[name]
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist > 1e-3:
                continue
            a, b = int(c.geom1), int(c.geom2)
            if (a in g) != (b in g) and not ((a in self._fingers) or (b in self._fingers)):
                return True
        return False

    # ------------------------------------------------------------------ goals
    def _goal_in(self, data, g):
        met = []
        for n in g["objects"]:
            ok = False
            for c in g["containers"]:
                if not self._in_box(data, c, n, margin=g["margin"], above=g["above"], floor_slack=g["floor_slack"]):
                    continue
                # supports: the container (and the goal's extra supports), or another goal object inside its box
                sup = set(self._geoms[c]).union(*[self._geoms[x] for x in g.get("supports", [])])
                for o in self._goal_objects:
                    if o != n and o != c and self._in_box(data, c, o, above=g["above"], floor_slack=g["floor_slack"]):
                        sup |= self._geoms[o]
                if (_touching(data, self._geoms[n], sup) and not _touching(data, self._geoms[n], self._table)
                        and _tilt_ok(data.xmat[self._body[n]], g["tilt"])):
                    ok = True
                    break
            met.append(ok)
        return met

    def _off_base(self, data, g):
        """Per object of an ``off`` goal: not touching / inside any container and within the tilt range."""
        cont_geoms = set().union(*[self._geoms[c] for c in g["containers"]])
        support = self._static - cont_geoms
        base = {}
        for n in g["objects"]:
            outside = not any(self._in_box(data, c, n, margin=-g["clear"], above=g["clear"], floor_slack=g["clear"])
                              for c in g["containers"])
            touching = _touching(data, self._geoms[n], cont_geoms)
            # leaning on the container's outside is fine where the goal allows it (bags, racks), if the object rests
            # on the table or a fixed prop
            away = outside and (not touching or (g.get("touch_ok", False) and _touching(data, self._geoms[n], support)))
            xm = data.xmat[self._body[n]]
            tilt_ok = _tilt_ok(xm, g["tilt"]) or (g.get("either_face", False) and g["tilt"] is not None
                                                   and g["tilt"][0] <= 180.0 - _tilt_deg(xm) <= g["tilt"][1])
            base[n] = away and tilt_ok
        return base

    def _goal_off(self, data, g):
        if self._pos0 is None:  # first evaluation after reset: where every goal object starts, and which are off already
            self._pos0 = {n: data.xipos[self._body[n]].copy() for n in self._goal_objects}
            self._off0 = {n for gg in self.cfg["goals"] if gg["kind"] == "off" for n, v in self._off_base(data, gg).items() if v}
        base = self._off_base(data, g)
        for n in base:
            if base[n] and n not in self._off0:
                base[n] = float(np.linalg.norm(data.xipos[self._body[n]] - self._pos0[n])) >= self.off_move_m
        cont = set().union(*[self._geoms[c] for c in g["containers"]])
        support = self._static - cont
        supported = {n for n, v in base.items() if v and _touching(data, self._geoms[n], support)}
        grew = True
        while grew:  # objects resting on objects that are off (a stack of plates)
            grew = False
            for n, v in base.items():
                if v and n not in supported and any(_touching(data, self._geoms[n], self._geoms[s]) for s in supported):
                    supported.add(n)
                    grew = True
        return [n in supported for n in g["objects"]]

    def _goal_upright(self, data, g):
        out = []
        for n in g["objects"]:
            ok = _tilt_ok(data.xmat[self._body[n]], g["tilt"]) and self._resting(data, n)
            if ok and g.get("on_table"):
                ok = _touching(data, self._geoms[n], self._table)
            out.append(ok)
        return out

    def _erased_frac(self) -> float:
        if not len(self._ink):
            return 0.0
        return float(np.mean(self.model.geom_rgba[self._ink, 3] < 0.5))

    def _evaluate_goals(self, data):
        """-> per goal (items met, fractional for erased; items needed)."""
        out = []
        for g in self.cfg["goals"]:
            k = g["kind"]
            if k == "in":
                met = self._goal_in(data, g)
            elif k == "off":
                met = self._goal_off(data, g)
            elif k == "upright":
                met = self._goal_upright(data, g)
            elif k == "joint":
                q = float(data.qpos[self._jadr[g["joint"]]])
                met = [g["lo"] <= q <= g["hi"]]
            elif k == "erased":
                out.append((min(self._erased_frac() / g["frac"], 1.0), 1))
                continue
            else:
                raise ValueError(k)
            need = g["count"] or len(met)
            out.append((float(min(sum(met), need)), need))
        return out

    def goal_detail(self, data) -> list[dict]:
        """Per goal object whether it is met (debugging / replay reports)."""
        rows = []
        for g in self.cfg["goals"]:
            k = g["kind"]
            met = {"in": self._goal_in, "off": self._goal_off, "upright": self._goal_upright}.get(k)
            if met is None:
                continue
            for n, ok in zip(g["objects"], met(data, g)):
                rows.append({"goal": k, "object": n, "met": bool(ok), "tilt": _tilt_deg(data.xmat[self._body[n]])})
        return rows

    def _pinched(self, data, geoms) -> bool:
        """Both finger pads of one gripper touch the object (a pinch, not a push or a single-finger tip-over)."""
        if not hasattr(self, "_pads"):
            name = lambda g: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[g])) or ""
            self._pads = {}
            for g in self._fingers:
                n = name(g)
                arm = "left" if n.startswith("left_") else "right"
                side = "lf" if "_lf_" in n or "left_finger" in n else "rf"
                self._pads.setdefault((arm, side), set()).add(g)
        return any(_touching(data, geoms, self._pads.get((arm, "lf"), set())) and
                   _touching(data, geoms, self._pads.get((arm, "rf"), set())) for arm in ("left", "right"))

    # ------------------------------------------------------------------ subtask envs (instruction = enforced spec)
    def _pad_touch(self, data) -> dict:
        """(arm, finger) -> the objects that finger touches (one pass over the contacts)."""
        out = {("left", "lf"): set(), ("left", "rf"): set(), ("right", "lf"): set(), ("right", "rf"): set()}
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist > 1e-3:
                continue
            a, b = int(c.geom1), int(c.geom2)
            pa, pb = self._pad_of.get(a), self._pad_of.get(b)
            if (pa is None) == (pb is None):
                continue
            n = self._geom_obj.get(b if pa is not None else a)
            if n is not None:
                out[pa if pa is not None else pb].add(n)
        return out

    @staticmethod
    def _pin(touch, arm, name) -> bool:
        """Both finger pads of ``arm`` touch the object (a pinch, not a push or a single-finger tip-over)."""
        return name in touch[(arm, "lf")] and name in touch[(arm, "rf")]

    def _blocked(self, data, name, touch, arms) -> bool:
        """Does the object touch anything but the gripper(s) of ``arms``: fixed scene geometry, any other part of the
        robot, or another object that such a gripper does not pinch as well?"""
        own = self._geoms[name]
        grip = set().union(*(self._grip_geoms[a] for a in arms))
        for i in range(data.ncon):
            c = data.contact[i]
            if c.dist > 1e-3:
                continue
            a, b = int(c.geom1), int(c.geom2)
            if (a in own) == (b in own):
                continue
            other = b if a in own else a
            if other in grip:
                continue
            if other in self._static or other in self._robot_geoms:
                return True
            m = self._geom_obj.get(other)
            if m is not None and m != name and not any(self._pin(touch, arm, m) for arm in arms):
                return True
        return False

    def _zrange(self, data, geoms) -> tuple[float, float]:
        """Lowest and highest world z of the geoms (mesh vertices; box corners of other geom types)."""
        lo, hi = np.inf, -np.inf
        for g in geoms:
            pts = self._zcache.get(g)
            if pts is None:
                m = self.model
                if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
                    k = int(m.geom_dataid[g])
                    a, n = int(m.mesh_vertadr[k]), int(m.mesh_vertnum[k])
                    pts = np.asarray(m.mesh_vert[a:a + n], float)
                else:
                    c, h = np.asarray(m.geom_aabb[g][:3]), np.asarray(m.geom_aabb[g][3:])
                    pts = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * h + c
                self._zcache[g] = pts
            z = pts @ data.geom_xmat[g].reshape(3, 3)[2] + float(data.geom_xpos[g][2])
            lo, hi = min(lo, float(z.min())), max(hi, float(z.max()))
        return lo, hi

    def _solid(self, name) -> list[int]:
        """The object's collision geoms (its physical shape: a bag's walls, not its visual handles); all if none."""
        m = self.model
        gs = [g for g in self._geoms[name] if m.geom_contype[g] or m.geom_conaffinity[g]]
        return gs or sorted(self._geoms[name])

    def _out_of(self, data, name, containers) -> bool:
        """Is the object's lowest point above the highest point of every container in ``containers`` (collision
        geometry of both: the rim of a bag, not its handles)?"""
        if not containers:
            return True
        low = self._zrange(data, self._solid(name))[0]
        return all(low > self._zrange(data, self._solid(c))[1] for c in containers if c in self._geoms)

    def _clock(self, data, with_velocity) -> float:
        if with_velocity:
            return float(data.time)
        self._vclock += self.control_steps * float(self.model.opt.timestep)  # qpos only: one control step per call
        return self._vclock

    def _held_for(self, key, ok, t) -> float:
        """Seconds the condition has held without a break (false, or evaluations > max_gap_s apart, restart it)."""
        if not ok:
            self._hold.pop(key, None)
            return 0.0
        st = self._hold.get(key)
        st = (t, t) if (st is None or t < st[1] or t - st[1] > self.max_gap_s) else (st[0], t)
        self._hold[key] = st
        return st[1] - st[0]

    def _score_subtask(self, data, *, with_velocity):
        if self._spec is None:  # evaluator used without the env's reset (no randomization metadata): the state at hand
            spec = resolve_spec(self.model, data, self.task, self.variant, self.object, members=self.members,
                                group=self.group, seed=0, names=list(self._objs),
                                extents={n: o["extent"] for n, o in self._objs.items()})
            self.set_spec(dict(spec, source="first evaluation"), data=data)
        if self._ref is None:
            self._ref = self._ref_from_data(data)
        spec, ref = self._spec, self._ref
        t = self._clock(data, with_velocity)
        dt = 0.0 if self._t_prev is None else float(np.clip(t - self._t_prev, 0.0, self.max_gap_s))
        self._t_prev = t
        touch = self._pad_touch(data)
        arm = spec.get("arm")
        targets = [n for n in spec.get("targets", []) if n in self._body and n in ref]
        arms = (arm,) if arm else ("left", "right")
        pin = {n: any(self._pin(touch, a, n) for a in arms) for n in targets}
        lift = {n: float(data.xipos[self._body[n]][2] - ref[n][0][2]) for n in targets}
        valid = bool(spec.get("valid")) and bool(targets)
        # metrics that do not gate success
        if arm and any(self._pin(touch, "right" if arm == "left" else "left", n) for n in targets):
            self._wrong_arm = True
        if not self._wrong_object:
            for n in self._task_goal_objects:
                if n not in targets and n in ref and (self._pin(touch, "left", n) or self._pin(touch, "right", n)) \
                        and float(data.xipos[self._body[n]][2] - ref[n][0][2]) >= self.lift_m:
                    self._wrong_object = n
                    break
        disturbed = 0
        for n in self._free:
            if n in targets or n not in ref:
                continue
            b = self._body[n]
            if (float(np.linalg.norm(data.xipos[b] - ref[n][0])) > self.disturbed_m
                    or float(data.xmat[b].reshape(3, 3)[:, 2] @ ref[n][1]) < self.disturbed_cos):
                disturbed += 1
        self._disturbed_max = max(self._disturbed_max, disturbed)
        if self.variant == "pick":
            reward, success, info = self._score_pick(data, t, touch, targets, pin, lift, valid)
        else:
            reward, success, info = self._score_obj(data, t, dt, touch, targets, pin, lift, valid, with_velocity)
        info.update({"wrong_arm": bool(self._wrong_arm), "wrong_object": bool(self._wrong_object),
                     "wrong_object_name": self._wrong_object, "disturbed": int(disturbed),
                     "disturbed_max": int(self._disturbed_max), **spec_metrics(spec)})
        return reward, success, info

    def _score_pick(self, data, t, touch, targets, pin, lift, valid):
        spec, ref = self._spec, self._ref
        arm = spec.get("arm")
        arms = (arm,) if arm else ("left", "right")
        lift_free = spec.get("lift_free", True) is not False  # lifted off: touching nothing but the designated gripper
        out_of = [c for c in spec.get("lift_out_of") or [] if c in self._geoms]
        progress, hold, lifted, free, outside = 0.0, 0.0, False, False, False
        for n in targets:
            up = pin[n] and lift[n] >= self.lift_m
            clear = up and not (lift_free and self._blocked(data, n, touch, arms))
            out = clear and self._out_of(data, n, out_of)
            lifted, free, outside = lifted or up, free or clear, outside or out
            clear = out
            hold = max(hold, self._held_for(n, clear, t))
            if pin[n]:
                progress = max(progress, 0.3 + 0.7 * float(np.clip(lift[n] / self.lift_m, 0.0, 1.0)))
        success = bool(valid and hold >= self.hold_s - 1e-9)
        # failed, as before: the target instance pushed away without a grasp (either gripper), or off the table
        n0 = self._obj
        failed, moved, lift0 = False, 0.0, 0.0
        if n0 in ref:
            p = data.xipos[self._body[n0]]
            moved, lift0 = float(np.linalg.norm(p[:2] - ref[n0][0][:2])), float(p[2] - ref[n0][0][2])
            grasped = self._pin(touch, "left", n0) or self._pin(touch, "right", n0)
            failed = bool((moved > 0.08 and not grasped and lift0 < self.lift_m) or p[2] < 0.70)
        touched = any(pin.values())
        best = max([lift[n] for n in targets if pin[n]], default=lift0)
        reward = 1.0 if success else (float(min(progress, 0.99)) if valid else 0.0)
        return reward, success, {"items_done": float(success), "items": 1, "touched": touched, "lift_m": best,
                                 "pushed_m": moved, "failed": failed, "held": touched, "at_rest": True,
                                 "constraints_ok": True, "hold_s": float(hold), "lifted": bool(lifted),
                                 "lifted_free": bool(free), "lifted_out": bool(outside)}

    def _score_obj(self, data, t, dt, touch, targets, pin, lift, valid, with_velocity):
        spec = self._spec
        met = {n: False for n in targets}
        for g in self.cfg["goals"]:
            fn = {"in": self._goal_in, "off": self._goal_off, "upright": self._goal_upright}[g["kind"]]
            for n, ok in zip(g["objects"], fn(data, g)):
                if n in met:
                    met[n] = met[n] or bool(ok)
        fingers = set().union(*touch.values())
        done = held = False
        at_rest, success, pinched_s, hold = True, False, 0.0, 0.0
        for n in targets:
            if spec.get("arm"):  # grasp, then place: pinch time counts while the goal is not met yet
                if pin[n] and not met[n]:
                    self._pinch_t[n] = self._pinch_t.get(n, 0.0) + dt
                grasped = self._pinch_t.get(n, 0.0) >= self.pinch_s - 1e-9
                pinched_s = max(pinched_s, self._pinch_t.get(n, 0.0))
            else:
                grasped = True
            if not (met[n] and grasped):
                self._held_for(n, False, t)
                continue
            done = True
            h = n in fingers
            d = self._dof.get(n)
            rest = (float(np.linalg.norm(data.qvel[d: d + 3])) <= self.rest_speed) if (with_velocity and d is not None) else True
            ok = bool(valid and not h and rest)  # placed, released, at rest: the goal hold runs
            hold = max(hold, self._held_for(n, ok, t))
            if ok:
                success = True
            else:
                held, at_rest = held or h, at_rest and rest
        if success:
            held, at_rest = False, True
        success = bool(valid and success and hold >= self.hold_s - 1e-9)
        reward = 1.0 if success else (0.99 if (done and valid) else 0.0)
        return reward, success, {"items_done": float(done), "items": 1, "constraints_ok": True, "held": bool(held),
                                 "at_rest": bool(at_rest), "goal_met": bool(any(met.values())),
                                 "touched": any(pin.values()), "pinch_s": float(pinched_s), "hold_s": float(hold),
                                 "lift_m": max(lift.values(), default=0.0)}

    SIM_FAULT_GAIN, SIM_FAULT_LIN = 1.5, 20.0  # m/s (fallback without the sim guard)

    def _sim_fault(self, data) -> bool:
        """Latched when contacts blew up (a thin shaft through a container floor fired out; a bogus box-box contact
        threw a tin). Progress then stays where it was: no reward from it. The physics-step monitor
        (abc_sim/sim_guard.py, polled here once per control step) decides: an object that neither the robot nor
        anything the robot touches moves gains > 1.5 m/s inside one control step, or such an object wedged with
        > 5 kN gains > 0.5 m/s or > 20 rad/s, or anything exceeds 20 m/s (with a guard that only watches: also a
        bogus contact). A wedge that moves nothing, and real overlaps that survive the guard's nudges, are reported
        by the guard, not faults by themselves. Without the guard
        (R2G_SIM_GUARD=0) the gain rule on what is visible at control steps: a
        free object > 1.5 m/s faster than at the last evaluation with no robot contact now or then.
        (The earlier rule, a goal object above 4 m/s or 60 rad/s at a control step, fired on objects the arm swung
        or threw (4 - 11 m/s in 9 recorded episodes, no contact fault) and missed 18 of 24 episodes with one: only
        goal objects, and most launches are slower than 4 m/s.)"""
        if getattr(self, "_fault", None) is None:
            self._fault, self._fault_score = False, None
        if not self._fault:
            guard = sim_guard.lookup(data)
            if guard is not None:
                reason = guard.poll()["fault"]
            else:
                reason = self._sim_fault_sampled(data)
            if reason:
                self._fault, self._fault_reason = True, str(reason)
        return self._fault

    def _sim_fault_sampled(self, data):
        m = self.model
        if not hasattr(self, "_free"):
            robot = {g for g in range(m.ngeom)
                     if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.body_rootid[m.geom_bodyid[g]])) or "").startswith(("left_", "right_"))}
            self._free = [(n, int(m.jnt_dofadr[m.body_jntadr[b]]), self._geoms[n]) for n, b in self._body.items()
                          if m.body_jntnum[b] and m.jnt_type[m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE]
            self._robot_geoms = robot
        state = {}
        for n, dof, geoms in self._free:
            state[n] = (float(np.linalg.norm(data.qvel[dof:dof + 3])), _touching(data, geoms, self._robot_geoms))
        prev, self._free_prev = getattr(self, "_free_prev", None), state
        for n, (v, touched) in state.items():
            if v > self.SIM_FAULT_LIN:
                return f"speed: {n} at {v:.1f} m/s"
            if prev is not None and n in prev and not touched and not prev[n][1] and v - prev[n][0] > self.SIM_FAULT_GAIN:
                return f"launch: {n} gained {v - prev[n][0]:.1f} m/s in one control step with no robot contact"
        return None

    def _score(self, data, *, with_velocity):
        if not self._ready:
            return 0.0, False, {}
        if with_velocity and self._sim_fault(data):
            if self._fault_score is None:
                self._fault_score = (float(getattr(self, "_last_reward", 0.0)), {"items_done": 0.0, "items": 0})
            r, info = self._fault_score
            return r, False, {**info, "sim_fault": True, "sim_fault_reason": getattr(self, "_fault_reason", ""),
                              "failed": True, "held": False, "at_rest": False, "constraints_ok": True}
        out = self._score_inner(data, with_velocity=with_velocity)
        self._last_reward = out[0]
        return out

    def _score_inner(self, data, *, with_velocity):
        if self.variant in self.SUBTASKS:
            return self._score_subtask(data, with_velocity=with_velocity)
        goals = self._evaluate_goals(data)
        if self._baseline is None:
            self._baseline = [m for m, _ in goals]
        cons_ok = True
        for kind, adr, mx in self._cons:
            if kind == "joints_max" and max(float(data.qpos[a]) for a in adr) > mx:
                cons_ok = False
        for c in self.cfg.get("constraints", []):
            if c["kind"] == "in_box":
                n_in = sum(self._in_box(data, c["container"], o, above=c["above"]) for o in c["objects"])
                cons_ok = cons_ok and n_in >= c["count"]
        held = any(_touching(data, self._geoms[n], self._fingers) for n in self._goal_objects)
        speeds = [float(np.linalg.norm(data.qvel[d: d + 3])) for n, d in self._dof.items() if d is not None]
        at_rest = (max(speeds) <= self.rest_speed) if (with_velocity and speeds) else True
        done = all(m >= need - 1e-9 for m, need in goals)
        success = bool(done and cons_ok and not held and at_rest)
        todo = sum(need - b for (_, need), b in zip(goals, self._baseline))
        gained = sum(max(m - b, 0.0) for (m, _), b in zip(goals, self._baseline))
        progress = (gained / todo) if todo > 1e-9 else float(done)
        reward = 1.0 if success else float(min(progress, 0.99))
        return reward, success, {"items_done": float(sum(m for m, _ in goals)), "items": int(sum(n for _, n in goals)),
                                 "constraints_ok": cons_ok, "held": held, "at_rest": at_rest}


__all__ = ["Real2GymEvaluator"]
