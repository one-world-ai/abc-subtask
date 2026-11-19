"""Physics-step guard and sim-fault monitor for contact blow-ups (used by the Real2Gym envs).

Two things happen around every ``mj_step`` (``SimGuard.step`` replaces it in ``MuJoCoYAMEnv.advance_physics``):

1. Bogus contacts are removed. MuJoCo's collision routines can return a contact far deeper than the true overlap:
   the analytic box-box routine (recorded: the lid of the chocolate tin slid on the table until its edge touched a
   corner of the tin, true overlap 1.5 um, reported 189 mm; the solver pushed the two apart with 290 N and lid, tin
   and liners flew off at 8 and 3 m/s), the convex routine on cylinders standing side by side (cans in a bag
   23 - 54 mm, cookies 13 mm; launched at 1.5 - 5.7 m/s) and the plane-cylinder routine (the disc floor of the
   lunch box at rest on the table: 11 mm). They are knife-edge numerical cases. Penetration cannot grow faster
   than the bodies move, so: when a step used a contact (not with the robot) deeper than ``depth`` (6 mm) whose
   body pair was more than ``jump`` (5 mm) + 2 x (relative speed at the contact x time step) shallower one step
   earlier, the step is undone (``mj_setState``), the lighter body is moved by micrometres and the step is taken
   again (``_repair``). A contact that survives that is a real overlap and is left alone.

2. Launches are detected. Per free object the monitor keeps, between two polls (the evaluator polls once per control
   step), the peak speed of its centre of mass, whether any robot geom touched it, which free objects it touched and
   the largest contact force on it that does not involve the robot. An object is "driven" in a window when a robot
   geom touched it, or it touched an object a robot geom touched, in that window or (robot contact) the one before.
   ``poll`` reports a fault when
     - an object that is not driven gained more than ``launch_dv`` of speed inside one control step (gravity gives
       0.33 m/s per step; what the robot does not move can only slow down; a bowl hit by a dish rack the arm swings
       is the robot's doing),
     - an object moves faster than ``max_lin`` (20 m/s, a sanity bound: arm swings throw held objects at up to
       11 m/s in recorded rollouts, so speed alone does not tell a blow-up from the robot), or
     - an object that is not driven carries a contact without the robot above ``max_force`` (5 kN, in this control
       step or the one before: a wedge under the pyramidal friction cone; a dropped screwdriver between bin walls:
       6 - 790 kN) and in this control step gains more than ``wedge_dv`` (0.5 m/s) or ``wedge_dw`` (20 rad/s): the
       wedge spins or pushes it (in the drop test the launch rule alone flagged 13 of 26 such drops from 5 - 20 cm:
       the centre of mass of a wedged tool often gains less than 1.5 m/s per control step while it spins).
   A wedge that moves nothing is reported (``warning``, ``warnings``), not a fault: the tray dropped into the bin in
   the release's mixed-dishes demo sits wedged against the bin wall with 25 kN and nothing moves (as a fault this
   failed that demo); in 1641 recorded rollouts re-simulated under these defaults no 5 kN contact occurred.
   The numbers come from re-simulating 1641 recorded rollouts and the 36 release demos (tools/
   real2gym_blowup_scan.py, real2gym_physics_replay.py --speeds). ``fix=False`` only watches (a bogus contact is then
   itself reported as a fault). Cost: 10 - 40 us per physics step (3 - 20 % of a step, by scene).

``lookup(data)`` returns the guard stepping an ``MjData`` (the evaluator only sees model and data).
"""
from __future__ import annotations

import weakref

import mujoco
import numpy as np

_GUARDS: "weakref.WeakValueDictionary[int, SimGuard]" = weakref.WeakValueDictionary()
_NOKEYS, _NODEPTH = np.zeros(0, np.int64), np.zeros(0)
_TURN_AXIS = np.array([0.6, 0.48, 0.64])  # body-frame axis of the micro-radian turn in a nudge
_SIG = mujoco.mjtState.mjSTATE_INTEGRATION


def lookup(data: mujoco.MjData) -> "SimGuard | None":
    g = _GUARDS.get(id(data))
    return g if g is not None and g.data is data else None


class SimGuard:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, *, depth: float = 0.006, jump: float = 0.005,
                 track: float = 0.004, nudge: float = 2e-6, retries: int = 3, launch_dv: float = 1.5,
                 max_lin: float = 20.0, max_force: float = 5000.0, wedge_dv: float = 0.5, wedge_dw: float = 20.0,
                 fix: bool = True) -> None:
        self.model, self.data = model, data
        self.depth, self.jump, self.track, self.nudge, self.retries, self.fix = depth, jump, track, nudge, retries, fix
        self._vel6 = np.zeros(6)
        self.launch_dv, self.max_lin, self.max_force = launch_dv, max_lin, max_force
        self.wedge_dv, self.wedge_dw = wedge_dv, wedge_dw
        m = model
        self._state = np.zeros(mujoco.mj_stateSize(m, _SIG))
        name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)) or ""
        root = np.asarray(m.body_rootid).copy()
        self._groot = root[m.geom_bodyid]
        # free objects (everything on a free joint that is not part of the robot)
        self.names, body, dof, qadr = [], [], [], []
        for j in range(m.njnt):
            b = int(m.jnt_bodyid[j])
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and not name(b).startswith(("left_", "right_")):
                self.names.append(name(b)); body.append(b); dof.append(int(m.jnt_dofadr[j])); qadr.append(int(m.jnt_qposadr[j]))
        self.n = len(body)
        self._body = np.asarray(body, int)
        obj_of_root = -np.ones(m.nbody, int)
        for i, b in enumerate(body):
            obj_of_root[b] = i
        self._gobj = obj_of_root[self._groot]  # geom -> free object index or -1
        self._grobot = np.array([name(root[b]).startswith(("left_", "right_")) for b in range(m.nbody)])[m.geom_bodyid]
        self._qadr = np.asarray(qadr, int)
        self._mass = np.array([m.body_subtreemass[b] for b in body]) if self.n else np.zeros(0)
        self._pyramidal = m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
        self.removed = 0  # bogus contacts removed by a nudge
        self.unresolved = 0  # deep contacts that survived the nudges (real overlaps, left alone)
        self.exempt = {"launch": 0, "force": 0}  # control steps where only the 'driven' rule kept a launch / wedge from being flagged
        self.warning: str | None = None  # first wedge force (not a fault)
        self.warnings = 0  # control steps with one
        self.events: list[dict] = []
        self.fault: str | None = None
        self._quat, self._quat2 = np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(4)
        self._prev = np.zeros(m.nbody * m.nbody)  # deepest penetration per body pair in the previous step (> track)
        self._prev_keys = _NOKEYS
        self._primed = False  # False until one step was seen (and after the state was set from outside)
        self._nstep = 0
        self._t_after: float | None = None  # sim time after the last guarded step
        self._reset_window()
        self._lin0, self._ang0 = self._lin(), self._ang()
        self._touch_prev = np.zeros(self.n, bool)
        self._fobj_prev, self._fobj_pair_prev = np.zeros(self.n), {}
        self._poll_time = None
        self._poll_out: dict = {}
        _GUARDS[id(data)] = self

    # ------------------------------------------------------------------ stepping
    def _lin(self) -> np.ndarray:
        """Speed of each free object's centre of mass (m/s): ``cvel`` of a free root body is taken at its subtree's
        centre of mass. (The free joint's own linear velocity is that of the body origin: on a long tool with the
        origin at one end it jumps by omega x r when the other end lands.) ``cvel`` is from the last forward pass,
        one physics step old after ``mj_step``: fine for a peak over a control step."""
        if not self.n:
            return np.zeros(0)
        v = self.data.cvel[self._body, 3:]
        return np.sqrt((v * v).sum(1))

    def _ang(self) -> np.ndarray:
        """Angular speed of each free object (rad/s), from ``cvel`` like ``_lin``."""
        if not self.n:
            return np.zeros(0)
        w = self.data.cvel[self._body, :3]
        return np.sqrt((w * w).sum(1))

    def _reset_window(self) -> None:
        self._peak = np.zeros(self.n)
        self._peak_ang = np.zeros(self.n)
        self._touch = np.zeros(self.n, bool)
        self._adj = np.zeros((self.n, self.n), bool)  # free objects that touched each other
        self._fmax = 0.0  # largest contact force without the robot (any object; reported, not a fault by itself)
        self._fobj = np.zeros(self.n)  # per free object: largest such force above max_force on it
        self._fobj_pair: dict[int, tuple[int, int]] = {}

    def _point_speed(self, d, i: int) -> float:
        """Relative speed (m/s) of the two bodies of contact i at the contact point."""
        m, c = self.model, d.contact[i]
        v = []
        for g in c.geom:
            b = int(m.geom_bodyid[int(g)])
            if b == 0:
                v.append(np.zeros(3))
                continue
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, self._vel6, 0)
            v.append(self._vel6[3:] + np.cross(self._vel6[:3], c.pos - d.xpos[b]))
        return float(np.linalg.norm(v[0] - v[1]))

    def _scan(self, d):
        """Contacts of the step just taken -> (index of a bogus contact or None, body-pair keys of the contacts
        deeper than ``track``, their depths). Bogus: deeper than ``depth`` and deeper than its body pair was one
        step earlier by more than ``jump`` plus twice what the two bodies' relative speed at the contact covers in
        a step (penetration cannot grow faster than they move)."""
        if not d.ncon:
            return None, _NOKEYS, _NODEPTH
        dist = d.contact.dist
        if dist.min() > -self.track:
            return None, _NOKEYS, _NODEPTH
        g = d.contact.geom
        # contacts with the robot are left alone: its soft finger pads sink up to a centimetre into what they squeeze
        # and a thin pad box wholly inside an object box loses its contacts for a few steps, then gets them back
        idx = np.nonzero((dist < -self.track) & ~(self._grobot[g[:, 0]] | self._grobot[g[:, 1]]))[0]
        if not len(idx):
            return None, _NOKEYS, _NODEPTH
        roots = self._groot[g[idx]]
        keys = roots[:, 0] * self.model.nbody + roots[:, 1]
        dep = -dist[idx]
        bad = None
        if self._primed:
            jump = dep - np.maximum(self._prev[keys], self.track)  # _prev: deepest per body pair one step earlier
            cand = np.nonzero((dep > self.depth) & (jump > self.jump))[0]
            if len(cand):  # rare: look closer
                dt = float(self.model.opt.timestep)
                for j in cand[np.argsort(-jump[cand])]:
                    if jump[j] > self.jump + 2.0 * self._point_speed(d, int(idx[j])) * dt:
                        bad = int(idx[j])
                        break
        return bad, keys, dep

    def _accept(self, keys, dep) -> None:
        """The step stands: its deep contacts become 'one step earlier'."""
        self._prev[self._prev_keys] = 0.0
        if len(keys):
            np.maximum.at(self._prev, keys, dep)
        self._prev_keys = keys
        self._primed = True

    def _free_side(self, c):
        """Of the two geoms of contact c the one on the lighter free object -> (object index, geom id, sign of the
        contact normal that moves it away from the other) or None if neither is on a free object."""
        g1, g2 = int(c.geom[0]), int(c.geom[1])
        cand = [(self._mass[o], o, g, sgn) for o, g, sgn in ((int(self._gobj[g1]), g1, -1.0), (int(self._gobj[g2]), g2, 1.0)) if o >= 0]
        return min(cand)[1:] if cand else None

    def step(self, m: mujoco.MjModel, d: mujoco.MjData) -> None:
        if self._t_after is not None and abs(float(d.time) - self._t_after) > 1e-9:
            self._primed = False  # the state was set from outside since the last step (reset, start bank): start afresh
            self._reset_window()
            self._lin0, self._ang0 = self._lin(), self._ang()
            self._touch_prev = np.zeros(self.n, bool)
            self._fobj_prev, self._fobj_pair_prev = np.zeros(self.n), {}
        if self.fix:
            mujoco.mj_getState(m, d, self._state, _SIG)
        mujoco.mj_step(m, d)
        bad, keys, dep = self._scan(d)
        if bad is not None:
            keys, dep = self._repair(m, d, bad, keys, dep)
        self._accept(keys, dep)
        self._t_after = float(d.time)
        self._monitor(m, d)

    def _repair(self, m, d, bad, keys, dep):
        """The step just taken used a contact deeper than motion allows. Undo it, move the lighter free body of the
        pair by micrometres (``nudge`` x 1, 2, 3 along the contact normal, with a micro-radian turn) and take the
        step again: a collision-routine artefact (box-box, cylinder-cylinder, a flat cylinder on the table plane) is
        gone after that. If the contact survives the nudges it is a real overlap (a part that got inside another
        while its contacts were missing: a thin box wholly inside a box has none): the step stands as first taken
        and the contact is left to push the part out; ``unresolved`` counts these, the launch and force rules watch
        what follows."""
        c = d.contact[bad]
        ev = {"time": float(d.time), "depth_mm": round(-float(c.dist) * 1000, 1),
              "geoms": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(x)) or f"g{int(x)}" for x in c.geom], "removed": False}
        if self.fix:
            side = self._free_side(c)
            tried = False
            for k in range(self.retries if side is not None else 0):
                tried = True
                o, _, sgn = side
                normal = np.array(c.frame[:3])  # from geom 1 to geom 2
                mujoco.mj_setState(m, d, self._state, _SIG)
                a = self._qadr[o]
                d.qpos[a:a + 3] += sgn * normal * self.nudge * (k + 1)
                mujoco.mju_quatIntegrate(self._quat, _TURN_AXIS, 0.5e-6 * (k + 1))
                mujoco.mju_mulQuat(self._quat2, d.qpos[a + 3:a + 7], self._quat)
                d.qpos[a + 3:a + 7] = self._quat2
                self._quat[:] = (1.0, 0.0, 0.0, 0.0)
                mujoco.mj_step(m, d)
                bad, keys, dep = self._scan(d)
                if bad is None:
                    ev["removed"], ev["nudge_um"] = True, round(self.nudge * (k + 1) * 1e6, 1)
                    break
                c = d.contact[bad]
                side = self._free_side(c)
                if side is None:
                    break
            if not ev["removed"] and tried:  # back to the step as first taken
                mujoco.mj_setState(m, d, self._state, _SIG)
                mujoco.mj_step(m, d)
                _, keys, dep = self._scan(d)
        if ev["removed"]:
            self.removed += 1
        else:
            self.unresolved += 1
            if not self.fix:  # only watching: nothing took the contact away
                self.fault = self.fault or f"bogus contact {ev['geoms'][0]} / {ev['geoms'][1]} {ev['depth_mm']} mm"
        if len(self.events) < 50:
            self.events.append(ev)
        return keys, dep

    def _monitor(self, m, d) -> None:
        if not self.n:
            return
        self._nstep += 1
        # some constraint row above max_force / 100: a contact may carry > max_force / 10 (<= 10 rows per contact)
        big = d.nefc and float(d.efc_force[:d.nefc].max()) * 10.0 > 0.1 * self.max_force
        # speeds and robot contact on alternate steps (both last longer than 2 steps); both on a step with a large
        # constraint force (a wedge: the launch it makes must not fall between the samples)
        if self._nstep & 1 or big:
            np.maximum(self._peak, self._lin(), out=self._peak)
            np.maximum(self._peak_ang, self._ang(), out=self._peak_ang)
            if not big:
                return
        if not d.ncon:
            return
        g = d.contact.geom
        rob = self._grobot[g]
        near = d.contact.dist < 1e-3
        sel = (rob[:, 0] != rob[:, 1]) & near  # robot against something else, touching
        if sel.any():
            o = self._gobj[g[sel]].ravel()
            self._touch[o[o >= 0]] = True
        oo = self._gobj[g]
        sel = near & (oo[:, 0] >= 0) & (oo[:, 1] >= 0) & (oo[:, 0] != oo[:, 1])  # two free objects touching
        if sel.any():
            self._adj[oo[sel, 0], oo[sel, 1]] = True
        # largest normal force of a contact without the robot (exact only when some constraint row is large)
        if big:
            adr = d.contact.efc_address
            ok = (adr >= 0) & ~(rob[:, 0] | rob[:, 1])
            if ok.any():
                if self._pyramidal:
                    dim = d.contact.dim[ok]
                    nrow = np.where(dim > 1, 2 * (dim - 1), 1)
                    cs = np.concatenate(([0.0], np.cumsum(d.efc_force[:d.nefc])))
                    fn = cs[adr[ok] + nrow] - cs[adr[ok]]
                else:
                    fn = d.efc_force[adr[ok]]
                k = int(np.argmax(fn))
                if fn[k] > self._fmax:
                    self._fmax = float(fn[k])
                hi = np.nonzero(fn > self.max_force)[0]
                if len(hi):  # rare: the free objects that carry it
                    for j, f in zip(np.nonzero(ok)[0][hi], fn[hi]):
                        for gg in g[j]:
                            o = int(self._gobj[gg])
                            if o >= 0 and f > self._fobj[o]:
                                self._fobj[o] = float(f)
                                self._fobj_pair[o] = (int(g[j, 0]), int(g[j, 1]))

    # ------------------------------------------------------------------ evaluator side
    def poll(self) -> dict:
        """Called by the evaluator once per control step: the window since the last poll. Returns
        {"fault": reason or None, "peak_lin", "launch", ...}; repeated polls at the same sim time return the same."""
        d = self.data
        if self._poll_time == float(d.time):
            return self._poll_out
        lin, ang = self._lin(), self._ang()
        np.maximum(self._peak, lin, out=self._peak)  # the window's last state counts too
        np.maximum(self._peak_ang, ang, out=self._peak_ang)
        out = {"fault": self.fault, "removed": self.removed, "unresolved": self.unresolved}
        if self.n and self.fault is None:
            gain = self._peak - self._lin0
            touched = self._touch | self._touch_prev  # in the robot's hands
            driven = touched | ((self._adj | self._adj.T) & touched[None, :]).any(1)  # or hit by something that is
            free = ~driven
            o = int(np.argmax(np.where(free, gain, -1.0)))
            fw = np.maximum(self._fobj, self._fobj_prev)  # wedge force in this control step or the one before
            hot = free & (fw > self.max_force)
            gain_ang = self._peak_ang - self._ang0
            moved = hot & ((gain > self.wedge_dv) | (gain_ang > self.wedge_dw))
            pairs = {**self._fobj_pair_prev, **self._fobj_pair}
            gname = lambda o: [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, x) or f"g{x}" for x in pairs.get(o, (-1, -1))]
            if free[o] and gain[o] > self.launch_dv:
                self.fault = f"launch: {self.names[o]} gained {gain[o]:.1f} m/s in one control step with no robot contact"
            elif self._peak.max() > self.max_lin:
                o = int(np.argmax(self._peak))
                self.fault = f"speed: {self.names[o]} at {self._peak[o]:.1f} m/s"
            elif moved.any():
                o = int(np.argmax(np.where(moved, fw, -1.0)))
                gn = gname(o)
                self.fault = (f"wedge: {self.names[o]} with {fw[o]:.0f} N ({gn[0]} / {gn[1]}) and no robot contact gained "
                              f"{gain[o]:.1f} m/s, {gain_ang[o]:.0f} rad/s in one control step")
            elif (~touched & driven & (gain > self.launch_dv)).any():  # only the contact chain to the robot kept it
                self.exempt["launch"] += 1
            if hot.any() and self.fault is None:  # a wedge that moved nothing: reported, not a fault
                self.warnings += 1
                if self.warning is None:
                    o = int(np.argmax(np.where(hot, fw, -1.0)))
                    gn = gname(o)
                    self.warning = (f"force: {fw[o]:.0f} N on {self.names[o]} ({gn[0]} / {gn[1]}) with no robot "
                                    f"contact at t={float(d.time):.3f}")
            if (driven & (self._fobj > self.max_force)).any():
                self.exempt["force"] += 1
            out["fault"] = self.fault
        out["peak_lin"] = float(self._peak.max()) if self.n else 0.0
        out["peak_force"] = self._fmax
        out["exempt"] = dict(self.exempt)
        out["warning"], out["warnings"] = self.warning, self.warnings
        self._touch_prev = self._touch
        self._lin0, self._ang0 = lin, ang
        self._fobj_prev, self._fobj_pair_prev = self._fobj, self._fobj_pair
        self._reset_window()
        self._poll_time, self._poll_out = float(d.time), out
        return out

    def reset(self) -> None:
        """After the env state was set from outside (reset, start bank): forget the previous step and window."""
        self._primed = False
        self._t_after = None
        self.fault = None
        self.removed = self.unresolved = 0
        self.exempt = {"launch": 0, "force": 0}
        self.warning, self.warnings = None, 0
        self.events = []
        self._reset_window()
        self._lin0, self._ang0 = self._lin(), self._ang()
        self._touch_prev = np.zeros(self.n, bool)
        self._fobj_prev, self._fobj_pair_prev = np.zeros(self.n), {}
        self._poll_time = None


__all__ = ["SimGuard", "lookup"]
