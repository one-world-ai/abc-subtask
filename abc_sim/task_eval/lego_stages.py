"""Approved LEGO reach/pick/handoff/carry/drop stages, measured from live CPU simulation state.

Handoff (stage table amendment 3, user-approved 2026-11-14): a block is *cross-side* when its rest spot and its
matching bin's rest spot are on opposite sides of the table (y sign; the same rule the scoped drop used). The
receiver of block k is the arm on its bin's side, the giver the other arm. Cross-side blocks go
reach -> pick -> handoff -> carry -> drop; same-side blocks keep reach -> pick -> carry -> drop. The handoff tests are
the task-independent ``handoff_stage`` on this scene's per-block arrays (com, lift, f_give, f_recv, touch_give,
touch_recv, gap_give, env_touch) with the LEGO region ``HANDOFF``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import mujoco
import numpy as np

from abc_sim.task_eval import handoff_stage as hs
from abc_sim.task_eval.lego_blocks import LegoBlocksSortingEvaluator
from abc_sim.task_eval.stagekit.scene import ObjectScene, free_joints_matching, geoms_of, subtree
from abc_sim.task_eval.stagekit.tracker import StageTracker

COLORS = ("red", "yellow", "blue")
STAGES = ("reach", "pick", "handoff", "carry", "drop", "done")
R, P, H, C, D, DONE = range(6)
FALLEN = len(STAGES)
LABELS = STAGES + ("fallen",)
STAGE_PROMPTS = {
    "reach": "lego sorting subtask: move a gripper to the target block",
    "pick": "lego sorting subtask: grasp the target block and lift it",
    "handoff": "lego sorting subtask: hand the target block to the arm on its bin's side",
    "carry": "lego sorting subtask: carry the target block over its matching bin",
    "drop": "lego sorting subtask: release the target block into its matching bin",
}


# Handoff region (block CoM, world). Both arms reach it (bases at x 0.252, y +-0.31); x <= 0.60 keeps it clear of the
# middle bin's opening (bins at x 0.70, openings >= 0.63), so a block in it is never already over a bin (the
# carry goal). z 0.80-1.10: 4.5-35 cm above the tabletop (block rest CoM 0.757). |y| <= 0.10 as inhand.
HANDOFF = hs.HandoffConfig(region_lo=(0.40, -0.10, 0.80), region_hi=(0.60, 0.10, 1.10), span=1.55)


@dataclass(frozen=True)
class StageConfig:
    reach_m: float = 0.05
    reach_exit_m: float = 0.07
    lift_m: float = 0.03
    push_m: float = 0.03
    fallen_m: float = 0.05
    clearance_m: float = 0.005
    opening_exit_m: float = 0.02
    bin_moved_m: float = 0.05
    bin_tilt_rad: float = np.deg2rad(20)
    at_rest_speed: float = 0.02
    at_rest_angular_speed: float = 0.2
    hold: int = 5
    handoff: hs.HandoffConfig = HANDOFF
    region_shaping_m: float = 0.30  # handoff progress: -min(distance to the region, this) + the shared progress
    eps: dict = field(default_factory=lambda: dict(reach=.005, pick=.05, handoff=.02, carry=.005, drop=.005))
    # handoff span = shared 1.25 + region approach 0.30
    span: dict = field(default_factory=lambda: dict(reach=.30, pick=1., handoff=1.55, carry=.40, drop=.10))


def _ray_entry(normals, offsets, origin, direction):
    a, b = normals @ direction, normals @ origin + offsets
    parallel = np.abs(a) < 1e-10
    if np.any(parallel & (b > 1e-7)):
        return np.inf
    ratio = -b / np.where(parallel, 1., a)
    enter = max(0., np.max(np.where(a < -1e-10, ratio, -np.inf)))
    leave = np.min(np.where(a > 1e-10, ratio, np.inf))
    return enter if enter <= leave + 1e-7 else np.inf


class LegoScene(ObjectScene):
    def __init__(self, model, cfg=None):
        super().__init__(model, free_joints_matching(model, r"(?:red|yellow|blue)_lego_(\d+)_joint"))
        self.cfg = cfg or StageConfig()
        self.geometry = LegoBlocksSortingEvaluator(model=model, spec=None)
        self.active = np.ones(len(self.names), dtype=bool)
        self.colors = np.array([COLORS.index(name.split('_')[0]) for name in self.names])
        self.bin_adrs = np.array([self.geometry._bin_qpos_addrs[c] for c in COLORS])
        self.bin_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{c}_lego_sorting_bin") for c in COLORS]
        self.bin_geoms = [set(geoms_of(model, subtree(model, body))) for body in self.bin_bodies]
        self.bin_owner = np.full(model.ngeom, -1, dtype=np.int64)
        for c, gs in enumerate(self.bin_geoms):
            self.bin_owner[list(gs)] = c
        self.block_geoms = []
        for name, body in zip(self.names, self.bodies):
            gs = [g for g in geoms_of(model, subtree(model, body))
                  if model.geom_contype[g] or model.geom_conaffinity[g]]
            if len(gs) != 1 or model.geom_type[gs[0]] != mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError(f"Expected the measured collision box of {name}")
            self.block_geoms.append(gs[0])
        self.box_corners = [np.array(list(product((-1., 1.), repeat=3))) * model.geom_size[g] for g in self.block_geoms]
        self.reaches = np.array([np.linalg.norm(model.geom_size[g]) for g in self.block_geoms])
        self.arm_of = np.full(model.ngeom, -1, dtype=np.int64)  # geom -> 0 left arm, 1 right arm (whole subtree)
        for a, root in enumerate(('left_arm', 'right_arm')):
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
            if b < 0:
                raise ValueError(f'{root} not found')
            self.arm_of[geoms_of(model, subtree(model, b))] = a
        self.link6 = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'{arm}_link_6') for arm in ('left', 'right')]
        scratch = mujoco.MjData(model)
        for arm in ('left', 'right'):
            for side in ('left', 'right'):
                joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'{arm}_{side}_finger')
                scratch.qpos[model.jnt_qposadr[joint]] = 0.
        mujoco.mj_forward(model, scratch)
        self.pad_link6 = []
        for a, arm in enumerate(('left', 'right')):
            pads = [g for g in range(model.ngeom) if self.owner[g, 0] == 1 and self.owner[g, 1]//2 == a
                    and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
            if not pads:
                raise ValueError(f'No collision pad spheres for {arm}')
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'{arm}_link_6')
            self.pad_link6.append((scratch.geom_xpos[pads].mean(axis=0)-scratch.xpos[body])
                                  @ scratch.xmat[body].reshape(3, 3))
        self.pad_link6 = np.array(self.pad_link6)
        self.rim_local = []
        self.opening_lo, self.opening_hi = [], []
        for c, color in enumerate(COLORS):
            b, a = self.bin_bodies[c], self.bin_adrs[c]
            rotation = scratch.xmat[b].reshape(3, 3)
            vertices = []
            for g in self.bin_geoms[c]:
                if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                    continue
                mesh = int(model.geom_dataid[g])
                points = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
                vertices.append((points @ scratch.geom_xmat[g].reshape(3, 3).T + scratch.geom_xpos[g]
                                 - scratch.qpos[a:a+3]) @ rotation)
            top = float(np.concatenate(vertices)[:, 2].max())
            self.rim_local.append(top)
            # Measure the four inner wall faces just below the lip, independent of the authored goal box.
            distances = []
            # Some rim corners/handles protrude above the continuous wall.
            # Descend to the highest cross-section with four measured walls.
            for depth in (.005, .01, .02, .03, .04, .05, .06):
                origin = np.array([0., 0., top-depth])
                distances = []
                for direction in ([1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]):
                    direction = np.array(direction)
                    sideways = np.array([-direction[1], direction[0], 0.])
                    # Collision decomposition can leave a tiny seam on a wall's
                    # centreline. Sample adjacent rays and keep the inner bound.
                    hit = min(_ray_entry(n, d, origin + offset*sideways, direction)
                              for offset in (-.02, -.01, 0., .01, .02)
                              for n, d in self.geometry._bin_collision_planes[color])
                    distances.append(hit)
                if all(np.isfinite(hit) and hit > .01 for hit in distances):
                    break
            else:
                raise ValueError(f"Cannot measure the {color} bin opening: {distances}")
            self.opening_lo.append([-distances[1], -distances[3]])
            self.opening_hi.append([distances[0], distances[2]])
        self.rim_local = np.array(self.rim_local)
        self.opening_lo, self.opening_hi = np.array(self.opening_lo), np.array(self.opening_hi)

    def heights(self, data):
        return np.vstack((data.subtree_com[self.bodies], [data.qpos[a:a+3] for a in self.bin_adrs]))

    def opening_world(self, data, k):
        c = self.colors[k]
        return data.qpos[self.bin_adrs[c]:self.bin_adrs[c]+3] + data.xmat[self.bin_bodies[c]].reshape(3, 3) @ [0., 0., self.rim_local[c]]

    def observe(self, data, rest):
        n = len(self.names)
        rest = np.asarray(rest).reshape(-1, 3)
        pos = np.asarray(data.subtree_com[self.bodies]).reshape(n, 3)
        memberships = np.column_stack([self.geometry._inside_bin(pos[None], data.qpos[None], c)[0] for c in COLORS])
        memberships &= self.active[:, None]
        matched = memberships[np.arange(n), self.colors]
        wrong = memberships.copy(); wrong[np.arange(n), self.colors] = False
        local, extents = [], []
        for k, g in enumerate(self.block_geoms):
            c = self.colors[k]; a = self.bin_adrs[c]
            rot = data.xmat[self.bin_bodies[c]].reshape(3, 3)
            local.append((pos[k] - data.qpos[a:a+3]) @ rot)
            points = self.box_corners[k] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            extents.append(np.max(np.abs((points-pos[k]) @ rot), axis=0)[:2])
        local, extents = np.array(local), np.array(extents)
        lo = self.opening_lo[self.colors] + extents + self.cfg.clearance_m
        hi = self.opening_hi[self.colors] - extents - self.cfg.clearance_m
        gaps = np.maximum(np.maximum(lo-local[:, :2], local[:, :2]-hi), 0.)
        xy_gap = np.linalg.norm(gaps, axis=1)
        above = local[:, 2] - self.rim_local[self.colors]
        speed = np.array([np.linalg.norm(data.qvel[a:a+3]) for a in self.dof_adr])
        angular = np.array([np.linalg.norm(data.qvel[a+3:a+6]) for a in self.dof_adr])
        contacts = self.contact_obs(data)
        direct = np.zeros((n, 3), dtype=bool)
        neighbors = [set() for _ in self.names]
        resting = np.zeros(n, dtype=bool)
        touch_arm = np.zeros((n, 2), dtype=bool)
        env_touch = np.zeros(n, dtype=bool)
        for contact in data.contact[:data.ncon]:
            for obj, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                kind, i = self.owner[obj]; other_kind, j = self.owner[other]
                if kind != 0:
                    continue
                if self.arm_of[other] >= 0:
                    touch_arm[i, self.arm_of[other]] = True
                else:
                    env_touch[i] = True
                if other_kind != 1:
                    resting[i] = True
                if self.bin_owner[other] >= 0:
                    direct[i, self.bin_owner[other]] = True
                elif other_kind == 0 and i != j:
                    neighbors[i].add(j)
        stable = (self.active & ~contacts['touched'] & (speed <= self.cfg.at_rest_speed)
                  & (angular <= self.cfg.at_rest_angular_speed))
        supported = direct & stable[:, None]
        for _ in self.names:
            grown = supported.copy()
            for i, ns in enumerate(neighbors):
                if stable[i] and ns:
                    grown[i] |= np.any(supported[list(ns)], axis=0)
            if np.array_equal(grown, supported):
                break
            supported = grown
        bin_pos = np.array([data.qpos[a:a+3] for a in self.bin_adrs])
        # Handoff arrays (handoff_stage). Sides come from the episode's rest spots, so they stay fixed.
        receiver = np.where(rest[n:][self.colors, 1] > 0, 0, 1)
        giver = 1 - receiver
        cross = (rest[:n, 1] > 0) != (rest[n:][self.colors, 1] > 0)
        per_arm = contacts['hits'].reshape(n, 2, 2).sum(axis=2)
        rows = np.arange(n)
        pads = np.stack([data.xpos[l] + data.xmat[l].reshape(3, 3) @ self.pad_link6[a] for a, l in enumerate(self.link6)])
        gap_give = np.empty(n)
        for k, g in enumerate(self.block_geoms):
            local = (pads[giver[k]] - data.geom_xpos[g]) @ data.geom_xmat[g].reshape(3, 3)
            gap_give[k] = np.linalg.norm(np.maximum(np.abs(local) - self.model.geom_size[g], 0.))
        return dict(com=pos, cross=cross, giver=giver, receiver=receiver,
                    f_give=per_arm[rows, giver], f_recv=per_arm[rows, receiver],
                    touch_give=touch_arm[rows, giver], touch_recv=touch_arm[rows, receiver],
                    gap_give=gap_give, env_touch=env_touch,
                    m_gap=hs.region_gap(pos, self.cfg.handoff.region_lo, self.cfg.handoff.region_hi),
                    y_recv=pos[:, 1]*np.where(receiver == 0, 1., -1.),
                    active=self.active.copy(), xy=pos[:, :2], lift=pos[:, 2]-rest[:n, 2], speed=speed, angular_speed=angular,
                    gap=np.hypot(xy_gap, np.maximum(-above, 0.)), opening_gap=xy_gap,
                    above_rim=above, over=(xy_gap <= 1e-9) & (above >= 0), in_bin=matched,
                    memberships=memberships, wrong_bin=wrong.any(axis=1),
                    supported=supported[np.arange(n), self.colors] & matched, resting_contact=resting,
                    bin_moved=np.linalg.norm(bin_pos[:, :2]-rest[n:, :2], axis=1),
                    bin_tilt=np.array([np.arccos(np.clip(data.xmat[b].reshape(3, 3)[2, 2], -1, 1)) for b in self.bin_bodies]),
                    **contacts)


def classify(obs, prev, cfg):
    # Entering pick needs the grasp site within reach_m. A finger contact alone is not a reach: the side of a
    # closed finger touches a block 6-8 cm from the grasp site, and that is how an ungrasped push begins
    # (G5 palm/side-push hack, scripted round 1 seed 930000005). Once in pick, contact keeps the label.
    near = np.where(prev == P, (obs['dist'] <= cfg.reach_exit_m) | obs['touched'], obs['dist'] <= cfg.reach_m)
    out = np.full(prev.shape, R, dtype=np.int64)
    out[near] = P
    lifted = obs['lift'] >= cfg.lift_m
    cross = obs['cross']
    out[obs['grasped'] & lifted & ~cross] = C  # same-side blocks: unchanged
    # Cross-side blocks (amendment 3): the giver's lifted hold, or the shared "handing" test, is the handoff stage;
    # carry needs the shared handoff goal (receiver holds it alone, giver clear, touching nothing but the receiver)
    # and stays carry while the receiver holds it, giver clear, still lifted.
    h = cfg.handoff
    out[cross & (((obs['f_give'] == 2) & lifted) | hs.handing(obs, prev == H, h))] = H
    out[cross & lifted & (hs.received_in_air(obs, h) | (hs.received(obs, h) & (prev >= C) & (prev <= DONE)))] = C
    inside = obs['in_bin'] & (obs['above_rim'] <= 0)
    over = obs['over'] | inside | ((prev == D) & (obs['opening_gap'] <= cfg.opening_exit_m)
                                   & (obs['above_rim'] >= -cfg.opening_exit_m))
    # Cross-side: never drop while the giver still touches it (it is the receiver's block to drop).
    out[over & (obs['grasped'] | ((prev == D) & obs['touched'])) & ~(cross & obs['touch_give'])] = D
    at_rest = (obs['speed'] <= cfg.at_rest_speed) & (obs['angular_speed'] <= cfg.at_rest_angular_speed)
    free = ~obs['touched'] & (~at_rest | ~obs['resting_contact'])
    out[free] = prev[free]
    out[obs['in_bin'] & obs['supported'] & ~obs['touched'] & at_rest] = DONE
    out[(obs['lift'] < -cfg.fallen_m) & ~obs['in_bin']] = FALLEN
    return out


def progress(stage, obs, cfg=None):
    cfg = cfg or StageConfig()
    if stage == R: return -obs['dist']
    if stage == P: return .25*obs['fingers'] + .5*np.clip(obs['lift']/.03, 0, 1)*obs['grasped']
    if stage == H: return hs.progress(obs, cfg.handoff) - np.minimum(obs['m_gap'], cfg.region_shaping_m)
    if stage == C: return -np.minimum(obs['gap'], 1.)
    if stage == D: return -obs['opening_gap'] + .05*~obs['touched']
    raise ValueError(stage)


class LegoTracker(StageTracker):
    def __init__(self, scene, cfg=None):
        super().__init__(scene, classify, cfg or scene.cfg, start_label=R)
