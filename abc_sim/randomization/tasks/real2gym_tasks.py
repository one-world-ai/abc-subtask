"""Real2Gym tasks (OneWorld-AI/ABC_simulators, the user-approved releases) rebuilt on abc_sim's table scene.

Each task's objects come from its Real2Gym release scene, imported once by ``tools/real2gym_import.py`` into
``models/assets_real2gym/<task>/`` (objects.json: names, joints, start poses in the abc frame, extents;
objects.xml: assets and body subtrees with explicit inertials; assets/: meshes and textures). Real2Gym's arms stand
where abc_sim's do relative to the table, so the start layout is the release's example episode moved into the abc
frame (the midpoint of the two arm bases aligned, z from the table top).

Randomization (per reset, seeded): objects that touch at the start (an item in a bag, plates in a rack, hats on a
stand) form one cluster and move rigidly together; each cluster is shifted by up to ``xy_jitter`` and turned by up to
``yaw_jitter`` about its centre, the whole layout shifted by up to ``layout_jitter``; clusters keep a clearance and
stay on the table (a cluster the release put beyond abc_sim's shorter table is moved in first). The scene is rebuilt,
objects settle for ``settle_steps`` physics steps with the arms held, and the settled poses are the start state.

``abc_sim/real2gym_catalog.py`` (``TASKS``) holds each task's prompt, randomization and goals; ``task_eval/real2gym_tasks.py`` scores the goals.
"""

from __future__ import annotations

import os

import copy
import fnmatch
import functools
import json
from pathlib import Path as _Path
from typing import Any
import xml.etree.ElementTree as _ET

import mujoco
import numpy as np

from ..assets.paths import _MODELS_DIR
from ..core import RandomizationState
from ...real2gym_catalog import TASKS, physics, task_config
from ...real2gym_spec import resolve_spec
from .openwam_tasks import _TableSceneRandomizer, _build_table_scene

LIB_DIR = _Path(os.environ["R2G_LIB_DIR"]) if os.environ.get("R2G_LIB_DIR") else _MODELS_DIR / "assets_real2gym"  # R2G_LIB_DIR: a library copy
TABLE_X = (0.3025, 0.8975)  # abc table_plane (centre 0.6, half 0.2975) ...
TABLE_Y = (-0.65, 0.65)  # ... and the side walls
TABLE_MARGIN = 0.012
CLUSTER_CLEARANCE = 0.012
ARM_BASES_XY = ((0.252, 0.31), (0.252, -0.31))  # objects keep this far from the arm base axes
ARM_KEEPOUT_M = 0.05  # the YAM base near the table: a 6.6 cm square, corners 4.7 cm from its axis
ABC_ARM_MID = np.array([0.252, 0.0, 0.75])


# ---------------------------------------------------------------------------------------------- library
@functools.lru_cache(maxsize=None)
def load_library(task: str) -> dict[str, Any]:
    d = LIB_DIR / task
    meta = json.loads((d / "objects.json").read_text())
    root = _ET.parse(d / "objects.xml").getroot()
    assets = []
    for el in root.find("asset"):
        el = copy.deepcopy(el)
        for k, v in list(el.attrib.items()):
            if k.startswith("file") and not _Path(v).is_absolute():
                el.set(k, str((d / v).resolve()))
        assets.append("    " + _ET.tostring(el, encoding="unicode").strip())
    bodies = {b.get("name"): b for b in root.find("worldbody")}
    contact = root.find("contact")
    excludes = [dict(e.attrib) for e in contact] if contact is not None else []
    shift, dropped = _fit_table(task, meta["objects"])
    objs = []
    for o in meta["objects"]:
        if o["name"] in dropped:
            continue
        o = dict(o, home_pos=[o["home_pos"][0] + shift[0], o["home_pos"][1] + shift[1], o["home_pos"][2]])
        objs.append(o)
    names = {b.get("name") for o in objs for b in bodies[o["name"]].iter("body")}
    excludes = [e for e in excludes if e["body1"] in names and e["body2"] in names]
    quads = [_corners(o["home_pos"][:2], o["home_yaw"], o["footprint"]) for o in objs]
    close = {}  # pairs closer than the clearance at the start: keep at least half their start gap (>= 2 mm)
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            if _overlap(quads[i], quads[j], gap=CLUSTER_CLEARANCE):
                lo, hi = 0.0, CLUSTER_CLEARANCE
                for _ in range(12):  # start gap by bisection
                    mid = (lo + hi) / 2
                    lo, hi = (lo, mid) if _overlap(quads[i], quads[j], gap=mid) else (mid, hi)
                close[(i, j)] = max(0.002, lo / 2)
    clusters = _clusters(objs)
    for group in task_config(task).get("groups", []):  # catalog: objects moved as one (a tightly packed grid)
        idx = {i for i, o in enumerate(objs) if o["name"] in group}
        merged = sorted(set().union(*[set(cl) for cl in clusters if idx & set(cl)]))
        clusters = [cl for cl in clusters if not (idx & set(cl))] + [merged]
    return {"meta": meta, "assets": assets, "bodies": bodies, "excludes": excludes, "objects": objs,
            "clusters": sorted(clusters), "close_pairs": close, "shift": shift, "dropped": sorted(dropped)}


def _table_x(task: str) -> tuple[float, float]:
    """Usable table depth: the play table (x 0.3025 .. 0.8975); a task may start objects nearer the robot
    (``table_x_min``: the station base between the arms, level with the table; the collision plane is infinite)."""
    return (float(task_config(task).get("table_x_min", TABLE_X[0])), TABLE_X[1])


def _task_objects(task: str) -> set[str]:
    """Objects the task's goals name (goal objects, containers, welded bodies): never dropped."""
    cfg = task_config(task)
    out = set(cfg["fixed"])
    for g in cfg["goals"]:
        out |= set(g.get("objects", [])) | set(g.get("containers", []))
    return out


def _fit_table(task: str, objs) -> tuple[np.ndarray, set[str]]:
    """Real2Gym tables are deeper than abc_sim's (back wall x 0.9): shift the whole start layout toward the robot so
    its far edge clears the table edge (relative placement kept); if its near edge would then leave the table, drop
    the distractor clusters (no goal object or container in them) farthest from the robot until it fits."""
    keep = _task_objects(task)
    dropped: set[str] = set(task_config(task).get("drop", []))  # distractors the catalog removes outright
    objs = [o for o in objs if o["name"] not in dropped]
    clusters = _clusters(objs)
    lo_x, hi_x = _table_x(task)[0] + TABLE_MARGIN, TABLE_X[1] - TABLE_MARGIN
    lo_y, hi_y = TABLE_Y[0] + TABLE_MARGIN, TABLE_Y[1] - TABLE_MARGIN

    def bounds(skip):
        pts = np.concatenate([_corners(o["home_pos"][:2], o["home_yaw"], o["footprint"]) for o in objs if o["name"] not in skip])
        return pts.min(0), pts.max(0)
    while True:
        lo, hi = bounds(dropped)
        if (hi[0] - lo[0]) <= hi_x - lo_x and (hi[1] - lo[1]) <= hi_y - lo_y:
            break
        far = [cl for cl in clusters if not any(objs[i]["name"] in keep for i in cl)
               and not all(objs[i]["name"] in dropped for i in cl)]
        if not far:
            break  # the task objects alone do not fit: leave it to the per-cluster moves
        cl = max(far, key=lambda c: max(_corners(objs[i]["home_pos"][:2], objs[i]["home_yaw"], objs[i]["footprint"])[:, 0].max()
                                        for i in c))
        dropped |= {objs[i]["name"] for i in cl}
    shift = np.zeros(2)
    for k, (a, b) in enumerate(((lo_x, hi_x), (lo_y, hi_y))):
        if hi[k] > b:
            shift[k] = b - hi[k]
        elif lo[k] < a:
            shift[k] = a - lo[k]
    return shift, dropped


def _corners(xy, yaw, fp):
    """Footprint rectangle [xmin, xmax, ymin, ymax] (heading frame) at xy, yaw -> 4 x 2 world corners."""
    c, s = np.cos(yaw), np.sin(yaw)
    loc = np.array([[fp[0], fp[2]], [fp[1], fp[2]], [fp[1], fp[3]], [fp[0], fp[3]]])
    return loc @ np.array([[c, s], [-s, c]]) + np.asarray(xy)


def _overlap(a, b, gap=0.0):
    """Separating-axis test of two convex quads (corners in order); True if closer than gap."""
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]]) / max(np.linalg.norm(e), 1e-12)
            pa, pb = a @ n, b @ n
            if pa.max() + gap <= pb.min() or pb.max() + gap <= pa.min():
                return False
    return True


def _quad_disc(q, c, r) -> bool:
    """Does the convex quad (4 x 2 corners in order) come within r of point c?"""
    c = np.asarray(c, float)
    inside = all(np.cross(q[(i + 1) % 4] - q[i], c - q[i]) >= 0 for i in range(4)) or \
        all(np.cross(q[(i + 1) % 4] - q[i], c - q[i]) <= 0 for i in range(4))
    if inside:
        return True
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        t = np.clip(np.dot(c - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0.0, 1.0)
        if np.linalg.norm(a + t * (b - a) - c) < r:
            return True
    return False


def _clusters(objs) -> list[list[int]]:
    """Objects whose start footprints touch (within 3 mm): one cluster, moved rigidly."""
    n = len(objs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    quads = [_corners(o["home_pos"][:2], o["home_yaw"], o["footprint"]) for o in objs]
    for i in range(n):
        for j in range(i + 1, n):
            if _overlap(quads[i], quads[j], gap=0.003):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values())


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_mul(a, b) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def r2g_to_abc(meta: dict[str, Any], pos, quat) -> tuple[np.ndarray, np.ndarray]:
    """A pose in the Real2Gym release scene's world -> abc_sim's world (the importer's transform)."""
    f = meta["frame"]
    yaw0 = float(f["yaw0"])
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    d = R @ (np.asarray(pos, float) - np.array([f["mid"][0], f["mid"][1], f["table_top"]]))
    return d + ABC_ARM_MID, quat_mul(yaw_quat(-yaw0), quat)


# ---------------------------------------------------------------------------------------------- layout
def sample_layout(task: str, rng: np.random.Generator | None, scale: float = 1.0) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Per object (pos, quat) in the abc frame; rng None = the start layout. Each cluster is turned and shifted
    (rng), then pushed back onto the table if it left it; None if two clusters come closer than the clearance
    (CLUSTER_CLEARANCE, or what the pair had at the start if less)."""
    lib, cfg = load_library(task), task_config(task)
    objs = lib["objects"]
    pos = [np.asarray(o["home_pos"], float) for o in objs]
    quat = [np.asarray(o["home_quat"], float) for o in objs]
    yaw = [float(o["home_yaw"]) for o in objs]
    lay = np.zeros(2) if rng is None else scale * rng.uniform(-cfg["layout_jitter"], cfg["layout_jitter"], 2)
    lo_t = np.array([_table_x(task)[0], TABLE_Y[0]]) + TABLE_MARGIN
    hi_t = np.array([TABLE_X[1], TABLE_Y[1]]) - TABLE_MARGIN
    for cl in lib["clusters"]:
        pts = np.concatenate([_corners(pos[i][:2], yaw[i], objs[i]["footprint"]) for i in cl])
        centre = (pts.min(0) + pts.max(0)) / 2
        dyaw = 0.0 if rng is None else scale * float(rng.uniform(-cfg["yaw_jitter"], cfg["yaw_jitter"]))
        dxy = lay + (np.zeros(2) if rng is None else scale * rng.uniform(-cfg["xy_jitter"], cfg["xy_jitter"], 2))
        c, s_ = np.cos(dyaw), np.sin(dyaw)
        R = np.array([[c, -s_], [s_, c]])
        for i in cl:
            pos[i] = np.array([*(centre + R @ (pos[i][:2] - centre) + dxy), pos[i][2]])
            quat[i] = quat_mul(yaw_quat(dyaw), quat[i])
            yaw[i] += dyaw
        # back onto the table
        pts = np.concatenate([_corners(pos[i][:2], yaw[i], objs[i]["footprint"]) for i in cl])
        lo, hi = pts.min(0), pts.max(0)
        back = np.where(hi > hi_t, hi_t - hi, 0.0) + np.where(lo < lo_t, lo_t - lo, 0.0)
        if np.any((hi - lo) > (hi_t - lo_t) + 1e-9):
            return None
        for i in cl:
            pos[i][:2] += back
    quads = [_corners(pos[i][:2], yaw[i], objs[i]["footprint"]) for i in range(len(objs))]
    if any(_quad_disc(q, c, ARM_KEEPOUT_M) for q in quads for c in ARM_BASES_XY):
        return None
    cl_of = {i: k for k, cl in enumerate(lib["clusters"]) for i in cl}
    close = lib["close_pairs"]
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            if cl_of[i] == cl_of[j]:
                continue
            gap = 0.0 if rng is None else close.get((i, j), CLUSTER_CLEARANCE)
            if _overlap(quads[i], quads[j], gap=gap):
                return None
    return list(zip(pos, quat))


def build_scene_xml(task: str, layout: list[tuple[np.ndarray, np.ndarray]]) -> tuple[str, list[dict[str, Any]]]:
    """The abc table scene with the task's objects at ``layout``. Returns (xml, per-object start qpos blocks)."""
    lib = load_library(task)
    fixed = set(task_config(task)["fixed"])
    ph = physics(task)
    cfg = task_config(task)
    bodies, key_blocks, blocks, slabs = [], [], [], []
    for o, (p, q) in zip(lib["objects"], layout):
        el = copy.deepcopy(lib["bodies"][o["name"]])
        el.set("pos", " ".join(f"{v:.6f}" for v in p))
        el.set("quat", " ".join(f"{v:.8f}" for v in q))
        for pat, scale in task_config(task).get("geom_scale", {}).items():  # per-task collision fixes
            for g in el.iter("geom"):
                if fnmatch.fnmatch(g.get("name", ""), pat) and g.get("size"):
                    vals = [float(x) for x in g.get("size").split()]
                    g.set("size", " ".join(f"{v * k:.6g}" for v, k in zip(vals, list(scale) + [1.0] * 3)))
        _object_solref_floor(el)
        _cylinder_tilt(el)  # before the floor slab: _floor_slab accepts geoms turned < 1 mrad
        _object_solref_cap(el, ph["solref_max"])
        pat = cfg.get("floor_slab", {}).get(o["name"])
        if pat and ph["floor_slab"] > 0 and _floor_slab(el, pat, ph["floor_slab"]):
            slabs.append(o["name"] + SLAB_SUFFIX)
        free = o["free_joint"]
        if o["name"] in fixed:  # welded where it starts (e.g. the whiteboard Real2Gym leans on its robot mount)
            for j in el.findall("joint"):
                if j.get("type") == "free":
                    el.remove(j)
            free = None
        bodies.append("    " + _ET.tostring(el, encoding="unicode").strip())
        vals = (list(p) + list(q)) if free else []
        for j in o["internal_joints"]:
            vals += list(j["qpos"])
        key_blocks.append(tuple(vals))
        blocks.append({"name": o["name"], "free_joint": free, "internal": o["internal_joints"]})
    if "r2g_decor" in lib["bodies"]:
        el = copy.deepcopy(lib["bodies"]["r2g_decor"])
        el.set("pos", f"{lib['shift'][0]:.6f} {lib['shift'][1]:.6f} 0")
        bodies.append("    " + _ET.tostring(el, encoding="unicode").strip())
    xml = _build_table_scene(lib["assets"], bodies, [b for b in key_blocks if b])
    xml = _apply_contact_bits(xml, lib["meta"].get("contact_bits"), ph, thin_shell=bool(cfg.get("thin_shell")))
    excludes = list(lib["excludes"]) + [{"body1": "world", "body2": b} for b in slabs]  # slabs do not touch the table
    if excludes:
        ex = "\n".join(f'    <exclude body1="{e["body1"]}" body2="{e["body2"]}"/>' for e in excludes)
        xml = xml.replace("</mujoco>", f"  <contact>\n{ex}\n  </contact>\n</mujoco>")
    return xml, blocks


FINGER_PRIORITY = 5
# Experiment knob (off): contact dimension of the finger geoms (abc: 6, sliding + torsional + rolling friction). With
# the elliptic cone the three share one cone and rolling torques use up the sliding friction; R2G_FINGER_CONDIM=4 / 3.
FINGER_CONDIM = os.environ.get("R2G_FINGER_CONDIM", "")


# Finger contact and gripper settings of the Real2Gym scenes (read once from the environment).
#
# With priority 5 (below) the fingers' own contact softness applies to every object. The yam pad plates are soft
# (solref 0.02, solimp 0.9-0.95): replaying recorded rollouts they sat 3-9 mm inside glass walls 1.8 mm and bowl walls
# 3-4 mm thick, and the stiff 0.6 mm pad-tip spheres then gripped the wall from both sides, so the object hung on one
# finger of an open gripper. In the tasks the catalog marks thin_shell the finger geoms softer than PAD_SOLREF get
# PAD_SOLREF / PAD_SOLIMP (the pad-tip spheres keep their 0.004). Not in the other tasks: following the releases'
# demos, our longer fingertips dip 3-9 mm into the table and bin walls, which only soft pads let through, and stiff
# pads there made demos fail (condiments 5 -> 3 of 5, screwdriver 1 -> 0).
#   R2G_PAD_SOLREF   time constant (s); set explicitly it applies to every task; "0" = off
#   R2G_PAD_SOLIMP   impedance "d0 dmax width"
#   R2G_PAD_SCOPE    "shell" (thin_shell tasks), "all" or "off"
#   R2G_PAD_TIPS     "0": drop the pad-tip spheres (tests only)
#
# Gripper: abc's position actuator (kp 400 N/m on one finger joint, the other coupled: 200 N/m per pad) presses a
# 1.8 mm glass wall with 0.8 N per pad; the releases' own gripper controller has 2000 N/m and a 10 N limit per finger
# and holds the same wall with 6-8 N. GRIP_KP / GRIP_KV / GRIP_FORCE are that controller on abc's coupled actuator.
#   R2G_GRIP_KP / R2G_GRIP_KV / R2G_GRIP_FORCE   gain (N/m), damping (N s/m), force limit (N); KP "0" = abc's own
#   R2G_GRIP_SCOPE   "all", "shell" or "off"
#   R2G_OBJ_SOLREF_MIN   floor on the objects' own contact time constant (off)
def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    return float(v) if v != "" else default


def _scope(name: str, explicit: str, default: str) -> str:
    """Scope of a setting: R2G_<name>_SCOPE if set; else "all" when its value was set explicitly, else the default."""
    v = os.environ.get(f"R2G_{name}_SCOPE", "")
    if v in ("all", "shell", "off"):
        return v
    return "all" if os.environ.get(explicit, "") != "" else default


PAD_SOLREF_TC = _env_float("R2G_PAD_SOLREF", 0.006)
PAD_SOLREF = f"{PAD_SOLREF_TC:g} 1" if PAD_SOLREF_TC > 0 else None
PAD_SOLIMP = os.environ.get("R2G_PAD_SOLIMP") or "0.98 0.998 0.001"
PAD_SCOPE = _scope("PAD", "R2G_PAD_SOLREF", "shell")
PAD_TIPS = os.environ.get("R2G_PAD_TIPS", "1") != "0"
GRIP_KP = _env_float("R2G_GRIP_KP", 4000.0)
GRIP_KV = _env_float("R2G_GRIP_KV", 60.0)
GRIP_FORCE = _env_float("R2G_GRIP_FORCE", 20.0)
GRIP_SCOPE = _scope("GRIP", "R2G_GRIP_KP", "shell")
OBJ_SOLREF_MIN = _env_float("R2G_OBJ_SOLREF_MIN", 0.0)
# MuJoCo 3.8.1 returns no plane-cylinder contact while the cylinder's axis is numerically parallel to the plane normal
# and the body's quaternion norm is 2-4 ulps above 1 (it renormalizes only beyond 1e-15): an upright can, glass or
# bowl foot then falls through the table until another geom catches it. Each object cylinder is turned by this angle
# (rad) in its body, so it is never exactly parallel when the body stands upright (1 um over 5 cm; below about 1e-6
# the compiler treats the geom frame as the body frame and nothing changes). R2G_CYL_TILT=0 switches it off.
CYL_TILT = _env_float("R2G_CYL_TILT", 2e-5)


def _object_solref_floor(el) -> None:
    """Raise the contact time constant of an imported object's geoms to at least OBJ_SOLREF_MIN."""
    if OBJ_SOLREF_MIN <= 0:
        return
    for g in el.iter("geom"):
        sr = (g.get("solref") or "").split()
        tc = float(sr[0]) if sr else 0.02
        if 0 < tc < OBJ_SOLREF_MIN:
            g.set("solref", f"{OBJ_SOLREF_MIN:g} " + (sr[1] if len(sr) > 1 else "1"))


SOFT_SOLREF = 0.01  # s: contact time constants above this are MuJoCo's default (0.02), not a value the release set


def _object_solref_cap(el, cap: float) -> None:
    """Imported collision geoms left at MuJoCo's default contact softness (no solref: time constant 0.02 s) get
    ``cap`` s instead (0 = off). At 0.02 s a tool dropped from 10 cm sinks 10 mm into what it lands on: through a
    9 mm container floor (catalog: PHYSICS_DEFAULTS). Values the releases set themselves (0.0012 - 0.008 s) and
    negative (direct stiffness) solrefs are left alone."""
    if cap <= 0:
        return
    for g in el.iter("geom"):
        if g.get("contype") == "0" and g.get("conaffinity") == "0":
            continue
        sr = (g.get("solref") or "").split()
        tc = float(sr[0]) if sr else 0.02
        if tc > SOFT_SOLREF and tc > cap:
            g.set("solref", f"{cap:g} " + (sr[1] if len(sr) > 1 else "1"))


SLAB_SUFFIX = "_floor_slab"


def _floor_slab(el, pattern: str, depth: float) -> bool:
    """Move the floor geoms of a container (direct child geoms of ``el`` named like ``pattern``: boxes or cylinders
    with their thin axis along the body z) into a welded child body ``<name>_floor_slab`` and make them ``depth``
    thicker downward (the top stays where it is). The caller excludes that body from contact with the world body,
    so the slab reaches below the table top and the container stands on its walls. A thin floor lying on the table
    pushes whatever sinks past its mid-plane out through its underside, into the table; the slab only pushes up."""
    moved = []
    for g in list(el):
        if g.tag != "geom" or not fnmatch.fnmatch(g.get("name", ""), pattern):
            continue
        if g.get("contype") == "0" and g.get("conaffinity") == "0":
            continue
        if g.get("type") not in ("box", "cylinder") or g.get("euler") or g.get("axisangle") or not g.get("size"):
            continue
        if g.get("quat") and not _near_identity(g.get("quat")):
            continue
        size = [float(x) for x in g.get("size").split()]
        pos = [float(x) for x in (g.get("pos") or "0 0 0").split()]
        k = 2 if g.get("type") == "box" else 1  # half height: box z, cylinder half length
        size[k] += depth / 2
        pos[2] -= depth / 2
        g.set("size", " ".join(f"{v:.6g}" for v in size))
        g.set("pos", " ".join(f"{v:.6g}" for v in pos))
        g.set("mass", "0")
        el.remove(g)
        moved.append(g)
    if not moved:
        return False
    slab = _ET.SubElement(el, "body", {"name": el.get("name") + SLAB_SUFFIX})
    _ET.SubElement(slab, "inertial", {"pos": "0 0 0", "mass": "1e-6", "diaginertia": "1e-9 1e-9 1e-9"})
    for g in moved:
        slab.append(g)
    return True


def _near_identity(quat: str, tol: float = 1e-3) -> bool:
    """A geom quaternion within ``tol`` rad of no turn (e.g. a micro-radian tilt given to cylinders elsewhere): the slab
    still extends the geom along its body's z."""
    q = np.array([float(x) for x in quat.split()])
    n = float(np.linalg.norm(q))
    return n > 0 and 2.0 * float(np.arccos(min(1.0, abs(q[0]) / n))) < tol


def _apply_physics(root, ph: dict) -> None:
    """Solver options of the scene (catalog.physics): friction cone, impratio, noslip iterations."""
    opt = root.find("option")
    if opt is None:
        opt = _ET.SubElement(root, "option")
    opt.set("cone", ph["cone"])
    opt.set("impratio", f"{ph['impratio']:g}")
    if ph["noslip"]:
        opt.set("noslip_iterations", str(ph["noslip"]))


def _cylinder_tilt(el) -> None:
    """Turn every cylinder geom of an imported object by CYL_TILT about its x axis (see CYL_TILT)."""
    if CYL_TILT <= 0:
        return
    tilt = [np.cos(CYL_TILT / 2), np.sin(CYL_TILT / 2), 0.0, 0.0]
    for g in el.iter("geom"):
        if g.get("type") != "cylinder" or any(g.get(k) for k in ("euler", "axisangle", "xyaxes", "zaxis", "fromto")):
            continue
        q = [float(x) for x in (g.get("quat") or "1 0 0 0").split()]
        g.set("quat", " ".join(f"{v:.12g}" for v in quat_mul(q, tilt)))


def _in_scope(scope: str, thin_shell: bool) -> bool:
    return scope == "all" or (scope == "shell" and thin_shell)


def _finger_priority(wb, stiff: bool = False) -> None:
    """abc's finger collision geoms get contact priority FINGER_PRIORITY: Real2Gym objects carry priority 1-2 geoms tuned
    for Real2Gym's catcher fingers (wine bottles: friction 0.10 for sliding into the rack), which would otherwise set the
    friction of every finger contact; with this the gripper's own parameters (friction 2.0, pads 4.0, condim 6) apply,
    as in abc's own tasks."""
    def walk(body, finger):
        name = body.get("name", "")
        finger = finger or ((name.startswith("left_") or name.startswith("right_"))
                            and ("finger" in name or "_lf_" in name or "_rf_" in name))
        for ch in list(body):
            if ch.tag == "geom" and finger and ch.get("class", "") != "visual" and not (
                    ch.get("contype") == "0" and ch.get("conaffinity") == "0"):
                tip = ch.get("class", "") == "sphere_collision"  # pad-tip spheres: their own stiff contact (0.004)
                if tip and not PAD_TIPS:
                    body.remove(ch)
                    continue
                ch.set("priority", str(FINGER_PRIORITY))
                if FINGER_CONDIM:
                    ch.set("condim", FINGER_CONDIM)
                sr = (ch.get("solref") or "0.02").split()
                if stiff and PAD_SOLREF and not tip and float(sr[0]) > PAD_SOLREF_TC:
                    ch.set("solref", PAD_SOLREF)
                    ch.set("solimp", PAD_SOLIMP)
            elif ch.tag == "body":
                walk(ch, finger)
    for b in wb.findall("body"):
        walk(b, False)


def _gripper_actuator(root) -> None:
    """Gripper position actuators (default class "finger"): gain, damping and force limit GRIP_*."""
    if GRIP_KP <= 0:
        return
    for d in root.iter("default"):
        if d.get("class") == "finger":
            for pos in d.findall("position"):
                if GRIP_KP:
                    pos.set("kp", f"{GRIP_KP:g}")
                if GRIP_KV:
                    pos.set("kv", f"{GRIP_KV:g}")
                if GRIP_FORCE:
                    pos.set("forcerange", f"{-GRIP_FORCE:g} {GRIP_FORCE:g}")


def _apply_contact_bits(xml: str, bits: dict[str, list[int]] | None, ph: dict | None = None,
                        thin_shell: bool = False) -> str:
    """OR Real2Gym's table / arm / finger contact bits into the abc table plane, arm and finger geoms (objects whose
    masks were tuned to Real2Gym's workspace, e.g. cords contype 16, finger-only grasp boxes conaffinity 8); also
    sets the finger priority, the stiff pads and the release gripper (PAD_* / GRIP_*, ``thin_shell`` tasks by
    default) and the scene's solver options (``ph``: catalog.physics)."""
    root = _ET.fromstring(xml)
    wb = root.find("worldbody")
    _finger_priority(wb, stiff=_in_scope(PAD_SCOPE, thin_shell))
    if _in_scope(GRIP_SCOPE, thin_shell):
        _gripper_actuator(root)
    if ph is not None:
        _apply_physics(root, ph)
    if not bits or all(v == [1, 1] or v == [0, 0] for v in bits.values()):
        return _ET.tostring(root, encoding="unicode")

    def patch(g, ct, ca):
        if g.get("contype") == "0" and g.get("conaffinity") == "0":
            return  # visual-only
        g.set("contype", str(int(g.get("contype", "1")) | ct))
        g.set("conaffinity", str(int(g.get("conaffinity", "1")) | ca))

    for g in wb.findall("geom"):
        if g.get("name") == "table_plane":
            patch(g, *bits["table"])

    def walk(body, arm, finger):
        name = body.get("name", "")
        if body.tag == "body" and (name.startswith("left_") or name.startswith("right_")):
            arm = True
            finger = finger or "finger" in name or "_lf_" in name or "_rf_" in name
        for ch in body:
            if ch.tag == "geom" and arm:
                cls = ch.get("class", "")
                if cls in ("visual",) or (ch.get("contype") == "0" and ch.get("conaffinity") == "0"):
                    continue
                patch(ch, *(bits["finger"] if finger else bits["arm"]))
            elif ch.tag == "body":
                walk(ch, arm, finger)
    for b in wb.findall("body"):
        walk(b, False, False)
    return _ET.tostring(root, encoding="unicode")


def set_internal_joints(model, data, blocks) -> None:
    for b in blocks:
        for j in b["internal"]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j["name"])
            adr = int(model.jnt_qposadr[jid])
            data.qpos[adr: adr + len(j["qpos"])] = j["qpos"]


# ---------------------------------------------------------------------------------------------- randomizer
class Real2GymRandomizer(_TableSceneRandomizer):
    """One Real2Gym task's objects on the abc table, jittered per seed (module docstring). For a subtask env
    (variant "pick" / "obj"; ``obj``: the fixed instance of a legacy alias, else a class env with ``members``) the
    spec of the settled layout (real2gym_spec.resolve_spec: target instance, arm, targets, qualifier, position,
    instruction) goes into the metadata as ``r2g_spec`` and its instruction is the prompt. The class env's target is
    drawn with its own RNG (reset seed + class), so the layout of a seed is the same for every env of a task."""

    def __init__(self, task: str, variant: str | None = None, obj: str | None = None, members=None,
                 group: str | None = None) -> None:
        super().__init__()
        if task not in TASKS:
            raise KeyError(f"unknown Real2Gym task {task!r}")
        self.task, self.variant, self.obj = task, variant, obj
        self.members = [obj] if obj is not None else list(members or [])
        self.group = group
        self.settle_steps = int(task_config(task)["settle_steps"])

    def clone(self) -> "Real2GymRandomizer":
        return type(self)(self.task, self.variant, self.obj, self.members, self.group)

    def _goals_met(self, env) -> dict[str, bool]:
        """Per goal object of the task whether its goal already holds in the settled state."""
        from ...task_eval.real2gym_tasks import Real2GymEvaluator
        ev = Real2GymEvaluator(model=env.model, spec=None, task=self.task)
        return {r["object"]: bool(r["met"]) for r in ev.goal_detail(env.data)} if getattr(ev, "_ready", False) else {}

    def _reload_with_joints(self, xml: str, prompt: str, blocks) -> None:
        if self._scene_xml_transform_options is not None:
            from abc_sim.scene_xml import transform_scene_xml
            opts = self._scene_xml_transform_options
            if opts.clean or opts.mocap or opts.flexible_gripper:
                xml, _ = transform_scene_xml(xml, options=opts)
        env = self._env_ref
        if env is None:
            return
        arm = env._get_reset_arm_state()
        ph = physics(self.task)
        env._physics_dt, env._control_decimation = ph["dt"], ph["decimation"]  # also set by the task registry
        env.reload_from_xml(xml)
        mujoco.mj_resetData(env.model, env.data)
        env._set_qpos_from_state(arm)
        set_internal_joints(env.model, env.data, blocks)
        env.prompt = prompt
        mujoco.mj_forward(env.model, env.data)
        if self.settle_steps:
            env.data.ctrl[:] = 0.0
            for i, idx in enumerate(env._ctrl_indices):  # hold the arms while the objects settle
                env.data.ctrl[idx] = env.data.qpos[env._qpos_indices[i]] if i not in env._gripper_set else 0.0475
            for _ in range(int(round(self.settle_steps * 0.002 / ph["dt"]))):  # settle_steps: 2 ms steps
                mujoco.mj_step(env.model, env.data)
            env.data.qvel[:] = 0.0
            env.data.time = 0.0
            mujoco.mj_forward(env.model, env.data)
        if ph["sim_guard"]:
            from abc_sim.sim_guard import SimGuard
            env._sim_guard = SimGuard(env.model, env.data)

    def randomize(self, model: Any, data: Any, seed: int | None = None, request: Any | None = None) -> RandomizationState:
        cfg = task_config(self.task)
        rng = np.random.default_rng(seed)
        layout, jitter = None, 0.0
        for scale in (1.0, 0.5, 0.25):  # dense layouts: smaller moves if the full ones keep colliding
            for _ in range(150):
                layout = sample_layout(self.task, rng, scale)
                if layout is not None:
                    break
            if layout is not None:
                jitter = scale
                break
        jittered = layout is not None
        if layout is None:  # start layout (on the table) if no jittered one fits
            layout = sample_layout(self.task, None)
        if layout is None:
            raise RuntimeError(f"{self.task}: the start layout does not fit the abc table")
        xml, blocks = build_scene_xml(self.task, layout)
        prompt = cfg["prompt"]
        self._reload_with_joints(xml, prompt, blocks)
        env = self._env_ref
        spec = None
        if env is not None and self.variant in ("pick", "obj") and self.members:
            lib = load_library(self.task)["objects"]
            spec = resolve_spec(env.model, env.data, self.task, self.variant, self.obj, members=self.members,
                                group=self.group, seed=seed, names=[o["name"] for o in lib],
                                extents={o["name"]: o["extent"] for o in lib},
                                goal_met=self._goals_met(env) if self.variant == "obj" else None)
            prompt = env.prompt = spec["instruction"]
        object_states = {}
        for o, (p, q) in zip(load_library(self.task)["objects"], layout):
            if not o["free_joint"] or o["name"] in cfg["fixed"]:
                continue
            if env is not None:
                jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, o["free_joint"])
                a = int(env.model.jnt_qposadr[jid])
                v = env.data.qpos[a: a + 7]
                object_states[o["free_joint"]] = {"pos": v[:3].tolist(), "quat": v[3:].tolist()}
            else:
                object_states[o["free_joint"]] = {"pos": list(map(float, p)), "quat": list(map(float, q))}
        metadata = {"prompt": prompt, "real2gym_task": self.task, "jittered": jittered, "jitter_scale": jitter,
                    "objects": [o["name"] for o in load_library(self.task)["objects"]]}
        if spec is not None:
            metadata["r2g_spec"] = spec
        return RandomizationState(seed=seed or 0, object_states=object_states, metadata=metadata)


__all__ = ["TASKS", "Real2GymRandomizer", "build_scene_xml", "load_library", "r2g_to_abc", "sample_layout",
           "set_internal_joints", "task_config"]
