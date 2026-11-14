"""Randomizers for the OpenWAM-added tasks: open / close a named drawer, put a named fruit on the plate, stack and
unstack plates.

The drawer tasks use ``yam_drawer_openclose_scene.xml`` (the drawer scene's three-drawer cabinet without the markers);
one drawer is the target each world and the prompt names it by position. The fruit and plate tasks reload the
put_relative table scene with their own objects: fruit models from ``task_put_relative/objects`` and plate models from
``task_dishrack/plate`` (assembled the way the dish-rack task assembles them).
"""

from __future__ import annotations

from pathlib import Path as _Path
import re
from typing import Any
import xml.etree.ElementTree as _ET

import mujoco
import numpy as np

from ..assets.dishrack import (
    _dishrack_build_object_block,
    _dishrack_compiled_xy_half_extents,
    _dishrack_variant_dir,
    _dishrack_variant_names,
)
from ..assets.inhand import _inhand_parse_model_xml
from ..assets.paths import _MODELS_DIR
from ..assets.xml_common import _dishrack_compiled_model_bounds, _dishrack_serialize_elements
from ..core import PerturbRange, RandomizationState, SceneRandomizer, _resolve_scene_xml_paths
from .put_relative import _put_relative_get_variants, _put_relative_variant_metrics

_TABLE_Z = 0.75
_ARM_QPOS = "0 1.047 1.047 0 0 0 0 0  0 1.047 1.047 0 0 0 0 0"
_HOME_KEY_RE = re.compile(r'(<key name="home"\s+qpos=")(.*?)("\s+ctrl=)', re.DOTALL)
_TABLE_SCENE_XML = _MODELS_DIR / "yam_put_relative_scene.xml"
_ASSETS_PLACEHOLDER = "<!-- RELATIVE_PUT_TASK_ASSETS_PLACEHOLDER -->"
_OBJECTS_BEGIN = "<!-- RELATIVE_PUT_OBJECTS_BEGIN -->"
_OBJECTS_END = "<!-- RELATIVE_PUT_OBJECTS_END -->"

# ----------------------------------------------------------------------------------------------- drawers
DRAWERS: dict[str, str] = {  # prompt word -> slide joint (range 0 .. 0.2 m, + = open)
    "top": "joint_top_drawer_low",
    "middle": "joint_middle_drawer_low",
    "bottom": "joint_bottom_drawer_low",
}
DRAWER_CLOSE_START_RANGE_M = (0.14, 0.19)  # close task: how far the target drawer starts open


class DrawerOpenCloseRandomizer(SceneRandomizer):
    """Cabinet position/yaw jitter (as the drawer task), a target drawer, and its starting opening.

    mode "open": every drawer starts closed; prompt "open the <top|middle|bottom> drawer".
    mode "close": the target starts 14-19 cm open, the others closed; prompt "close the <...> drawer".
    """

    min_clearance_m = 0.04
    perturbations = [
        PerturbRange("drawer_body", delta_x=(-0.06, 0.0), delta_y=(-0.08, 0.08), delta_yaw=(-0.25, 0.25), fixed_body=True),
    ]

    def __init__(self, mode: str = "open") -> None:
        super().__init__()
        if mode not in ("open", "close"):
            raise ValueError(f"mode must be open or close, got {mode!r}")
        self.mode = mode

    def clone(self) -> "SceneRandomizer":
        return type(self)(self.mode)

    def randomize(self, model: Any, data: Any, seed: int | None = None, request: Any | None = None) -> RandomizationState:
        state = super().randomize(model, data, seed=seed, request=request)
        rng = np.random.default_rng(None if seed is None else seed + 7919)
        target = list(DRAWERS)[int(rng.integers(0, len(DRAWERS)))]
        start = {k: 0.0 for k in DRAWERS}
        if self.mode == "close":
            start[target] = float(rng.uniform(*DRAWER_CLOSE_START_RANGE_M))
        for word, joint in DRAWERS.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            data.qpos[model.jnt_qposadr[jid]] = start[word]
            data.qvel[model.jnt_dofadr[jid]] = 0.0
        mujoco.mj_forward(model, data)
        prompt = f"{self.mode} the {target} drawer"
        if self._env_ref is not None:
            self._env_ref.prompt = prompt
        metadata = dict(state.metadata or {})
        metadata.update(prompt=prompt, mode=self.mode, target_drawer=target, target_joint=DRAWERS[target],
                        drawer_start={DRAWERS[k]: v for k, v in start.items()})
        return RandomizationState(seed=state.seed, object_states=state.object_states, scale_states=state.scale_states,
                                  metadata=metadata)

    def apply(self, model: Any, data: Any, state: RandomizationState) -> None:
        super().apply(model, data, state)
        for joint, q in (state.metadata or {}).get("drawer_start", {}).items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            data.qpos[model.jnt_qposadr[jid]] = float(q)
        mujoco.mj_forward(model, data)


# ----------------------------------------------------------------------------------------------- table scenes
def _object_block(prefix: str, name: str, variant_dir: _Path, pos: tuple[float, float, float], yaw: float) -> tuple[list[str], list[str]]:
    """put_relative-style object (visual meshes + box collision proxy) as a free body named ``name``."""
    parsed = _inhand_parse_model_xml(variant_dir)
    used = {g["mesh"] for g in parsed["vis_geoms"] if g["mesh"]} | {g.get("mesh") for g in parsed["col_geoms"] if g.get("mesh")}
    assets = [f"    <!-- {name}: {variant_dir.parent.name}/{variant_dir.name} -->"]
    for mesh in parsed["meshes"]:
        if mesh["name"] in used:
            extra = "".join(f' {k}="{v}"' for k, v in mesh["extra"].items())
            assets.append(f'    <mesh file="{variant_dir / mesh["file"]}" name="{prefix}_{mesh["name"]}"{extra}/>')
    for tex in parsed["textures"]:
        assets.append(f'    <texture file="{variant_dir / tex["file"]}" name="{prefix}_{tex["name"]}" type="{tex["type"]}"/>')
    for mat in parsed["materials"]:
        attrs = f'name="{prefix}_{mat["name"]}"'
        if mat["texture"]:
            attrs += f' texture="{prefix}_{mat["texture"]}"'
        for key in ("rgba", "shininess", "specular"):
            if mat[key]:
                attrs += f' {key}="{mat[key]}"'
        assets.append(f"    <material {attrs}/>")
    qw, qz = np.cos(yaw / 2), np.sin(yaw / 2)
    body = [f'    <body name="{name}" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" quat="{qw:.6f} 0 0 {qz:.6f}">',
            f'      <joint name="{name}_joint" class="relative_object"/>']
    for i, g in enumerate(parsed["vis_geoms"]):
        mesh = f'mesh="{prefix}_{g["mesh"]}"' if g["mesh"] else ""
        mat = f' material="{prefix}_{g["material"]}"' if g["material"] else ""
        body.append(f'      <geom name="{name}_visual_{i}" type="mesh" {mesh}{mat} contype="0" conaffinity="0" group="2" density="0"/>')
    for i, g in enumerate(parsed["col_geoms"]):
        attrs = {"name": f"{name}_geom" if i == 0 else f"{name}_geom_{i}"}
        for key, value in g.items():
            if value:
                attrs[key] = f"{prefix}_{value}" if key == "mesh" else value
        attrs.setdefault("type", "mesh")
        body.append("      <geom" + "".join(f' {k}="{v}"' for k, v in attrs.items()) + "/>")
    body.append("    </body>")
    return assets, body


def _plate_block(index: int, variant: str, pos: tuple[float, float, float], yaw: float) -> tuple[str, str]:
    """A dish-rack plate as free body ``plate_<index>`` (joint ``plate_<index>_joint``), bottom at ``pos`` z."""
    assets, wrapper = _dishrack_build_object_block(kind="plate", variant_name=variant, scale_factor=1.0,
                                                   object_name=f"plate_{index}", joint_name=f"plate_{index}_joint",
                                                   instance_index=min(index, 3))
    wrapper.set("pos", f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}")
    wrapper.set("quat", f"{np.cos(yaw / 2):.6f} 0 0 {np.sin(yaw / 2):.6f}")
    return _dishrack_serialize_elements(assets), _dishrack_serialize_elements([wrapper])


def _build_table_scene(assets: list[str], bodies: list[str], free_poses: list[tuple[float, ...]]) -> str:
    """The put_relative table scene with these objects instead of its placeholder boxes."""
    text = _TABLE_SCENE_XML.read_text()
    start, end = text.find(_OBJECTS_BEGIN), text.find(_OBJECTS_END) + len(_OBJECTS_END)
    text = text[:start] + "\n".join([_OBJECTS_BEGIN, *bodies, "    " + _OBJECTS_END]) + text[end:]
    text = text.replace(_ASSETS_PLACEHOLDER, "\n".join(assets))
    qpos = "\n            ".join([" ".join(f"{v:.6f}" for v in p) for p in free_poses] + [_ARM_QPOS])
    text, n = _HOME_KEY_RE.subn(lambda m: m.group(1) + qpos + m.group(3), text, count=1)
    if n != 1:
        raise RuntimeError("table scene has no home key")
    return _resolve_scene_xml_paths(text, _TABLE_SCENE_XML.parent)


def plate_geometry(variant: str) -> dict[str, float]:
    """Radius (min xy half extent) and height of a plate model."""
    model = mujoco.MjModel.from_xml_path(str(_dishrack_variant_dir("plate", variant) / "model.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lo, hi = _dishrack_compiled_model_bounds(model, data)
    hx, hy = _dishrack_compiled_xy_half_extents("plate", variant)
    return {"radius": float(min(hx, hy)), "height": float(hi[2] - lo[2])}


class _TableSceneRandomizer(SceneRandomizer):
    """Base for tasks that rebuild the table scene each reset (like put_relative)."""

    perturbations: list = []
    settle_steps = 0  # physics steps run after the reload so stacked objects come to rest

    def clone(self) -> "SceneRandomizer":
        return type(self)()

    def prepare_env(self) -> None:
        if self._env_ref is not None:
            self.randomize(self._env_ref.model, self._env_ref.data)

    def _reload(self, xml: str, prompt: str) -> None:
        if self._scene_xml_transform_options is not None:
            from abc_sim.scene_xml import transform_scene_xml
            opts = self._scene_xml_transform_options
            if opts.clean or opts.mocap or opts.flexible_gripper:
                xml, _ = transform_scene_xml(xml, options=opts)
        env = self._env_ref
        if env is None:
            return
        arm = env._get_reset_arm_state()
        env.reload_from_xml(xml)
        mujoco.mj_resetData(env.model, env.data)
        env._set_qpos_from_state(arm)
        env.prompt = prompt
        mujoco.mj_forward(env.model, env.data)
        if self.settle_steps:
            env.data.ctrl[:] = 0.0
            for i, idx in enumerate(env._ctrl_indices):  # hold the arms where they are while objects settle
                env.data.ctrl[idx] = env.data.qpos[env._qpos_indices[i]] if i not in env._gripper_set else 1.0 * 0.0475
            for _ in range(self.settle_steps):
                mujoco.mj_step(env.model, env.data)
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)


# ----------------------------------------------------------------------------------------------- fruit on plate
FRUITS: tuple[str, ...] = ("apple", "lemon", "orange", "pear", "tomato")
_FRUIT_COUNT_RANGE = (3, 4)
_FRUIT_REGION = (0.40, 0.70, -0.40, 0.40)  # x_min, x_max, y_min, y_max
_FRUIT_CLEARANCE_M = 0.03


class FruitOnPlateRandomizer(_TableSceneRandomizer):
    """One plate and 3-4 different fruits; the prompt names one: "put the <fruit> on the plate"."""

    def __init__(self) -> None:
        super().__init__()
        self._variants: dict[str, list[_Path]] = {}

    def _fruit_variants(self, category: str) -> list[_Path]:
        if category not in self._variants:
            self._variants[category] = _put_relative_get_variants(category)
        return self._variants[category]

    def randomize(self, model: Any, data: Any, seed: int | None = None, request: Any | None = None) -> RandomizationState:
        rng = np.random.default_rng(seed)
        plate_variant = _dishrack_variant_names("plate")[int(rng.integers(0, len(_dishrack_variant_names("plate"))))]
        geo = plate_geometry(plate_variant)
        for _ in range(200):
            n = int(rng.integers(_FRUIT_COUNT_RANGE[0], _FRUIT_COUNT_RANGE[1] + 1))
            cats = rng.choice(FRUITS, size=n, replace=False).tolist()
            plate_xy = np.array([rng.uniform(0.47, 0.62), rng.uniform(-0.25, 0.25)])
            placed = [("plate", plate_xy, geo["radius"])]
            fruits = []
            for cat in cats:
                vdir = self._fruit_variants(cat)[int(rng.integers(0, len(self._fruit_variants(cat))))]
                met = _put_relative_variant_metrics(vdir)
                for _ in range(100):
                    xy = np.array([rng.uniform(_FRUIT_REGION[0], _FRUIT_REGION[1]), rng.uniform(_FRUIT_REGION[2], _FRUIT_REGION[3])])
                    if all(np.linalg.norm(xy - p) > r + met["footprint_radius"] + _FRUIT_CLEARANCE_M for _, p, r in placed):
                        break
                else:
                    break
                placed.append((cat, xy, met["footprint_radius"]))
                fruits.append({"category": cat, "variant_dir": vdir, "xy": xy, "z": met["spawn_z"], "yaw": float(rng.uniform(-np.pi, np.pi)),
                               "footprint_radius": met["footprint_radius"]})
            if len(fruits) == n:
                break
        else:
            raise RuntimeError("fruit_on_plate could not place the objects")
        target = int(rng.integers(0, len(fruits)))
        plate_yaw = float(rng.uniform(-np.pi, np.pi))
        assets, bodies, poses = [], [], []
        pa, pb = _plate_block(0, plate_variant, (plate_xy[0], plate_xy[1], _TABLE_Z + 0.001), plate_yaw)
        assets.append(pa); bodies.append(pb)
        poses.append((plate_xy[0], plate_xy[1], _TABLE_Z + 0.001, np.cos(plate_yaw / 2), 0, 0, np.sin(plate_yaw / 2)))
        objects = []
        for i, f in enumerate(fruits, start=1):
            name = f"fruit_{i:02d}"
            a, b = _object_block(f"fruit_obj_{i:02d}", name, f["variant_dir"], (f["xy"][0], f["xy"][1], f["z"]), f["yaw"])
            assets += a; bodies += b
            poses.append((f["xy"][0], f["xy"][1], f["z"], np.cos(f["yaw"] / 2), 0, 0, np.sin(f["yaw"] / 2)))
            objects.append({"name": name, "joint": f"{name}_joint", "category": f["category"], "variant": f["variant_dir"].name})
        label = fruits[target]["category"]
        prompt = f"put the {label} on the plate"
        self._reload(_build_table_scene(assets, bodies, poses), prompt)
        object_states = {"plate_0_joint": {"pos": list(poses[0][:3]), "quat": list(poses[0][3:])}}
        for o, p in zip(objects, poses[1:]):
            object_states[o["joint"]] = {"pos": list(p[:3]), "quat": list(p[3:])}
        return RandomizationState(seed=seed or 0, object_states=object_states, metadata={
            "prompt": prompt, "target_object": objects[target]["name"], "target_category": label, "objects": objects,
            "plate": "plate_0", "plate_variant": plate_variant, "plate_radius": geo["radius"], "plate_height": geo["height"]})


# ----------------------------------------------------------------------------------------------- stack / unstack plates
_PLATE_COUNT_RANGE = (2, 3)


class PlateStackRandomizer(_TableSceneRandomizer):
    """2-3 identical plates. mode "stack": spread on the table, prompt "stack the plates".
    mode "unstack": one stack, prompt "unstack the plates" (lay each plate on the table on its own)."""

    def __init__(self, mode: str = "stack") -> None:
        super().__init__()
        if mode not in ("stack", "unstack"):
            raise ValueError(f"mode must be stack or unstack, got {mode!r}")
        self.mode = mode
        self.settle_steps = 1500 if mode == "unstack" else 0

    def clone(self) -> "SceneRandomizer":
        return type(self)(self.mode)

    def randomize(self, model: Any, data: Any, seed: int | None = None, request: Any | None = None) -> RandomizationState:
        rng = np.random.default_rng(seed)
        variants = _dishrack_variant_names("plate")
        variant = variants[int(rng.integers(0, len(variants)))]
        geo = plate_geometry(variant)
        r = geo["radius"]
        n = int(rng.integers(_PLATE_COUNT_RANGE[0], _PLATE_COUNT_RANGE[1] + 1))
        poses = []
        if self.mode == "stack":
            for _ in range(500):
                xy = [np.array([rng.uniform(0.45 + 0.0, 0.66), rng.uniform(-0.38 + r * 0.5, 0.38 - r * 0.5)]) for _ in range(n)]
                if all(np.linalg.norm(a - b) > 2 * r + 0.03 for i, a in enumerate(xy) for b in xy[i + 1:]):
                    break
            else:
                raise RuntimeError("plates_stack could not place the plates apart")
            poses = [(p[0], p[1], _TABLE_Z + 0.001, float(rng.uniform(-np.pi, np.pi))) for p in xy]
        else:
            base = np.array([rng.uniform(0.50, 0.60), rng.uniform(-0.15, 0.15)])
            step = min(geo["height"], 0.03) * 0.6 + 0.006  # nested plates sit a fraction of a plate height apart
            poses = [(base[0], base[1], _TABLE_Z + 0.001 + k * step, float(rng.uniform(-np.pi, np.pi))) for k in range(n)]
        assets, bodies, free = [], [], []
        for k, (x, y, z, yaw) in enumerate(poses):
            a, b = _plate_block(k, variant, (x, y, z), yaw)
            assets.append(a); bodies.append(b)
            free.append((x, y, z, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))
        prompt = "stack the plates" if self.mode == "stack" else "unstack the plates"
        self._reload(_build_table_scene(assets, bodies, free), prompt)
        object_states = {}
        env = self._env_ref
        for k in range(n):
            if env is not None:  # after settling (unstack), record where the plates came to rest
                jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"plate_{k}_joint")
                q = env.data.qpos[env.model.jnt_qposadr[jid]: env.model.jnt_qposadr[jid] + 7]
                object_states[f"plate_{k}_joint"] = {"pos": q[:3].tolist(), "quat": q[3:].tolist()}
            else:
                object_states[f"plate_{k}_joint"] = {"pos": list(free[k][:3]), "quat": list(free[k][3:])}
        return RandomizationState(seed=seed or 0, object_states=object_states, metadata={
            "prompt": prompt, "mode": self.mode, "plates": [f"plate_{k}" for k in range(n)], "plate_variant": variant,
            "plate_radius": r, "plate_height": geo["height"]})


__all__ = ["DRAWERS", "DrawerOpenCloseRandomizer", "FRUITS", "FruitOnPlateRandomizer", "PlateStackRandomizer", "plate_geometry"]


# ----------------------------------------------------------------------------------------------- wipe the whiteboard
BOARD_HALF = (0.20, 0.15, 0.006)  # 40 x 30 cm, 1.2 cm thick, lying flat on the table
DUSTER_PAD_HALF = (0.06, 0.025, 0.006)
DUSTER_HANDLE_HALF = (0.045, 0.014, 0.012)
_MARK_COLORS = ("0.05 0.08 0.35 1", "0.05 0.05 0.05 1", "0.55 0.05 0.05 1", "0.05 0.35 0.10 1")


class WipeWhiteboardRandomizer(_TableSceneRandomizer):
    """A whiteboard flat on the table with 3-6 marker strokes (short visual-only segments, ``mark_<k>``) and a duster
    (felt pad ``duster_pad`` + handle) beside it. Prompt "wipe the whiteboard clean". The task runtime erases the
    segments the pad wipes over (task_runtime/wipe_whiteboard.py)."""

    def randomize(self, model: Any, data: Any, seed: int | None = None, request: Any | None = None) -> RandomizationState:
        rng = np.random.default_rng(seed)
        bx, by, bz = BOARD_HALF
        cx, cy, yaw = rng.uniform(0.50, 0.58), rng.uniform(-0.10, 0.10), rng.uniform(-0.25, 0.25)
        top = 2 * bz
        # strokes: random-walk polylines inside the board, 3 cm from the frame, made of 1.6 cm segments
        segs = []
        for s in range(int(rng.integers(3, 7))):
            p = np.array([rng.uniform(-bx + 0.05, bx - 0.05), rng.uniform(-by + 0.05, by - 0.05)])
            ang = rng.uniform(-np.pi, np.pi)
            color = _MARK_COLORS[int(rng.integers(0, len(_MARK_COLORS)))]
            for _ in range(int(rng.integers(5, 13))):
                ang += rng.uniform(-0.6, 0.6)
                q = p + 0.016 * np.array([np.cos(ang), np.sin(ang)])
                if not (abs(q[0]) < bx - 0.03 and abs(q[1]) < by - 0.03):
                    ang += np.pi / 2
                    continue
                segs.append(((p + q) / 2, ang, color))
                p = q
        frame = '      <geom name="whiteboard_frame_{k}" type="box" size="{sx:.4f} {sy:.4f} 0.009" pos="{px:.4f} {py:.4f} 0.009" rgba="0.62 0.64 0.66 1" friction="1 0.02 0.002"/>'
        fr = [frame.format(k=0, sx=bx, sy=0.006, px=0, py=by), frame.format(k=1, sx=bx, sy=0.006, px=0, py=-by),
              frame.format(k=2, sx=0.006, sy=by, px=bx, py=0), frame.format(k=3, sx=0.006, sy=by, px=-bx, py=0)]
        marks = [f'      <geom name="mark_{i}" type="box" size="0.0085 0.0022 0.0004" pos="{c[0]:.4f} {c[1]:.4f} {top + 0.0005:.4f}" '
                 f'euler="0 0 {a:.4f}" rgba="{col}" contype="0" conaffinity="0" group="2"/>' for i, (c, a, col) in enumerate(segs)]
        board = [f'    <body name="whiteboard" pos="{cx:.4f} {cy:.4f} {_TABLE_Z:.4f}" quat="{np.cos(yaw / 2):.6f} 0 0 {np.sin(yaw / 2):.6f}">',
                 f'      <geom name="whiteboard_surface" type="box" size="{bx} {by} {bz}" pos="0 0 {bz}" rgba="0.97 0.97 0.97 1" friction="0.5 0.02 0.002" condim="3"/>',
                 *fr, *marks, "    </body>"]
        # duster on the table, off the board, on the side of the arm that is closer
        side = 1.0 if rng.random() < 0.5 else -1.0
        dx, dy, dyaw = cx + rng.uniform(-0.08, 0.05), cy + side * (by + 0.10 + rng.uniform(0, 0.04)), rng.uniform(-0.6, 0.6)
        px, py, pz = DUSTER_PAD_HALF
        hx, hy, hz = DUSTER_HANDLE_HALF
        dz = _TABLE_Z + pz + 0.001
        duster = [f'    <body name="duster" pos="{dx:.4f} {dy:.4f} {dz:.4f}" quat="{np.cos(dyaw / 2):.6f} 0 0 {np.sin(dyaw / 2):.6f}">',
                  '      <joint name="duster_joint" class="relative_object"/>',
                  f'      <geom name="duster_pad" type="box" size="{px} {py} {pz}" rgba="0.22 0.25 0.32 1" mass="0.03" friction="0.45 0.02 0.002" condim="3"/>',
                  f'      <geom name="duster_handle" type="box" size="{hx} {hy} {hz}" pos="0 0 {pz + hz}" rgba="0.55 0.36 0.20 1" mass="0.05" friction="1.5 0.05 0.005" condim="4"/>',
                  "    </body>"]
        poses = [(dx, dy, dz, np.cos(dyaw / 2), 0, 0, np.sin(dyaw / 2))]
        prompt = "wipe the whiteboard clean"
        self._reload(_build_table_scene([], board + duster, poses), prompt)
        return RandomizationState(seed=seed or 0, object_states={"duster_joint": {"pos": list(poses[0][:3]), "quat": list(poses[0][3:])}},
                                  metadata={"prompt": prompt, "marks": len(segs), "board_center": [cx, cy], "board_yaw": yaw})


__all__ += ["WipeWhiteboardRandomizer"]


# ----------------------------------------------------------------------------------------------- hang the headphones
_HEADPHONE_ROOT = _MODELS_DIR / "assets" / "task_hang_headphones"
# headset centre when hanging, in the stand frame (band over the top rest, cups on either side of the post)
STAND_HANG_POINT = (0.0, 0.013, 0.206)


def _fixed_block(prefix: str, name: str, variant_dir: _Path, pos, yaw: float) -> tuple[list[str], list[str]]:
    """Like _object_block but a fixed body (no joint)."""
    assets, body = _object_block(prefix, name, variant_dir, pos, yaw)
    return assets, [line for line in body if "<joint " not in line]


class HangHeadphonesRandomizer(_TableSceneRandomizer):
    """A headset stand (fixed, its rest facing the robot) and one of the RoboDojo headsets lying flat on the table.
    Prompt "hang the headphones on the stand"."""

    settle_steps = 1500

    def randomize(self, model: Any, data: Any, seed: int | None = None, request: Any | None = None) -> RandomizationState:
        rng = np.random.default_rng(seed)
        stands = sorted(p for p in (_HEADPHONE_ROOT / "stand").iterdir() if (p / "model.xml").exists())
        heads = sorted(p for p in (_HEADPHONE_ROOT / "headset").iterdir() if (p / "model.xml").exists())
        stand, head = stands[int(rng.integers(0, len(stands)))], heads[int(rng.integers(0, len(heads)))]
        sx, sy = rng.uniform(0.56, 0.64), rng.uniform(-0.08, 0.08)
        syaw = -np.pi / 2 + rng.uniform(-0.3, 0.3)  # stand +y (post side) away from the robot: the rest overhangs toward it
        side = 1.0 if rng.random() < 0.5 else -1.0
        hx, hy = rng.uniform(0.42, 0.52), sy + side * rng.uniform(0.20, 0.26)
        hyaw = rng.uniform(-np.pi, np.pi)
        # lying flat: the headset's plane (x-z) horizontal = 90 deg about its x axis, then yaw
        qx = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
        qz = np.array([np.cos(hyaw / 2), 0, 0, np.sin(hyaw / 2)])
        q = np.zeros(4); mujoco.mju_mulQuat(q, qz, qx)
        hz = _TABLE_Z + 0.05
        a1, b1 = _fixed_block("stand_0", "headset_stand", stand, (sx, sy, _TABLE_Z), syaw)
        a2, b2 = _object_block("headset_0", "headset", head, (hx, hy, hz), 0.0)
        b2[0] = f'    <body name="headset" pos="{hx:.4f} {hy:.4f} {hz:.4f}" quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}">'
        prompt = "hang the headphones on the stand"
        self._reload(_build_table_scene(a1 + a2, b1 + b2, [(hx, hy, hz, *q)]), prompt)
        env = self._env_ref
        if env is not None:
            jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "headset_joint")
            st = env.data.qpos[env.model.jnt_qposadr[jid]: env.model.jnt_qposadr[jid] + 7].copy()
        else:
            st = np.array([hx, hy, hz, *q])
        c, s_ = np.cos(syaw), np.sin(syaw)
        hp = np.array([sx + c * STAND_HANG_POINT[0] - s_ * STAND_HANG_POINT[1], sy + s_ * STAND_HANG_POINT[0] + c * STAND_HANG_POINT[1], _TABLE_Z + STAND_HANG_POINT[2]])
        return RandomizationState(seed=seed or 0, object_states={"headset_joint": {"pos": st[:3].tolist(), "quat": st[3:].tolist()}},
                                  metadata={"prompt": prompt, "headset_variant": head.name, "stand_variant": stand.name,
                                            "stand_pose": [sx, sy, _TABLE_Z, syaw], "hang_point": hp.tolist()})


__all__ += ["HangHeadphonesRandomizer", "STAND_HANG_POINT"]
