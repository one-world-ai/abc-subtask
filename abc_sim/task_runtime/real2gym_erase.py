"""Erasing for the Real2Gym whiteboard task (r2g_erase_the_whiteboard): an ink cell (geom ``ink_<k>`` on the
whiteboard) is erased when a felt geom of the eraser (``eraser_felt*``) touches the board surface (``board_surface``)
with its face roughly parallel to the board (within max_tilt_deg) and the cell lies under that felt geom (box: within
its half sizes + margin in the felt frame; cylinder: within its radius + margin). Erased cells are made transparent
and moved 3 cm behind the board surface, so the MuJoCo and Blender renderers stop drawing them (the Blender renderer
syncs geom poses, not colours). The evaluator reads the erased fraction from the cells' alpha."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


class Real2GymEraseRuntime:
    def __init__(self, check_every: int = 2, margin_m: float = 0.003, max_tilt_deg: float = 35.0) -> None:
        self.check_every, self.margin = int(check_every), float(margin_m)
        self.cos_tilt = float(np.cos(np.radians(max_tilt_deg)))
        self._n = 0
        self._ink = np.zeros(0, dtype=int)
        self._erased = np.zeros(0, dtype=bool)

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        name = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        self._ink = np.array([g for g in range(model.ngeom) if name(g).startswith("ink_")], dtype=int)
        self._erased = model.geom_rgba[self._ink, 3] < 0.5 if len(self._ink) else np.zeros(0, dtype=bool)
        self._felt = [g for g in range(model.ngeom) if name(g).startswith("eraser_felt")]
        self._board = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "board_surface")
        self._n = 0

    def after_reset(self, model: mujoco.MjModel, data: mujoco.MjData, *, randomization: Any = None) -> None:
        self.bind(model, data)

    def before_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        pass

    def after_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._n += 1
        if self._n % self.check_every or self._board < 0 or not self._felt or not len(self._ink) or self._erased.all():
            return
        pressing = set()
        for c in data.contact[: data.ncon]:
            a, b = int(c.geom1), int(c.geom2)
            if a == self._board and b in self._felt:
                pressing.add(b)
            elif b == self._board and a in self._felt:
                pressing.add(a)
        if not pressing:
            return
        normal = data.geom_xmat[self._board].reshape(3, 3)[:, 2]
        pts = data.geom_xpos[self._ink]
        hit = np.zeros(len(self._ink), dtype=bool)
        for f in pressing:
            R = data.geom_xmat[f].reshape(3, 3)
            if abs(float(R[:, 2] @ normal)) < self.cos_tilt:  # felt on its edge: not wiping
                continue
            d = (pts - data.geom_xpos[f]) @ R  # cells in the felt frame
            size = model.geom_size[f]
            if model.geom_type[f] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                hit |= np.hypot(d[:, 0], d[:, 1]) <= size[0] + self.margin
            else:
                hit |= (np.abs(d[:, 0]) <= size[0] + self.margin) & (np.abs(d[:, 1]) <= size[1] + self.margin)
        new = hit & ~self._erased
        if new.any():
            for g in self._ink[new]:
                model.geom_rgba[g, 3] = 0.0
                model.geom_pos[g, 2] -= 0.03
            self._erased |= new

    def debug_state(self) -> dict[str, Any]:
        return {"ink_cells": int(len(self._ink)), "erased": int(self._erased.sum())}


__all__ = ["Real2GymEraseRuntime"]
