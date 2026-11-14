"""Whiteboard wiping: a marker segment (geom ``mark_<k>``) is erased when the duster's felt pad (``duster_pad``) presses on
the board (contact with ``whiteboard_surface``, pad roughly level) and the segment's centre lies under the pad. Erased
segments are made transparent and moved 3 cm down (under the board and table), so both the MuJoCo and the Blender
renderer stop showing them (the Blender renderer syncs geom poses each frame, not colours)."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np


class WipeWhiteboardRuntime:
    def __init__(self, check_every: int = 4, margin_m: float = 0.004) -> None:
        self.check_every, self.margin = int(check_every), float(margin_m)
        self._n = 0
        self._marks = np.zeros(0, dtype=int)
        self._erased = np.zeros(0, dtype=bool)

    def bind(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._marks = np.array([g for g in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("mark_")], dtype=int)
        self._erased = np.zeros(len(self._marks), dtype=bool)
        self._pad = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "duster_pad")
        self._board = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "whiteboard_surface")
        self._n = 0

    def after_reset(self, model: mujoco.MjModel, data: mujoco.MjData, *, randomization: Any = None) -> None:
        self.bind(model, data)

    def before_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        pass

    def after_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._n += 1
        if self._n % self.check_every or self._pad < 0 or self._board < 0 or self._erased.all():
            return
        if not any((int(c.geom1), int(c.geom2)) in ((self._pad, self._board), (self._board, self._pad)) for c in data.contact[: data.ncon]):
            return
        R = data.geom_xmat[self._pad].reshape(3, 3)
        if R[2, 2] < 0.8:  # pad tilted: an edge touching, not wiping
            return
        half = model.geom_size[self._pad][:2] + self.margin
        d = data.geom_xpos[self._marks][:, :2] - data.geom_xpos[self._pad][:2]
        under = (np.abs(d @ R[:2, 0]) <= half[0]) & (np.abs(d @ R[:2, 1]) <= half[1]) & ~self._erased
        for g in self._marks[under]:
            model.geom_rgba[g, 3] = 0.0
            model.geom_pos[g, 2] -= 0.03
        self._erased |= under

    def debug_state(self) -> dict[str, Any]:
        return {"marks": int(len(self._marks)), "erased": int(self._erased.sum())}


__all__ = ["WipeWhiteboardRuntime"]
