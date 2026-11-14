"""Debounced per-object stage labels over an episode."""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np


class StageTracker:
    """``classify(obs, prev_labels, cfg)`` gives a stage label per object from one observation; a new label is
    committed once it has held for ``cfg.hold`` consecutive updates. ``update`` returns the objects whose committed
    label changed."""

    def __init__(self, scene, classify: Callable, cfg, *, start_label: int = 0, settle: int = 3) -> None:
        self.scene, self.classify, self.cfg = scene, classify, cfg
        self.start_label, self.settle = int(start_label), int(settle)
        self.rest_height: np.ndarray | None = None
        self.committed = self.candidate = self.count = None

    def reset(self, data: mujoco.MjData, rest_height: np.ndarray | None = None,
              labels: np.ndarray | None = None) -> np.ndarray:
        """Start an episode. A restored mid-episode start passes the original rest heights (and labels)."""
        self.rest_height = self.scene.heights(data) if rest_height is None else np.asarray(rest_height, np.float64)
        m = len(self.scene.names)
        self.obs = self.scene.observe(data, self.rest_height)
        if labels is None:
            labels = np.full(m, self.start_label, dtype=np.int64)
            for _ in range(self.settle):  # settle hysteresis from the start state
                labels = self.classify(self.obs, labels, self.cfg)
        self.committed = np.asarray(labels, dtype=np.int64).copy()
        self.candidate, self.count = self.committed.copy(), np.zeros(m, dtype=np.int64)
        return self.committed

    def update(self, data: mujoco.MjData) -> list[int]:
        self.obs = self.scene.observe(data, self.rest_height)
        raw = self.classify(self.obs, self.committed, self.cfg)
        self.count = np.where(raw == self.candidate, self.count + 1, 1)
        self.candidate = raw
        changed = np.flatnonzero((raw != self.committed) & (self.count >= self.cfg.hold))
        self.committed[changed] = raw[changed]
        return changed.tolist()
