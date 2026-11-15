"""Saved start states for single-stage RL: one ``<root>/<task>/<stage>.npz`` per stage.

A file holds N starts as parallel arrays (``seed``, ``target``, ``arm``, ``state``, ``rest_height``, ``labels`` plus
task extras), with per-row arrays of varying length stored as object arrays. Older files differ in small ways:
mug flip names the target ``mug`` and has no ``stage`` key; most files store ``stage`` as one scalar. ``load_starts``
hides that: every row comes out as a plain dict with the same keys, ready for ``env.reset_to_stage(row["seed"], row)``.
``save_starts`` writes the canonical layout.

    starts = load_starts("final_starts", "put_plastic_bottles_in_bin", "carry")
    row = starts.sample(rng)
    obs, info = env.reset_to_stage(row["seed"], row)
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED = ("seed", "target", "state", "rest_height", "labels")
TARGET_ALIASES = ("target", "mug")  # older files use the task's own name for the target index
ENV_ROOT = "ABC_SUBTASK_STARTS"  # default root directory when none is given


class StartSet(Sequence):
    """The starts of one stage, as a sequence of row dicts (keys: seed, stage, target, arm, state, rest_height,
    labels, and any task extras)."""

    def __init__(self, rows: list[dict[str, Any]], *, task: str = "", stage: str = "", path: Path | None = None):
        self.rows, self.task, self.stage, self.path = rows, task, stage, path

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i):  # type: ignore[override]
        return self.rows[i]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.rows)

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        """One start, uniformly at random."""
        return self.rows[int(rng.integers(len(self.rows)))]

    @property
    def seeds(self) -> np.ndarray:
        return np.array([r["seed"] for r in self.rows], dtype=np.int64)

    def __repr__(self) -> str:
        return f"StartSet({self.task}__{self.stage}, {len(self)} starts)"


def _scalar(value: np.ndarray) -> Any:
    item = value.item()
    return item.decode() if isinstance(item, bytes) else item


def read_starts(path: str | os.PathLike, *, stage: str | None = None) -> StartSet:
    """Load one start file. ``stage`` (default: the file's ``stage`` key, else its file name) fills rows without it."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        arrays = {k: z[k] for k in z.files}
    scalars = {k: _scalar(v) for k, v in arrays.items() if v.ndim == 0}
    columns = {k: v for k, v in arrays.items() if v.ndim > 0}
    if "target" not in columns:
        alias = next((a for a in TARGET_ALIASES if a in columns), None)
        if alias is None:
            raise KeyError(f"{path}: no target column (looked for {TARGET_ALIASES})")
        columns["target"] = columns[alias]
    missing = [k for k in REQUIRED if k not in columns]
    if missing:
        raise KeyError(f"{path}: missing {missing}")
    n = len(columns["seed"])
    if any(len(v) != n for v in columns.values()):
        raise ValueError(f"{path}: columns have different lengths")
    stage = stage or scalars.get("stage") or path.stem
    rows = []
    for i in range(n):
        row: dict[str, Any] = {**scalars, "stage": stage}
        for k, v in columns.items():
            x = v[i]
            row[k] = x.item() if isinstance(x, np.generic) else np.asarray(x)
        row["seed"], row["target"] = int(row["seed"]), int(row["target"])
        row["arm"] = int(row.get("arm", -1))
        for k in ("state", "rest_height"):
            row[k] = np.asarray(row[k], dtype=np.float64)
        row["labels"] = np.asarray(row["labels"], dtype=np.int64)
        rows.append(row)
    return StartSet(rows, task=path.parent.name, stage=stage, path=path)


def load_starts(root: str | os.PathLike | None, task: str, stage: str) -> StartSet:
    """The starts of ``<task>__<stage>`` from ``<root>/<task>/<stage>.npz`` (root defaults to $ABC_SUBTASK_STARTS)."""
    root = root if root is not None else os.environ.get(ENV_ROOT)
    if root is None:
        raise ValueError(f"no start root: pass one or set ${ENV_ROOT} (e.g. the downloaded final_starts/)")
    path = Path(root) / task / f"{stage}.npz"
    if not path.exists():
        raise FileNotFoundError(f"no starts for {task}__{stage} at {path}")
    return read_starts(path, stage=stage)


def save_starts(path: str | os.PathLike, rows: Sequence[dict[str, Any]], *, stage: str) -> Path:
    """Write rows (``build_start`` results or ``StartSet`` rows) in the canonical layout: ``stage`` scalar, one
    column per key, ragged arrays as object arrays."""
    if not rows:
        raise ValueError("no rows to save")
    keys = [k for k in rows[0] if k != "stage"]
    if any(set(r) - {"stage"} != set(keys) for r in rows):
        raise ValueError("rows have different keys")
    out: dict[str, Any] = {"stage": np.array(stage)}
    for k in keys:
        values = [r[k] for r in rows]
        if all(np.ndim(v) == 0 for v in values):
            out[k] = np.asarray(values)
        else:
            col = np.empty(len(values), dtype=object)
            col[:] = [np.asarray(v) for v in values]
            out[k] = col
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return path
