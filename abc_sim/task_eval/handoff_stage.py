"""A task-independent handoff stage: one arm (the giver) passes a held object to the other arm (the receiver) in mid-air.

Used by ``inhand_stages`` (inhand_transfer_item_to_other_side); written so any task whose single arm cannot reach
the goal (put_relative cross-table carries, clutter/lego far bins) can reuse it. Nothing here knows about the task's
goal: it needs per-object arrays describing the giver and the receiver, the object's lift above its table rest
height, and a box-shaped handoff region.

Observation keys it reads (arrays over objects; the task scene produces them, see ``InhandScene.observe``):
    com          (m, 3) centre of mass
    lift         CoM height above its table rest height (m)
    f_give       giver fingers on the object (0-2)        f_recv      receiver fingers on it (0-2)
    touch_give   any giver-arm geom touches it             touch_recv  any receiver-arm geom touches it
    gap_give     giver pinch centre (between the finger pads) -> nearest surface point of the object (m)
    env_touch    it touches something that is not a robot (table, fixtures, other objects)

Stage tests (``HandoffConfig`` thresholds):
    handing     in the region (exit hysteresis ``region_exit_m``), lifted, a robot on it, and the giver holds it with
                both fingers or the receiver has at least one finger on it
    received    receiver holds it with both fingers, the giver touches nothing of it and its grasp site is at least
                ``clear_m`` from the surface (the goal; a task enters its next stage with "received, lifted, touching
                nothing but the receiver")
Progress (span 1.25): 0.25 x receiver fingers + [receiver holds] x (0.25 x (2 - giver fingers)
                      + 0.25 x [giver has let go] x min(gap_give / clear_m, 1))
Failures: dropped (no robot touches it while more than ``drop_m`` up); touched anything but the two arms (decision
2 of the inhand stage table: the handoff happens in the air; setting it down and re-picking is not a handoff).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HandoffConfig:
    # Region where the handoff happens (CoM, world frame). inhand: scripted handoffs passed with the item centre at
    # x 0.49-0.60, y ~0, z 0.83-0.95, the policy's one handoff (exp12 world-05) at x 0.53, y 0.03, z 1.09; |y| <= 0.10
    # is the stock far band's inner edge, so being in the region can never already be far-side success.
    region_lo: tuple = (0.45, -0.10, 0.83)
    region_hi: tuple = (0.70, 0.10, 1.15)
    region_exit_m: float = 0.10  # leaving the region by more than this while handing over = back to the previous stage
    # Giver's pinch centre this far from the object's nearest surface point = clear. The approved table said 6 cm from
    # the CoM; the surface distance is never larger than the CoM distance for a site outside the object, so this is
    # the stricter test (long items grasped at an end are > 6 cm from the CoM while still held). Scripted handoffs
    # (A opens, lifts 10 cm, backs off 6 cm) end 8-14 cm clear.
    clear_m: float = 0.06
    air_m: float = 0.03  # lifted: same threshold as pick
    drop_m: float = 0.04  # free and more than this above rest = dropped (lift threshold + 1 cm, as put_relative)
    span: float = 1.25


def region_gap(com: np.ndarray, lo, hi) -> np.ndarray:
    """Euclidean distance of each CoM to the box [lo, hi] (0 inside)."""
    com = np.asarray(com, dtype=np.float64).reshape(-1, 3)
    d = np.maximum(np.maximum(np.asarray(lo) - com, com - np.asarray(hi)), 0.0)
    return np.linalg.norm(d, axis=1)


def robot_on(obs) -> np.ndarray:
    return obs["touch_give"] | obs["touch_recv"]


def handing(obs, was_handing: np.ndarray, cfg: HandoffConfig) -> np.ndarray:
    gap = region_gap(obs["com"], cfg.region_lo, cfg.region_hi)
    inside = gap <= np.where(was_handing, cfg.region_exit_m, 0.0)
    holding = (obs["f_give"] == 2) | (obs["f_recv"] >= 1)
    return inside & (obs["lift"] >= cfg.air_m) & robot_on(obs) & holding


def received(obs, cfg: HandoffConfig) -> np.ndarray:
    return ((obs["f_recv"] == 2) & (obs["f_give"] == 0) & ~obs["touch_give"]
            & (obs["gap_give"] >= cfg.clear_m))


def received_in_air(obs, cfg: HandoffConfig) -> np.ndarray:
    """The handoff goal: received, lifted, touching nothing but the receiver."""
    return received(obs, cfg) & (obs["lift"] >= cfg.air_m) & ~obs["env_touch"]


def progress(obs, cfg: HandoffConfig) -> np.ndarray:
    held = (obs["f_recv"] == 2).astype(np.float64)
    let_go = (obs["f_give"] == 0) & ~obs["touch_give"]
    return (0.25 * obs["f_recv"]
            + held * (0.25 * (2 - obs["f_give"]) + 0.25 * let_go * np.minimum(obs["gap_give"] / cfg.clear_m, 1.0)))


def failure(obs, k: int, cfg: HandoffConfig) -> str:
    if not robot_on(obs)[k] and obs["lift"][k] > cfg.drop_m:
        return "dropped during the handoff"
    if obs["env_touch"][k]:
        return "touched the table during the handoff"
    return ""
