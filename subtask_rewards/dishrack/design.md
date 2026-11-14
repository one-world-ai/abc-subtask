# Dish-rack subtasks (load_plates_into_dish_rack) — STOPPED

Task: `load_plates_into_dish_rack` (env task `dishrack`). The user approved the stage table on 2026-11-14 (G1, all
five default options). Gate numbers, repair rounds and evidence paths are in `STATUS.md`; this file says how the
subtasks work and why none of them ships.

**Outcome.** reach/pick: STOPPED at G1 (no scripted table pick of a flat plate). carry: STOPPED at G3 (65/100 valid
starts). insert: STOPPED at G3 (72/100). release: STOPPED at G4/G5 (scripted release keeps the plate in the rack on
8/16 planned starts; margin 0.07). The code, specs and 14 tests are in place so the stages can be resumed.

## Use (for resuming; not validated for training)

```python
env = abc_sim.make_env(task="load_plates_into_dish_rack__release", render_cameras=False)
obs, info = env.reset_to_stage(seed, start_row)   # rows: ../simulation/outputs/subtask_rewards/dishrack/starts/<stage>.npz
reward, done = env._task_evaluator.chunk_reward()  # after each 32-action chunk
```
Suffixes `__carry`, `__insert`, `__release`. CPU, single world; the qpos-only path raises (the goals need real
contacts and velocities). The first `begin` per rack/plate geometry computes the slot map (0.25-0.65 s, cached).

## Geometry (`dishrack_geom.py`)
- **Jaws**: pinch centre `PINCH = (0, -0.043, 0.134)` in link_6 (pads at y -0.052..-0.033, z 0.121..0.1465; open
  9.46 cm; `probes/dr_fingers.py`), not the stagekit grasp-site axis. Used only here; stagekit is unchanged.
- **Slot map**: per scene, the target plate stood upright (normal along rack x or y) on 3 lines across the rack
  footprint, stepped 5 mm along the axis and lowered 5 mm at a time until it would penetrate a rack collision geom by
  > 1 mm. A slot position sinks >= 3 cm below the tops of the rack parts within 2 cm of the plate's plane. Slots are
  runs of such positions: centre t, line u, seat depth profile, entry height. 19-43 slots per scene.
- **Grasp**: top-of-rim pinch (1.5 cm inside the rim, jaws across the plate), approach tilted 30-75 deg within the
  plate plane. Reach needs the tilt (`pilot_reach2.py`: collision-free IK at entry and seated poses for 10-36 slots
  per scene, all at 30-75 deg). The top pinch hangs in equilibrium; an off-top pinch creeps 1-2 deg per chunk
  (gravity torque; the finger servo gives only ~1-2 N on a 4-8 mm rim: kp 400 N/m x opening).

## Stages (`dishrack_stages.py`), one target plate per episode (the unracked plate nearest a gripper)
| stage | in it when | goal | progress / span |
|---|---|---|---|
| carry | both jaws of one gripper on the plate, off the table | held, upright, plane <= 15 deg of a slot plane, centre <= 2 cm (along the slot axis) of a slot centre on the nearest line, over the footprint, lowest point <= entry + 3 cm | −(xy distance of the centre to the nearest slot centre + 0.5 x plane angle) / 0.30 m |
| insert | carry goal | stock in-rack geometry (spatial, footprint, rack contact, upright) while a jaw touches, at a slot, lowest point <= 1 cm above the map's seat depth at the plate's position | −(height above that seat) / 0.15 m |
| release | insert goal | fixed stock success (no gripper contact), still (< 1 cm/s) for 64 actions, every jaw >= 5 cm away | 0.5 x (in rack, no jaw) − (nz − nz_start)+ / 1.0 |

Hysteresis: 2.5 cm / 20 deg / +4 cm on the carry goal, 1.5 cm on the seat; debounce 5 actions, done 64.
Failures (−1): rack moved > 3 cm or tilted > 10 deg; a plate racked at the start knocked out; carry/insert: dropped
(no jaw for 5 actions before the insert goal), hit the table outside the footprint; insert: flat on the rack > 2
chunks, jammed (held, pressed with > 2 plate weights, < 3 mm descent in 4 chunks); release: pulled out (held, lowest
point +2 cm), fell over (nz > 0.55) or left the stock in-rack geometry for 5 actions without a jaw.

Reward: `chunk_reward()` = +1 / −1 terminal, else 0.5 x Δprogress / span (potential-based; 1 big move = 8 small).

## Starts (`dishrack_starts.py`, SEED_BASE 1_300_000_000)
- carry: upright plate at its own spot (or 1/3, 2/3 of the way to the rack), random yaw, 6-10 cm up; a grasp is used
  only if it can also reach some slot entry; weightless close, gravity hold. **65/100 valid**: 30 seeds have no
  upright held pose in IK reach with a slot-reaching grasp.
- insert: chained through the scripted carry (16-35 per build), else the plate set 1.5 cm above a slot entry and
  held. **72/100**: the entry pose is out of collision-free reach for the rest within 10 s.
- release: chained through carry + insert, else seated in a slot and pressed down; kept only if the plate stays in
  the rack when the jaws open in place. 80/100 before that check, **75/100** with it.

## Checks (`../simulation/experiments/subtask_rewards/dishrack/`)
- `scripted_rollouts.py` behaviours. carry: solver, release_early, idle, lay_flat_on_rack. insert: solver,
  lean_on_outside, drop_from_above, jam. release: solver, yank, idle. Only release was run (the others stopped at G3).
- Release result: solver +1 8/20 (16 planned), yank −1 on all 17 planned, idle 0 on 20/20; margin 0.07. The 7
  planned failures: the plate topples or falls out after the jaws open (`scripted/release/solver_*.json`).
- Scripted carry (pilots, not a gate): +1 on 15 of 17 plannable starts; plan failures are IK reach.

## Open questions for the user
1. The kinematic slot map does not check that a slot holds a released plate. A physical release test per slot
   (dr_slots style, ~0.15 s per slot, cached) would make the insert goal mean "in a slot that holds it". It would
   probably fix release starts, but it makes `begin` slow for new geometry.
2. The release "left the rack" check also fires on a transient loss of rack contact while the plate settles (2 of the
   7 solver failures end back in the rack). Not changed because the repair rounds were used up.
3. The reach limit (upright plate with a tilted top pinch) is the main cause of the carry and insert stops. A
   two-arm approach or a different grasp family (side pinch with the plate tipped toward the robot) would need its
   own reach study.

## Codemap
`abc_sim/task_eval/dishrack_geom.py` (jaws, slot map, grasps), `dishrack_stages.py` (scene, classify, progress,
tracker), `dishrack_subtask.py` (evaluator, failures), `dishrack_starts.py` (motions, starts), specs in
`task_specs.py`, `registry.py` (`dishrack_subtask`), tests `tests/test_dishrack_subtasks.py`. Tooling and pilots in
`../simulation/experiments/subtask_rewards/dishrack/`.
