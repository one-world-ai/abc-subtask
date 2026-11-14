# load_plates_into_dish_rack: proposed stages (for approval)

Target per episode: one plate (`plate_count` 1–4 in the stock scene; the stage target is the not-yet-racked plate
nearest a gripper). Template: F (insertion). Baseline: exp12 3/60 → **1/60 after re-scoring** (worlds 12 and 54 were
plates leaning on the rack's outside; world 49 is a real slot insertion). Evidence: `STATUS.md`, phase 0.

## Recommendation: STOP the table-pick half, offer the rack half as a reduced task

The rack side is sound: every one of the 12 rack models takes an upright plate lowered into it and released
(`probes/dr_slots.py`: slots on every model, 2–5 cm pitch, 0–35 mm play for a 1.3 cm-deep plate; released plates stay
upright and score). The table side is not: a plate lying flat (rim 0.5–2 cm above the table, 17.5–23 cm across,
25–75 g) can only be taken by a rim pinch, and my scripted rim pinch never lifted a plate clear of the table
(`probes/dr_rim_grasp.py`). Details:
- Straight-down pinch: the jaws reach the rim to within 6 mm and close on it with both jaws in 3 of 4 scenes. On
  lifting, the plate pivots in the pinch with its far edge dragging on the table: after a 10 cm lift the centre of
  mass is up 4 cm and the plate is tilted about 35°. After a 16 cm lift the plate is lost in 4 of 4 scenes, and a
  25 cm lift is out of IK reach for the straight-down wrist in 24 of 24 attempts.
- Tilted pinch (30° or 50° lean): the jaw heel hits the table 2–6 cm above the rim. 0 of 48 attempts touched the plate.
- What it would take: a regrasp sequence (pinch, drag or tilt the plate up against something, regrasp at the top
  edge) or a two-arm tilt-and-catch. That is a multi-step manipulation planner, not one stage, and a policy did it
  once in 60 episodes (exp12 world 49).

Per the stop rule ("a stage needs a capability the robot doesn't have"), I recommend **STOP reach/pick**. The
**carry/insert/release** stages can still be built with grasp source 3 below, if you want the reduced task (decision 1).

## Draft table (only if decision 1 = build the rack half)

| stage | prompt | precondition (= start) | goal (= next precondition) | progress (span, eps) | failures (−1) |
|---|---|---|---|---|---|
| carry | `carry the plate upright above the dish rack` | plate pinched at its top edge by one gripper, both jaws on it, plate off the table, \|normal_z\| ≤ 0.55 (upright), plate bottom outside the slot zone | plate bottom over the rack footprint, within 2 cm (rack frame, along the slot axis) of a measured slot centre, plate plane within 15° of the slot plane, plate lowest point ≤ 3 cm above the slot entry (tine top), still held | −(xy distance of the plate bottom to the nearest slot centre) − 0.5·(plane angle error in rad); span 0.30 m, eps 0.005 | dropped (no jaw on the plate and plate lower than the slot entry, outside a slot); plate hits the table; rack moved > 3 cm or tilted > 10° |
| insert | `lower the plate into the rack slot` | goal of carry | footprint-fixed stock success with the plate still held: plate in the rack, \|normal_z\| ≤ 0.55, lowest point ≤ 1 cm above its depth from `dr_slots` (seated), rack contact | −(height of the plate's lowest point above the seated depth); span 0.15 m, eps 0.003 | dropped outside a slot; plate lying flat on the rack (\|normal_z\| ≥ 0.75) for > 2 chunks; plate jammed on a tine (no descent for 4 chunks while held, contact force above threshold); rack disturbed as in carry |
| release | `release the plate in the rack` | goal of insert | stock success (fixed scorer), no jaw contact, plate speed < 1 cm/s for 2 consecutive chunks, gripper ≥ 5 cm from the plate | +release bonus 0.5 when the jaws open while the plate is seated; −(normal_z − seated normal_z)⁺ | plate falls over (\|normal_z\| > 0.55) or leaves the rack after the release; rack disturbed |

Thresholds and where they come from:
- footprint tolerance 1 cm: the fixed stock scorer (`footprint_tolerance_m`), rack wire thickness. Lean-outside
  plates have their bottom 3–9 cm outside the footprint (`dr_slots_disagree.py`).
- upright \|normal_z\| ≤ 0.55: stock evaluator option `upright_normal_z_max`.
- slot centres and seated depth per rack model: measure from `dr_slots.py` (per model, per scale; never hard-code
  one model).
- 2 cm / 15° carry alignment: slot play is 0–35 mm for the thinnest plates (`dr_slots.py`). Calibrate on the deepest
  plates first (plate depth 1–5.5 cm; only `plate_13`, 1.3 cm deep, was measured). Bowls may fit no slot (open question).
- jaw geometry (`dr_fingers.py`): the jaws close along link_6 x, open 9.46 cm, 2 cm thick, pad 2.5 cm long. The pinch
  centre is at link_6 (0, −0.043, 0.134), not on the link_6 axis as `stagekit`'s 0.1347 grasp-site constant assumes.
  Handled locally (`dishrack_geom.PINCH`, used for grasp synthesis); `stagekit` is unchanged for other tasks.

Scripted behaviours to run (phase 4):
| stage | behaviour | expect |
|---|---|---|
| carry | S: IK move with the plate upright above the nearest slot | +1 |
| carry | release_early (open over the table) | −1 |
| carry | idle | 0 |
| carry | lay_flat_on_rack (rotate the plate horizontal and lower it) | −1 or 0, never +1 |
| insert | S: lower straight down along the slot | +1 |
| insert | lean_on_outside (lower outside the long side, tilt against the rack) | 0 / −1, never +1 (the exp12 false positive) |
| insert | drop_from_above (open 5 cm above the tines) | 0 / −1 |
| insert | jam (lower 2 cm off the slot centre onto a tine) | 0 or −1 (no +1) |
| release | S: open, then retreat 10 cm | +1 |
| release | yank (retreat without opening) | −1 (plate pulled out) |
| release | idle (keep holding) | 0 |

Start setup per stage (grasp sources):
- carry: source 3. The plate is set upright and weightless near the table (random pose in reach), the jaws close at
  its top edge (pinch centre 1.5 cm below the rim, closing across the plate thickness), then 10 idle chunks under
  gravity to check the hold. A plate pinched at its top edge hangs with its centre of mass under the pinch, so no
  pivot torque; exp12 world 49 held a plate this way. Not yet tested.
- insert: chain from the carry solver, else source 3 at the carry goal pose above a slot.
- release: chain from the insert solver, else source 3 with the plate seated in a slot (the `dr_slots.py` seated
  pose, jaws closed at the top edge). This source is close to tested: the released poses score on all 12 rack models.
- reach/pick: STOPPED (no grasp source: the scripted pick fails, and the baseline has 1/60 picks to replay).

**Decisions for you** (I picked the first option unless you say otherwise):
1. Table pick: **STOP reach/pick and build carry/insert/release from synthesized upright edge grasps** / stop the
   whole task (shipping nothing) / fund a regrasp planner (tilt the plate against the rack or table edge, then regrasp
   at the top edge) as a separate project.
2. What counts as "in the rack": **any upright plate over the footprint that touches the rack** (the fixed stock rule;
   it accepts a plate leaning diagonally inside the rack, e.g. `settle/rack7_004.png`) / require a true slot
   (tines on both faces, \|normal_z\| ≤ 0.3). The stricter rule needs per-model slot geometry in the scorer.
3. Multi-plate scenes: **one plate per episode (the nearest unracked); the others only matter through failures (a
   racked plate knocked out = −1)** / always start from one-plate scenes (`plate_count` 1).
4. Deep plates and bowls (depth up to 5.5 cm): **keep them and let starts fail where no slot fits (count it under G3)**
   / exclude variants whose depth exceeds the narrowest slot of the sampled rack.
5. Release: **must open and retreat with the plate staying seated** / count the insert goal (held in the slot) as the
   task end.
