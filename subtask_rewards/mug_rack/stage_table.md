# hang_mug_on_mug_rack: stages (APPROVED 2026-11-14, all default decisions)

Outcome (2026-11-14): pick PASSED; reach STOPPED (G5 by the letter); align/insert/release STOPPED (G3). See
`STATUS.md` and `design.md`.

Target per episode: **one not-yet-hanging mug** — the one nearest either gripper at the start (fixed for the
episode). Template F (insertion / hanging). Baseline: exp12 OpenWAM canon 14B, 0/10 after bit-exact re-scoring with
the current scorer (8 of the recorded 8/10 were cut short by the old false positive — not a policy verdict).
SEED_BASE = 1_200_000_000 (950M is taken by nuts_bolts). G1 approved by the user on 2026-11-14 with the first
option of every decision below.

Geometry the table rests on (measured, `STATUS.md` phase 0):
- 6 pegs, Ø 13.2–13.9 mm (median of the collision piece; up to ~18 mm at the trunk joint), 7.4–8.5 cm long,
  pointing 19–26° above horizontal, tips at z ≈ 0.91 / 0.97 / 1.05 (table 0.75). Pegs at the lowest and highest
  levels point ±x of the tree frame (towards / away from the robot), the middle pair ±y. Tree yaw ±0.25 rad.
- Handle hole (inscribed circle in the handle's mid-plane, scale 1): 16 models 18–30 mm; **mug_1 8 mm, mug_7 11 mm,
  mug_10 12 mm — narrower than any peg, these three cannot be threaded at all**. Mug scale 0.9–1.1 per reset.
- Cup Ø at the rim 70–111 mm; gripper opens 95 mm. mug_1/4/7/10/12 (rim Ø 99–111 mm) cannot be grasped across the
  rim; body grasps lower on tapered mugs or rim/handle grasps only.
- "Threaded" (stock scorer since this audit) = a tree piece passes through the measured handle hole.

| stage | prompt (unique) | precondition (= start) | goal (= next precondition) | progress (span, eps) | failures (−1) |
|---|---|---|---|---|---|
| reach | "move a gripper to the {color} mug" | target on the table / not held | grasp site ≤ r_near of the target's CoM, or a finger touches it | −dist (0.30 m, 5 mm) | target pushed off the table; target knocked over (tilt > 60° while not held) |
| pick | "grasp the {color} mug and lift it" | near/touched, not lifted | both fingers of one arm on the target, **no finger on the handle pieces**, lifted ≥ 3 cm | 0.25·fingers + 0.5·min(lift/3 cm,1)·grasped (1.0, 0.05) | back to reach after a lift (dropped); off table |
| align | "line the {color} mug's handle up with a free peg" | held, lifted, handle free, not aligned | handle-hole centre within 1.5 cm of a free peg's axis line, axially 0–4 cm beyond that peg's tip, hole axis ≤ 30° from the peg axis, still held, mug touching nothing but the fingers | −(radial gap + 0.5·axial gap + 0.02·angle/30°) to the best free peg (0.40, 5 mm) | dropped (no finger, > 4 cm above table, not on tree); a hanging mug knocked off |
| insert | "slide the {color} mug's handle onto the peg" | aligned (or threaded < 2 cm deep), held | **threaded** (stock rule) with the hole centre ≥ 2 cm along the peg past its tip, still held | −(radial gap of hole centre to the peg axis) − (2 cm − depth)₊ (0.06, 2 mm) | dropped off the tree; mug resting on the tree un-threaded with no finger (balancing hack) for > 2 chunks; a hanging mug knocked off |
| release | "let go of the {color} mug on the rack" | threaded ≥ 2 cm, held | stock per-mug hanging (in band, threaded, handle touch ≥ 1 cm above CoM, touches only the tree/other mugs) **and** no finger within 2 mm **and** at rest (< 2 cm/s) for 5 actions | −(finger contact) + 0.05·released − 0.5·(un-threaded depth loss) (0.10, 5 mm) | released and fell off / rests un-threaded; a hanging mug knocked off |

Every stage, also −1: the target or any already-hanging mug falls to the floor; an already-hanging mug stops
hanging (knocked off). Other mugs may be nudged on the table.

Thresholds and where they come from (to re-measure in phase 2/3; none is tuned yet):
- r_near = 10 cm: mug-flip measurement (grasp site 5–10 cm from a held mug's CoM); re-measure from mug-rack grasps.
- lift 3 cm, drop height 4 cm, hold 5 actions, rest 2 cm/s: mug flip / put_relative / bottles conventions.
- align radial 1.5 cm: hole Ø 18–30 mm vs peg Ø 13.5 mm leaves 2–8 mm radial slack; 1.5 cm is the template's
  "≤ 2 cm" tightened so that insert starts are not hopeless. Axial 0–4 cm: pre-insert offset used by the
  feasibility probe (7 cm worked as the approach start; 4 cm keeps the stage short).
- insert depth 2 cm: settled real hangs sit 2–6 cm along the peg (to measure exactly from the 57 settled hangs).
- "free peg": no other mug threaded on it. Pegs by the tree's lowest level are 12–17 cm above the table: a hanging
  mug of 9–10 cm height can touch the base disc → scorer rejects; prefer middle/upper pegs (decision 6).

Scripted behaviours to run (phase 4):
| stage | behaviour | expect |
|---|---|---|
| reach | move to the mug body / move away / idle / push the mug | +1 / 0 / 0 / 0 or −1 |
| pick | rim or body grasp + lift (handle free) / grasp the handle + lift / close only / lift then open | +1 / 0 (goal needs a free handle) / 0 / −1 |
| align | carry so the hole is in line with a free peg / carry next to a peg but rotated 90° / shake / open mid-air | +1 / 0 / 0 / −1 |
| insert | slide along the peg axis 4–6 cm / push sideways into the peg / hold still / open before threading | +1 / 0 (or −1 if dropped) / 0 / −1 or 0 |
| release | open, back off along −approach / keep holding / open then yank the mug back off the peg / open with the hole only on the peg tip | +1 / 0 / −1 / −1 or +1 (measure) |
| any | drop the mug onto the tree from above (audit: 400+ settles rest with a peg in the cup) / lean the handle on a peg with the body on another peg (old false positive) | never +1; −1 if it ends un-threaded at rest |

Start setup per stage (grasp source, how the precondition is made true, checks):
- reach: the reset scene; skip seeds whose target is mug_1/7/10 (cannot be threaded — decision 2).
- pick: open gripper around the resting target (mug-flip grasp library, same mug assets; filter grasps that touch the
  handle pieces).
- align: target held 8–15 cm above the table by a handle-free grasp (mug-flip library via kit `hold_object`; the
  library grasps include rim and body grasps); chain from the pick solver when it exists.
- insert: target held with its handle hole on a free peg's axis, 2–4 cm outside the tip, in the orientation of a
  settled real hang of that model on that peg (from the settle search, 57 real hangs over 5 models so far — enough
  only for a pilot; a per-model/per-peg hang library must be grown for 100 starts). Chain from the align solver.
- release: as insert but threaded 2–4 cm deep (end state of the insert solver; teleported fallback from settled
  hangs + 1 cm lift).
All held starts: 10-chunk idle slip check (G3).

**Decisions for you** (I picked the first option unless you say otherwise):
1. **Threading rule in the stock scorer** (benchmark change, done on this branch during the audit): a peg must pass
   through the measured handle hole / revert to the handle-touch rule (keeps 9 of 59 false positives, including the
   old regression pose `HANDLE_ON_BRANCH`).
2. **Unthreadable mug models** (mug_1, mug_7, mug_10: hole < peg): never the target of any stage, and episodes in
   which they are the only remaining mug are skipped / keep them (stages beyond pick can never succeed for them) /
   change the randomizer to drop them from mug_tree (benchmark change).
3. **One mug per episode**: the nearest not-yet-hanging mug, other mugs only matter via failures (knocking a hanging
   one off) / sample the target uniformly / require all mugs (stock task; no subtask split).
4. **Handle must stay free in pick** (no finger on handle pieces), so every pick success can be threaded / allow
   handle grasps (then align must re-grasp; adds a regrasp stage).
5. **Release placement**: release must leave the mug hanging at rest with no finger, i.e. "placed, not dropped" — the
   mug may fall a few mm onto the peg / also allow dropping the mug onto the peg from above (the audit shows this
   almost never threads, so it would only add hack surface).
6. **Pegs**: any free peg counts; align progress uses the best free peg (min over pegs) / fix one target peg per
   episode (closest to the arm, upper levels only) — fixing makes progress unambiguous but cuts coverage.
7. **Stage split**: five stages (reach, pick, align, insert, release) / merge align+insert into one "hang" stage
   (sparser, matches the template's carry+insert less closely).
8. **Feasibility / stop**: proceed with reach + pick (template A-like, expected to pass) and treat align / insert /
   release as a pilot with the STOP rule applied per stage (see feasibility in STATUS.md) / stop the task now.
