# grab_specific_object_from_clutter — stages (APPROVED 2026-11-13)

**Approval:** user approved the proposed defaults with “lets continue” after the concrete approval request.
Selected choices: reach/pick/carry/place; fixed prompted target; controlled placement with floor support; wrong-object lift fails; harmless distractor nudges allowed but off-table loss or moved/tipped box fails; direct reachable-carry pilot with explicit coverage; raw XY benchmark reward unchanged.
Any pending-approval wording below is historical and superseded by this approval.


Target per episode: the **prompted target**, identified by `target_object_joint` and randomization metadata. Never
replace it with whichever distractor is nearest. Template A. Baseline: unavailable; see `STATUS.md`.

The target box is randomized near a table side. Only 13/40 audit resets have target and box on the same y-sign
side; actual carry reachability has **not** been measured. Proposed pilot: direct single-arm carry starts only
where a verified collision-free IK path exists, reporting eligible/candidate counts separately. Cross-table
handoffs are a separate follow-up unless you approve including them now.

Each stage trains one episode from its own valid start. Transitions hold for 5 actions; each terminal goal is
the next stage's precondition. Scene changes use live body frames, not the original box pose.

| stage | live prompt | precondition (= start) | goal (= next precondition) | progress (span, eps) | failures (−1) |
|---|---|---|---|---|---|
| reach | move a gripper to the {target label} in the clutter | target resting in original scene; no later precondition | nearest grasp site ≤5 cm from target CoM, or finger contacts target | −target distance (0.30 m, 5 mm) | target pushed >3 cm without grasp; target/distractor off table; moved/tipped box; wrong object lifted |
| pick | grasp the {target label} and lift it from the clutter | near target; not yet lifted by a two-finger grasp | both fingers of one gripper on target and lift ≥3 cm above reset | 0.25·finger count + 0.5·clip(lift/3 cm)·grasped (1.0, 0.05) | regress to reach; same scene/object failures as reach |
| carry | carry the {target label} over the box | target held and lifted, outside opening alignment | target origin inside repaired scorer's box-local XY goal while held, with lowest collision point ≥rim+1 cm | −3-D distance to above-opening region (0.50 m, 5 mm) | release outside opening; regress to pick/reach; target/distractor off table; moved/tipped box; wrong object lifted |
| place | place the {target label} inside the box and let go | target held/touched above the opening, geometry clears rim | repaired stock success **and** no finger on target **and** speed <2 cm/s **and** box-floor support contact | −gap between target lowest collision point and box floor +0.05 once released (0.15 m, 5 mm) | release with >4 cm clearance above floor; lands outside; regresses; off-table target/distractor; moved/tipped box |

No mid-air contact-free state counts as terminal solely because `qvel` happens to be zero. A floor support check
prevents that loophole. A target bridging the rim does not count under the proposed final-stage goal. A naturally
moving target remains in its prior stage until a stable transition or a latched failure resolves it.

Thresholds and origins:

- Goal XY half-sizes: 6 and 5.5 cm — existing box-local scorer site; floor +1.2 cm and rim +10.4 cm — collision
  geometry in the audited models. Implementation must measure these per model, not hard-code the example.
- Near enter/exit 5/7 cm, lift 3 cm, drop clearance 4 cm — shared pilot defaults, **not yet calibrated on clutter
  grasps**. Re-measure against actual target categories in phase 2/3 without weakening validation gates.
- Opening exit hysteresis 1 cm beyond its XY boundary; lift exit 2 cm; 5-action debounce — proposed defaults.
- Box moved >3 cm XY from reset or tilted >20° — explicit anti-hack decision; `grab_clutter` stock success follows
  a tipped/moved box and does not reject it.
- Push threshold 3 cm — disallow solving a "grab" episode by pushing the target. Distractors may be displaced while
  reaching the target; only wrong-object lifting or off-table loss is proposed as failure.
- Default chunk reward: +1 goal, −1 failure; otherwise 0.5·Δprogress/span. No per-chunk sign reward.
- Each spec prompt must be unique (`clutter subtask: …`); the live prompt includes the target label.

Scripted behaviours (phase 4):

| stage | behaviour | expect |
|---|---|---|
| reach | approach target / move away / idle | +1 / 0 / 0 |
| reach | approach wrong object / push target | 0 / −1 |
| pick | grasp and lift target / close only / lift open / pick distractor | +1 / 0 / 0 or −1 on regression / −1 |
| carry | carry to box / wrong direction / open early / idle | +1 / 0 / −1 / 0 |
| place | lower, support, release / hold above box / release high / leave opening | +1 / 0 / −1 / −1 |
| all | move box to target / tip box / knock distractor off table | −1 / −1 / −1 |

Starts, in order of preferred grasp source:

1. Reach: fresh randomized scene; skip reset failures or already solved states and report counts.
2. Pick: successful reach-solver end states. Fallback: sampled target grasps, open around the target, no push or
   distractor collision. Different categories may need side grasps; top-down suitability is not established.
3. Carry: successful target pick states; verify direct path to the box. Preserve the full scene and metadata.
   Reject unreachable direct carries explicitly and report coverage; do not silently turn them into other targets.
4. Place: successful carry end states above the opening, with a grasp that can lower without fingers colliding
   with the rim. Held starts must pass 10 idle chunks (320 actions) before inclusion.

**Decisions for approval** (proposed first options):

1. **Direct single-arm pilot with explicit eligible-start coverage** / include handoff within carry now. The 27/40
   opposite-side resets are a reachability warning, not a measured exclusion rate.
2. **Controlled placement with floor support before release** / allow dropping above the opening, as approved for
   bottles. Bottle approval does not automatically apply to clutter.
3. **Wrong object lifted = failure** / no credit but episode continues. Target identity always remains fixed.
4. **Distractors may move, but none may fall off table; box movement/tipping fails** / stricter no-disturbance rule.
5. **Keep raw XY reward for benchmark compatibility; use only the new staged reward for RL** / revise the whole-task
   reward too. Raw XY reward can equal 1 for a target hovering outside the physical opening; success now rejects
   the above-rim/below-floor cases, and the proposed stage terminal additionally requires support/rest/release.

G1 is pending. This document is reviewable design only; no clutter subtask code or training starts exist yet.

**Implementation note (2026-11-13, after G4/G5 round 1):** the reach goal additionally requires the target to be
undisturbed when pick is entered (speed ≤ 5 mm/s, moved < 3 cm). This only tightens the approved "push without
grasp is not a reach" rule; evidence in STATUS.md repair log.

## Amendment 2026-11-14 — handoff stage (user-approved: add a handoff so the 55 % IK-unreachable scenes are covered)

Chain: **reach → pick → handoff → carry → place → done**; `handoff` only in handoff scenes, direct-reach scenes keep
reach → pick → carry → place with unchanged starts and solvers.

**Roles and the "needs handoff" test.** The box is always at y ≈ ±0.515 m; the arm bases are at y = ±0.31 m. The
**box arm** (left for a +y box, from the reset box position) is the only arm that carries to the box: in the direct
pilot, 53/53 scenes whose picking arm (nearest grasp site at reset = the pick-start arm) was on the other side were
"IK unreachable", and 45/47 same-side scenes had a verified direct carry (2 same-side exclusions: seeds +20, +79).
So: **giver = the non-box arm, receiver = the box arm, in every scene**. A scene *needs a handoff* (starts, coverage)
iff its picking arm ≠ box arm (53/100 of SEED_BASE + 0..99). In the reward, the label decides from state alone:
whoever holds the target, the giver holding it lifted is `handoff`, the box arm holding it alone is `carry`.

| stage | live prompt | precondition (= start) | goal (= next precondition) | progress (span 1.75, eps 0.02) | failures (−1) |
|---|---|---|---|---|---|
| handoff | hand the {target} to the other arm | giver holds it (both fingers), lifted ≥ 3 cm, touching nothing but the robot (= pick goal in a handoff scene) — or the `handoff_stage.handing` test (in region M, exit 10 cm) | `handoff_stage.received_in_air`: receiver both fingers, giver no contact and its pinch centre ≥ 6 cm from the target surface, lifted ≥ 3 cm, touching nothing but the receiver | `handoff_stage.progress` (0 … 1.25) − min(CoM distance to M, 0.5) | `handoff_stage.failure`: dropped (no robot within 2 mm, > 4 cm up); touched table / box / clutter (setting it down and re-picking is not a handoff); scene failures (box moved/tipped, distractor off table, wrong object lifted); back to pick (e.g. receiver has it alone outside M before the giver is clear) |

- `HandoffConfig` = the inhand defaults, unchanged (M: x 0.45–0.70, |y| ≤ 0.10, CoM z 0.83–1.15; clear 6 cm; air
  3 cm; drop 4 cm). Pinch centre link_6 (0, −0.031, 0.125) (inhand finding: the grasp site is 3 cm off the pads).
- Observation choice: `touch_give/touch_recv` count a finger geom within 2 mm of the target as touching. Measured:
  a receiver holding the target with both fingers lost every contact for one physics step while the target settled
  in its pinch, which read as "dropped" (seed +14). Finger counts still use real contacts.
- Pick in a handoff scene: its goal is `handoff` (label > pick), which additionally needs the lifted target to touch
  nothing but the robot (a lift still dragging clutter stays in pick). Direct scenes: unchanged (carry).
- Carry in a handoff scene: start = receiver holds it alone after a handoff, giver clear. The giver touching it again
  sends it back to `handoff` (−1). A target the giver holds can never be in `place` (place now also requires no
  giver contact; the giver cannot reach the box anyway).

Scripted behaviours (phase 4), handoff: solver +1; drop_between −1 (giver opens in M before the receiver has it);
both_hold never +1 and return ≤ 0.5; set_down_regrasp −1 (giver puts it back on the clutter, receiver re-grasps);
giver_alone never +1 (giver carries it towards the box by itself); idle 0; box_move / box_tip / distractor_loss −1.
Carry in handoff scenes: the carry behaviours + regrasp_giver −1.
