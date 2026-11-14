# sort_nuts_and_bolts — stages (APPROVED 2026-11-13; drop amended 2026-11-14; handoff added 2026-11-14)

**Handoff amendment (user approved adding a handoff, 2026-11-14).** Chain: reach → pick → **handoff** → carry → drop.
- *Cross-side* part: its rest CoM y and its assigned bin's rest y have opposite signs (fixed per episode from the rest
  positions). Measured on 1000 seeds: 157 cross-side targets, **all bolts** (bolts sit at +y, the orange bin at −y);
  no nut target is ever cross-side under the nearest-part rule. Receiver = the arm on the bin's side; giver = the other
  arm (the arm that picks it — in every screened seed the near arm is the giver). Same-side parts: no change at all
  (labels, progress, failures; regression test against a frozen copy of the old `classify`).
- Cross-side pick goal (held by the giver, lifted ≥ 3 cm) = **handoff** precondition. If the receiver picks it itself
  the pick goal is carry directly (it meets the handoff goal).
- Handoff stage: `handoff_stage.py` unchanged, configured by `HandoffConfig(region = the whole table volume
  x 0.25–0.90, |y| ≤ 0.50, z 0.70–1.40)` (judgement call: the handoff precondition is wherever the giver picked it, so
  the middle box cannot be a stage test). Goal = `received_in_air`: receiver both fingers, giver no contact and its
  pinch centre (link_6 (0, −0.031, 0.125), not the grasp site) ≥ 6 cm from the part's surface, lifted ≥ 3 cm, touching
  nothing but the receiver. Failures: `handoff_stage.failure` (dropped while > 4 cm up; touched anything but the two
  arms — table, bins, other parts) + the task-wide ones (bins moved/tipped, sorted part knocked out, part off table,
  wrong bin) + fall back to pick. Progress = `handoff_stage.progress` − min(CoM distance to a middle box x 0.40–0.55,
  |y| ≤ 0.05, z 0.82–0.95, 0.30) (span 1.25 + 0.30 = 1.55, eps 0.02); the middle term only shapes (judgement call).
- Cross-side carry/drop: carry needs the receiver holding it alone (received, or received after carry began); drop
  entry needs the giver not touching it. New failure in carry/drop: **giver touched the part again** (−1). Goals,
  other failures and thresholds of carry/drop unchanged.
- Starts (100 per stage, the first 100 cross-side seeds from SEED_BASE): handoff = giver holds it lifted 6–12 cm where
  it lay, grasp high on the bolt (CoM 0.5–1.3 cm below the fingertips, link_6 z .152–.160, `HIGH_PAD_Z`; 2.4–3.2 cm of the bolt sticks out below them); cross carry = the
  post-handoff state (receiver's side grasp in the middle, giver home); cross drop = the carry solver's end state
  (bolt head-up over the opening in the receiver's rolled side grasp). Both synthesized: chaining the scripted
  handoff costs 16–19 CPU-s per start, over the 10 s budget.
- Scripted (handoff): solver +1; drop_between −1; both_hold 0 (never +1); set_down_regrasp −1; giver carries it to
  the bin alone 0/−1 (never +1); idle 0. Cross-side carry/drop: the existing behaviours on cross-side starts.

**Drop amendment (user decision 2026-11-14):** only a low release is allowed. At the first executed action without
finger contact on the target, the target's lowest collision extent (bin frame, AABB of collision primitives) must be
≤ `release_max_clearance_m` = **5 mm** above the bin floor top, else terminal −1 "released too high". Threshold from
`../simulation/outputs/subtask_rewards/nuts_bolts/amend/release_sweep*.json`: releases at < 2 mm gave +1 on 14/20
(3 numerical blow-ups); 2–5 mm 1/4 numerical; 5–10 mm 8/15 numerical; > 10 mm most invalid or −1. 5 mm is the
smallest round value above the 2 mm release that works, with room for the part resting slightly tilted. Goal,
precondition, other failures, the numerical guard and the gate thresholds are unchanged. Carry is unchanged.

**Approval:** user approved the proposed defaults with “lets continue” after the concrete approval request.
Selected choices: reach/pick/carry/drop; closest unsorted active part; fixed nuts→teal, bolts→orange; dropping above assigned opening allowed; wrong bin, pushing, bin motion/tilt or loss of sorted contents fail; retain and measure cross-table coverage before handoff/exclusion decisions.
Any pending-approval wording below is historical and superseded by this approval.


Target per episode: one not-yet-sorted active part, chosen nearest either gripper at reset; save its identity and
that gripper side through the episode. Template A. SEED_BASE = 950_000_000 (reserved, starts not built).
Baseline revalidated: exp12 0/10, mean final 0.0258142, mean maximum 0.147727; ten worlds replay bit-exactly and
all 4,410 recorded reward checks are unchanged.

The stock scorer accepts either *consistent* partition: all nuts in teal and bolts in orange, or the inverse. The
proposed pilot freezes **nuts → teal, bolts → orange** for the episode because those are the named task bins.
Do not maximize between conventions independently for each target: that would reward mixed bins. This mapping is
a proposed user decision, not an implementation already made.

| stage | live prompt | precondition (= start) | goal (= next precondition) | progress (span, eps) | failures (−1) |
|---|---|---|---|---|---|
| reach | move the {side} gripper to the nearest {nut/bolt} | target at rest outside its assigned bin; no later precondition holds | grasp site ≤ 5 cm from target CoM, or a finger touches it | −TCP distance (0.30 m, 0.005 m) | target pushed > 3 cm without a grasp; part off table; either bin moved/tipped |
| pick | grasp the {nut/bolt} and lift it | target near gripper, or grasped with lift < 3 cm | two fingers of the same gripper touch target and lift ≥ 3 cm | 0.25·fingers + 0.5·clip(lift/0.03,0,1)·grasped (1.0, 0.05) | target pushed > 3 cm without a grasp; regression; part off table; bin moved/tipped |
| carry | carry the {nut/bolt} over the {teal/orange} bin | target grasped and lifted, outside the opening | target held above the assigned opening, with collision extent clear of rim | −3-D distance to clearance region (0.40 m, 0.005 m) | released outside assigned opening; wrong bin; regression; bin moved/tipped; previously sorted part knocked out |
| drop | release the {nut/bolt} into the {teal/orange} bin | held/touched above assigned opening, or safely lowered inside | assigned-bin membership **and** no finger contact **and** linear speed < 2 cm/s **and** support rooted in the assigned bin | −height above bin floor + 0.05 if released (0.15 m, 0.005 m) | released outside opening; **released too high (lowest collision extent > 5 mm above the bin floor at the first step without finger contact; amendment 2026-11-14)**; numerical instability; lands outside or in wrong bin; bin moved/tipped; previously sorted part knocked out |

The proposed final stage permits release above the correct opening, like the approved bottle design, but **the
bottle approval does not approve this task's semantics**. If the user wants careful placement instead, change the
last stage to place and reject release above the bin's measured low-release height before writing stage code.
A free-falling part cannot finish: keep its previous label until released-and-at-rest membership is satisfied.
Support must connect to the assigned fixture through released, stationary parts, as in the verified bottle
pipeline. Unsupported parts touching one another do not form a support root. Validate every release after a
regrasp; high-release permission still permits safe lowering inside the bin. Rest requires real MjData velocities
and contacts, not zeros manufactured by qpos-only scratch data. These are constraints on the proposed goal;
the manipulation choices below still require approval.

Thresholds and source:

- Bin frame and conservative goal half-sizes: read the current model's `nuts_bin_goal_region` / `bolts_bin_goal_region`
  sites, exactly as the scorer does, including scale/yaw randomization. Nominal half-size 6.8 × 8.0 × 4.2 cm, centre
  4.8 cm above bin origin. Do not hard-code those dimensions after randomization.
- Opening/clearance: measure collision walls/floor and the target's collision extent; enter opening at least 1 cm
  inside inner walls, leave only after a 1 cm hysteresis margin. Keep raw scorer membership distinct from over-rim
  carry completion.
- Near: enter 5 cm, exit 7 cm; provisional from the shared pick-and-place pilot, **not yet calibrated across a successful
  nuts/bolts grasp set**. The initial top-down helper failed, but a correctly transformed native-pad offset
  produced two real holds (one per type). This is limited calibration evidence, not a full start-set validation.
- Lift 3 cm; off-table 5 cm below reset rest height; target push 3 cm, bin moved 3 cm xy or tilted > 20° from reset:
  proposed policy choices carried from the pilot, requiring approval.
- At-rest speed 2 cm/s and no finger contact, held for 5 executed actions. Phase 0 found the raw qpos scorer accepts
  all parts floating within the box and even rotated/upside-down box layouts; the stage goal must add these guards.
- Unrelated parts: only off-table or loss of an already-sorted part is failure; harmless nudges are allowed.
- Shaping: `0.5 * Δprogress / span`, terminal +1/-1; potential differences sum independently of chunk subdivision.

Scripted behaviours required in phase 4:

| stage | behaviour | expected result |
|---|---|---|
| reach | solver / move away / idle / push target into bin | +1 / 0 / 0 / −1 |
| pick | grasp-lift / close only / open-gripper lift / one-finger drag / idle | +1 / 0 / 0 or −1 regression / never +1 / 0 |
| carry | solver / wrong bin / open before opening / shake / move bin toward target / idle | +1 / −1 / −1 / 0 / −1 / 0 |
| drop | lower to the floor and release slowly / release high over the opening / release beside opening / wrong bin / tip bin / idle | +1 after rest / −1 (too high) / −1 / −1 / −1 / 0 |

Starts and feasibility checks:

1. Reach from reset; skip reset-terminal cases and record all rejected seeds.
2. Pick: open gripper around resting target; no pushing or initial finger contact; source grasps preferably from a
   successful scripted approach, then recorded contacts, then sampled antipodal grasps of collision geometry.
3. Carry: end states of successful pick runs; lifted 5–12 cm, preserve original rest positions and target identity.
4. Drop: successful carry end states over the assigned opening. The naive top-down bin setup and a wrong-frame offset failed
   on 8 yaws/type; a correctly transformed native-pad offset yielded 2/2 real holds. Keep that coordinate conversion
   if sampled starts are needed; those two poses do not replace G3 validation or solver-chain coverage.
5. Each held-start set needs ≥ 80/100 valid candidates and ≥ 90 % retention after ten idle chunks; numerical-reset
   trials are invalid, never successful starts. Carry across table may need a handoff: do not silently exclude such
   targets. If a single-arm solver cannot reach, record the gate failure and ask about a handoff or scoped carry set.

**Decisions for approval**:

1. Bin convention: **fixed nuts→teal, bolts→orange** / retain either convention but freeze one in episode metadata
   (and provide a corresponding prompt). The latter needs a rule for empty-bin scenes.
2. Final stage: **allow dropping above the assigned opening, require release + rest** / lower carefully before release.
3. Wrong bin, pushing without grasp, bin motion/tilt, or knocking an already-sorted part out: **terminal −1** / choose
   different handling explicitly. Harmless motion of unsorted background parts remains allowed.
4. Target: **closest unsorted active part to either gripper** / prompt-selected type or balanced sampling by type.
5. Cross-table carry: **retain coverage and test, add handoff only after approval if needed** / approve a same-side
   carry subset up front. This decision changes task coverage.
