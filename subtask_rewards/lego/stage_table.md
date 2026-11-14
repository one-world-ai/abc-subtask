# sort_lego_blocks — stages (APPROVED 2026-11-13)

**Approval:** user approved the proposed defaults with “lets continue” after the concrete approval request.
Selected choices: reach/pick/carry/drop; nearest unsorted target; matching-color bin; dropping above the opening allowed; wrong bin, ungrasped pushing, disturbed bins or loss of sorted contents fail; measure carry reachability before exclusions/handoff.
Any pending-approval wording below is historical and superseded by this approval.

**Amendments after approval (2026-11-13, G5 repairs; both tighten "ungrasped pushing fails", see STATUS.md):**
1. Reach goal = grasp site within 5 cm of the block centre. Bare finger contact no longer completes reach (it
   only keeps a pick episode in pick). The row below still shows the approved wording.
2. During reach, moving the block more than 3 cm fails even when both fingers touch it.
3. **Handoff for cross-side blocks (user-approved 2026-11-14; replaces the same-side drop scope).** See the
   amendment table below. Same-side blocks are unchanged.

Implemented stage prompts are the fixed strings in `lego_stages.STAGE_PROMPTS` (task lookup keys); the per-block
wording ("grasp red block 3 and lift it") is `LegoSubtaskEvaluator.stage_prompt()`.


### Amendment 3 — handoff stage (2026-11-14)

A block is **cross-side** when its rest spot and its matching bin's rest spot are on opposite sides of the table
(sign of y at reset; the rule the scoped drop used, `start_side != bin_side`). Receiver = the arm on the bin's side,
giver = the other arm (the picking arm in every measured scene: the nearest block is always the left arm's).
Chains: same-side `reach → pick → carry → drop` (unchanged); cross-side `reach → pick → handoff → carry → drop`.
The handoff tests are the shared `abc_sim/task_eval/handoff_stage.py` (unchanged) on LEGO per-block arrays, with the
LEGO region M = block CoM in x 0.40–0.60, |y| ≤ 0.10, z 0.80–1.10 (both arms reach it; x ≤ 0.60 keeps it off the
middle bin's opening, so being in M is never already the carry goal).

| stage | live prompt | precondition (= start) | goal (= next start) | progress (span) | failures (−1) |
|---|---|---|---|---|---|
| pick (changed goal label only) | as before | as before | both fingers of one arm, lift ≥ 3 cm (cross-side: that is the handoff label when the giver holds it) | as before | as before |
| **handoff** (cross-side only) | hand {color} block {index} to the other arm | giver holds it with both fingers, lifted ≥ 3 cm (pick's goal), or the shared `handing` test (in M with 10 cm exit hysteresis, lifted, giver both fingers or receiver ≥ 1) | shared `received_in_air`: receiver both fingers, giver no finger and no geom on it, giver pinch centre ≥ 6 cm from the block surface, lifted ≥ 3 cm, touching nothing but the receiver | shared `handoff_stage.progress` − min(distance of the CoM to M, 0.30) (1.55 = 1.25 + 0.30) | shared: dropped (no robot on it > 4 cm up); touched anything but the two arms (table, bin, other block: no set-down-and-regrasp). LEGO: giver carried it across (giver touches it with the CoM > 10 cm past M on the receiver's side); plus the existing bin/sorted/wrong-bin/other-block failures |
| **carry** (scoped) | as before | same-side: as before (grasped, lifted). **Cross-side: the handoff goal** (receiver holds it alone, giver clear) | as before (over the matching opening); cross-side not while the giver touches it | as before | as before; cross-side: falling out of "received" (giver back on it or within 6 cm) or below 3 cm lift = back to an earlier stage |
| drop (scope removed) | as before | as before; cross-side: held by the receiver over the opening | as before | as before | as before |

Pinch centre for "clear": this scene's measured closed-pad midpoint `LegoScene.pad_link6` (link_6 (0, −0.044,
0.130)); the YAM `<arm>_grasp_site` sits ~3 cm beside the pads (inhand finding), so it is not used.

Scripted checks (handoff): solver (+1); drop_between (bring to M and let go: −1); both_hold (receiver closes, giver
never lets go: 0 or −1, never +1); set_down_regrasp (giver sets it down in the middle, receiver picks it from the
table: −1); giver_across (giver alone carries it towards / over the receiver's bin and lets go: −1); idle (0).
Cross-side carry/drop reuse the carry/drop checks with the receiver.

Template A: one active, not-yet-sorted target block per episode, chosen as the block nearest either gripper at
the start. Keep its identity and matching color bin fixed throughout the episode. Prompt uses a stable color and
index, e.g. "the second red block"; all other blocks remain present. Proposed SEED_BASE = 930_000_000.

The stage table is a draft. The audit repaired two confirmed near-wall false negatives in the stock fixed
goal-site volume (`probes/edge_geometry.json`, `edge0.png`, `edge1.png`) and rejected sideways/inverted bins.
Final goals must use the corrected collision-cavity scorer. No stage code or registrations have been written.

| stage | live prompt | precondition (= start) | goal (= next start) | progress (span; eps) | failures (−1) |
|---|---|---|---|---|---|
| reach | move a gripper to the {index} {color} block | target unsorted, no later condition holds | grasp site within 5 cm of target center, or finger contact | −distance (0.30 m; 0.005 m) | push target >3 cm without grasp; target off table; bin disturbed; wrong bin |
| pick | grasp the {index} {color} block and lift it | near/touched, or grasped below lift threshold | both fingers of one arm contact target, lift ≥3 cm | 0.25·fingers + 0.5·clip(lift/0.03,0,1)·grasped (1.0; 0.05) | push; regress to reach; off table; bin disturbed; wrong bin |
| carry | carry the {index} {color} block over the {color} bin | held and lifted, outside matching opening | held above matching opening, center inside opening inset by target clearance | −3-D distance to opening approach box (0.40 m; 0.005 m) | release outside opening; wrong bin; knock a sorted block out; bin disturbed; off table |
| drop | drop the {index} {color} block into the {color} bin | held/touched above matching opening, or safely lowered inside | valid matching-bin membership, no finger contact, speed <2 cm/s, support rooted in bin fixture, held for 5 actions | −gap beyond opening + 0.05·released (0.10 m; 0.005 m) | release outside opening; settle outside or in wrong bin; knock a sorted block out; bin disturbed |

Proposed reward: `reward_mode="progress"`, chunk size 32 actions, terminal +1/−1; otherwise
`0.5 × Δprogress / span`. Going backward cancels previous shaping; idle earns approximately zero. A free-falling
block released over the correct opening retains the drop label until it settles, so airborne release is not
mistaken for regression. Holding a block inside the geometric bin volume never completes drop.
As verified in the bottle pipeline, support must connect to the fixture through released stationary objects;
airborne objects touching one another do not form a support root. Validate every release after a regrasp.
Allowing high release does not forbid safe lowering and release inside. Use real MjData contacts and velocities;
qpos-only scratch data cannot establish rest. These are implementation constraints of the proposed goal, not
approval of the still-pending manipulation choices below.

Thresholds and provenance:

- Near 5 cm (exit 7 cm), lift 3 cm, debounce 5 actions: initial pick-and-place defaults; verify against actual
  LEGO grasps before G2/G3. Near is not yet measured from policy grasps.
- Rest speed <2 cm/s: same released-and-resting final-goal convention as put_relative. The audit found 2 moving
  accepted poses among 180 drops, so geometric membership alone is insufficient.
- Opening clearance: measure collision-mesh top opening per randomized bin; inset by the target's projected
  collision half-extent + 5 mm. Do not reuse the conservative stock goal site's xy halfsize as physical geometry.
- Bin disturbed: displacement >5 cm or tilt >20° relative to reset (proposed task policy). The scorer repair only
  rejects horizontal/downward openings; the 20° failure is a stricter subtask choice, not a benchmark threshold.
- Push target >3 cm without grasp: proposed manipulation policy, not inferred from the coarse sorting scorer.
- Wrong-bin membership is terminal −1. Other objects can move on the table; dropping them off the table or
  removing an already sorted block fails. Do not count another block's progress toward this target's goal.

Start setup:

1. reach: reset; reject already-sorted target or invalid scene. Record randomized block/bin scales and identities.
2. pick: open top-down gripper around the chosen resting block. A table-side grasp must survive closing/lifting;
   use yaw alternatives and a collision check. Chain successful reach solver states when available.
3. carry: chain successful pick solver end states first; sampled collision-mesh grasps are fallback. Require
   stage classification and a 10-chunk idle check under gravity; remove slipping starts.
4. drop: chain successful carry solver end states. The audit's simple top-down, over-bin hold synthesis failed
   in all 6 attempted seeds (IK or bin contact), so do not assume teleporting a top-down hold works here. Reuse
   the bottle leaning/approach solver only after its geometry and grasp stability have been verified.

Scripted checks to run after approval:

| stage | behaviour | expect |
|---|---|---|
| reach | move to target / move away / idle / push target | +1 / 0 / 0 / −1 |
| pick | grasp-and-lift / close only / one-finger drag / lift with open fingers / idle | +1 / 0 / 0 or −1 / −1 / 0 |
| carry | carry to matching opening / carry to wrong bin / shake / release early / idle | +1 / −1 / 0 / −1 / 0 |
| drop | release over matching opening / release beside / hold inside / release over wrong bin / tip bin / idle | +1 / −1 / 0 / −1 / −1 / 0 |

**Decisions for the user** (proposed, not approved):

1. Final action: allow dropping from above the correct opening, followed by settling, as for the approved bottle
   task / alternatively require lowering before release. Bottle approval has not been extended to LEGO.
2. Target: nearest unsorted block, fixed for the episode / alternatively sample a prompted color/index uniformly.
3. Wrong bin or disturbing a previously sorted block: terminal −1 / alternatively no credit and continue.
4. Grasp required: pushing into the bin is a failure / alternatively allow sorting by pushing.
5. Reachability: keep cross-table episodes and require handoff or angled grasps if necessary / alternatively
   explicitly scope carry starts to reachable same-side episodes after measuring coverage.

Proposed order after bottles verification: finish scorer replay validation, approve this table,
then implement G2 before spending work on G3 starts or G4/G5 policies. Each failed gate gets at most 3 real repair
rounds; no stage is marked passed from this design draft.
