# put_relative: proposed stages (for approval)

Target per episode: the **mover** named in the prompt (one per episode; the reference and the other 2–4 objects are
obstacles). Template A. Baseline: none. Arm: whichever is nearer (both allowed).

| stage | prompt (live, filled per episode) | precondition (= start) | goal (= next precondition) | progress (span, eps) | failures (−1) |
|---|---|---|---|---|---|
| reach | "move a gripper to the {mover}" | none of below | grasp site ≤ 5 cm of the mover's CoM, or a finger touches it | −dist (0.30 m, 5 mm) | mover pushed > 3 cm while not grasped |
| pick | "grasp the {mover} and lift it" | near | both fingers of one gripper on it, lifted ≥ 3 cm | 0.25·fingers + 0.5·min(lift/3 cm, 1)·grasped (1.0, 0.05) | back to reach (let go / pushed away) |
| carry | "carry the {mover} {phrase} the {reference}" | grasped and lifted ≥ 3 cm | mover xy inside the stock zone, still held | −(xy gap to the zone, as the stock reward) (0.30 m, 5 mm) | dropped (no finger, > 4 cm above rest); back to reach |
| place | "put the {mover} down {phrase} the {reference}" | xy in zone, held or touched | stock success **and** no finger touching **and** at rest (< 2 cm/s) | −height above rest + 0.05·released (0.10 m, 5 mm) | dropped from > 4 cm; left the zone (back to carry) |

Every stage, also −1: mover off the table; **reference moved > 3 cm** (see decision 2).

Thresholds and where they come from:
- near 5 cm (exit 7 cm): top-down grasps put the grasp site at the CoM (dist ≈ 0–1 cm); mug used 10 cm because its
  grasps were on the side, 5–10 cm from the CoM. Re-measure once solver grasps exist.
- lift 3 cm, drop height 4 cm, rest 3 cm: same as mug and the stock "placed" tolerance (3 cm).
- zone: the stock cone (along 6–30 cm, across ≤ 8 cm), measured from the live reference.
- hold = 5 actions debounce; hysteresis on near and on in-zone (exit 1 cm outside the cone).

Scripted behaviours (phase 4):
| stage | behaviour | expect |
|---|---|---|
| reach | pre-grasp above the mover | +1 |
| reach | move away / idle | 0 |
| reach | push the mover into the zone along the table | −1 |
| pick | close top-down and lift 10 cm | +1 |
| pick | lift then open | −1 |
| pick | close only, no lift / idle | 0 |
| carry | carry to the zone at 10 cm height | +1 |
| carry | carry in the wrong direction / shake in place / idle | 0 |
| carry | open mid-carry | −1 |
| place | lower, open, back off | +1 |
| place | open 8 cm up | −1 |
| place | lower and keep holding / idle | 0 |
| place | carry out of the zone, then place | −1 |

Starts (100 per stage, seeds 910_000_000 + i; skip seeds that crash or already succeed at reset):
- reach: the reset scene.
- pick: open gripper (+1.2 cm) at a top-down grasp around the resting mover; must not touch or move it.
- carry: mover held top-down, lifted 5–15 cm above its spawn spot.
- place: mover held, 3–10 cm above a random point of the zone.
Grasps: synthesized top-down (yaw search, `hold_object` checks); chain from the pick solver later if needed.

**Decisions for you** (I picked the first option unless you say otherwise):
1. **Pushing the mover** into the zone without a grasp: −1 in reach/pick (the task says "put") / allowed (counts as
   reaching the goal).
2. **Moving the reference** to satisfy the relation: −1 if the reference moves > 3 cm / allowed.
3. **Other objects knocked**: only off-the-table ends the episode (−1) / any other object moved > 5 cm is −1.
4. **Prompts**: per-episode prompts with the object names (above) / fixed generic prompts ("pick up the object").
5. **Stock fixes** (benchmark): leave `PutRelativeEvaluator` and the randomizer unchanged (our place goal adds
   release + at rest; starts skip bad seeds) / also fix the stock scorer (require release + at rest) and the
   randomizer (reject prompts already satisfied) — changes benchmark episodes and numbers.
6. **Baseline**: skip it for the pilot (top-down grasps make starts without rollouts) / run 10 OpenWAM worlds first
   (B200 jobs).

---

## Amendment 2026-11-14: handoff for cross-table episodes (user-approved scope, 2026-11-14)

Chain: reach → pick → **handoff** → carry → place. **Cross-table** episode = the goal-zone point (14 cm along the
prompted direction from the reference's rest spot) has the other y sign than the mover's rest spot (the rule the
same-side carry starts already used: 171/290 usable seeds of 910 000 000 + 0..299). **Giver G** = the arm on the
mover's side (the arm the pick starts use; left for y > 0), **receiver V** = the other arm. Same-side episodes have
no handoff label; their classification, progress, failures and starts are unchanged. The handoff tests are the
task-independent `abc_sim/task_eval/handoff_stage.py` (unchanged), configured through `HandoffConfig` and the scene's
observation arrays (`com, lift, f_give, f_recv, touch_give, touch_recv, gap_give, env_touch`); `gap_give` is G's
pinch centre (link_6 (0, −0.031, 0.125), between the pads) to the mover's collision box.

| stage | prompt (live) | precondition (= start) | goal (= next precondition) | progress (span, eps) | failures (−1) |
|---|---|---|---|---|---|
| handoff | `put relative subtask: hand the object to the other arm` ("hand the {mover} over to the other arm") | cross-table; lifted ≥ 3 cm and G holds it with both fingers (= the pick goal), or V has a finger on it (`handoff_stage.handing`, region = the whole work space) | `handoff_stage.received_in_air`: V both fingers, G no contact and its pinch centre ≥ 6 cm from the box, lifted ≥ 3 cm, touching nothing but V (= cross-table carry precondition) | `handoff_stage.progress` (0 → 1.25) + 0.25·(1 − min(d_mid / 25 cm, 1)), d_mid = CoM distance to the middle box x 0.45–0.70, \|y\| ≤ 0.10, z 0.83–1.15 (1.5, 0.02) | dropped (no robot contact, > 4 cm up); touched anything but the two arms (table, other objects: in the air only, as inhand decision 2); **G carried it into the goal zone itself** (G touches it while its xy is in the zone); moved the reference; back to pick / reach (let go low / pushed) |
| carry (cross-table) | unchanged | received in the air by V (stays carry while V holds it alone and G is clear) | xy in the zone, a V finger on it, G not touching (= place precondition) | −xy gap to the zone, as same-side (0.30, 0.005) | **G touched the mover again**; dropped (no finger, > 4 cm up); back to handoff (G holds again / V lost it / G within 6 cm) ; moved the reference |
| place (cross-table) | unchanged | xy in the zone, V finger on it, G not touching | unchanged (stock success + released + at rest) | unchanged | unchanged + G touched the mover again |

Scripted behaviours (phase 4, `experiments/subtask_rewards/put_relative/scripted_handoff.py`):
| stage | behaviour | expect |
|---|---|---|
| handoff | solver: G lifts and carries it to a handoff pose in the middle (yaw / tilt search), V grasps the free part (side-on from its side), G opens (monitored) and backs off | +1 |
| handoff | drop_between: G at the handoff pose opens before V grasps | −1 |
| handoff | both_hold: V grasps, G never lets go | never +1, return ≤ 0.5 |
| handoff | release_stay: V grasps, G opens but stays at the mover | never +1 |
| handoff | set_down_regrasp: G sets it down on the table in the middle and lets go | −1 |
| handoff | giver_alone: G carries it into the goal zone itself | −1 (or unplannable) |
| handoff | idle | 0 |
| carry | solver: V carries it to the zone point | +1 |
| carry | release_early (V opens) | −1 |
| carry | regrasp_giver (G grabs it again) | −1 |
| carry | wrong_way (6 cm away from the zone); shake; idle | 0 |

Starts: handoff — G holds the mover top-down with the **pinch centre** on its vertical axis (at max(box centre,
top − 2.5 cm)), lifted 5–15 cm above its spawn spot, kept only if a scripted handoff plan exists; cross-table carry —
chained (handoff start + scripted handoff), else (source 3) V holds it alone top-down 13–17 cm up in the middle.
Handoff points inside the goal zone (+3 cm) are skipped (zones along y can straddle the middle).

Decisions made for the user (recorded in STATUS.md): G = mover-side arm, V = zone-side arm; "G carried it into the
zone" is −1 in handoff (inhand decision 1 analogue); the handoff region is the whole work space (the user's chain has
no separate "present" stage, so bringing it to the middle is part of handoff, shaped by the middle-box term);
handoff in the air only (table or other-object contact −1); "clear" = 6 cm from the collision box surface.
