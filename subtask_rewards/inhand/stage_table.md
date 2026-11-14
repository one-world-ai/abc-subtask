# inhand_transfer_item_to_other_side: proposed stages (for approval)

Target per episode: the single kitchen item (10 categories, 20-100 g). **Giver A = the arm on the item's spawn
side** (left arm for spawn y > 0), **receiver B = the other arm**; far side = the stock band (y·s ≤ −0.10, s = +1
for a left spawn). Template: D (handoff), with A-template reach/pick before and carry/place after.
Baseline: exp12 (OpenWAM canon 14B, step 57000) 1/10, re-scored bit-exactly with the fixed scorer: still 1/10
(world-05, a real in-air handoff at x 0.53, y 0.03, z 1.09, then carried and dropped from ~11 cm onto the far side).

The `handoff` stage is written to be **task-independent** (put_relative's cross-table carries need the same one):
it only needs a target object, a giver, a receiver and a middle region; nothing in it refers to "far side".

| stage | prompt | precondition (= start) | goal (= next precondition) | progress (span, eps) | failures (−1) |
|---|---|---|---|---|---|
| reach | inhand transfer subtask: reach the item with the near arm | none of below | A's grasp site ≤ r_near of the item's CoM, or an A finger touches it | −dist(A site, CoM) (0.30 m, 0.005) | item pushed > 3 cm while not held; item off table; B touches the item |
| pick | inhand transfer subtask: grasp and lift the item with the near arm | A near / touching, not lifted 3 cm | both A fingers on it, lifted ≥ 3 cm above rest | 0.25·fingers_A + 0.5·lift/3 cm·grasped_A (1.0, 0.02) | back to reach; pushed; off table; B touches |
| present | inhand transfer subtask: bring the item to the middle | grasped by A only, lifted | item CoM inside the middle region M, held by A only | −distance of CoM to M (0.30 m, 0.005) | dropped (no finger, > 4 cm up); **carried by A past M into the far half** (if handoff required); back to pick |
| handoff | inhand transfer subtask: hand the item to the other arm | held by A only, in M (start: B open at home) | both B fingers on it, no A finger, A grasp site ≥ r_clear from CoM, item ≥ h_air above the table and touching nothing but B | 0.25·fingers_B + held_B·(0.25·(2 − fingers_A) + 0.25·[fingers_A = 0]·min(d_A / r_clear, 1)) (1.25, 0.02) | dropped (no finger of either arm, > 4 cm up); item touches the table (set down) — see decision 2; item leaves M by > 10 cm while held by A only |
| carry | inhand transfer subtask: carry the item to the other side | held by B only, A clear, lifted | CoM over the far zone (y·s ≤ −0.13, on table xy) and ≤ 15 cm above its rest height, held by B only | −xy gap to the far zone (0.30 m, 0.005) | dropped outside the far zone; **A touches the item again** (regrasp); back to handoff (B lost it) |
| place | inhand transfer subtask: set the item down on the other side | held by B over the far zone | **fixed stock success** (far band, ≤ 10 cm up, on table, no robot contact, at rest < 2 cm/s) | −height above rest + 0.05 on release (0.10 m, 0.002) | released with its lowest point > h_drop above the table (decision 3); lands outside the band; A touches |

Stage tests use sim state only (contacts per finger, poses, speeds, A/B grasp-site distances); hysteresis on near
(exit r_near + 2 cm), on M (exit +2 cm) and on the far zone (exit +1 cm); debounce 5 actions; a free-moving item keeps
its previous label (as put_relative). Reward: kit "progress" mode, 32-action chunks, +1 goal / −1 failure / 0.5·Δprogress/span.

Thresholds and where they come from:
- middle region M = x 0.45–0.70, |y| ≤ 0.10, z 0.83–1.15 (8–40 cm above the table) — scripted handoffs passed with the
  item centre at x 0.49–0.60, y ≈ 0, z 0.83–0.95; the policy's one handoff was at x 0.53, y 0.03, z 1.09. |y| ≤ 0.10 is
  the stock band's inner edge, so "in M" can never already be far-side success.
- r_clear = 6 cm (A's grasp site from the item's CoM after release) — scripted handoffs end at 13–17 cm; to re-measure
  from the closest giver/receiver grasps once starts exist.
- h_air = 3 cm above the item's table rest height — same lift threshold as pick.
- r_near — measure from synthesized grasps (put_relative: 5 cm); long items (whisk/ladle 25–34 cm) need a grasp-point
  distance, not CoM distance → measure in phase 2.
- far zone = stock band + 3 cm margin (y·s ≤ −0.13) so carry's goal implies place can finish inside the band.
- h_drop (only if decision 3 = "place"): lowest point ≤ 5 cm above the table at release (put_relative style).
- rest = 2 cm/s (scorer fix; world-05 rests at 0.0003 m/s, bouncing items 0.1–0.4 m/s).

Scripted behaviours to run (phase 4):
| stage | behaviour | expect |
|---|---|---|
| reach | S: A to a top-down pre-grasp over the item | +1 |
| reach | move_away; idle | 0 |
| reach | push_across: A's closed fingers push the item towards the far side | −1 (pushed) |
| reach | other_arm_reach: B reaches for the item | −1 |
| pick | S: close + lift | +1 |
| pick | close_only; one_finger_drag | 0 |
| pick | lift_open | −1 / back to reach |
| present | S: carry to M | +1 |
| present | shake in place; idle | 0 |
| present | release_early | −1 (dropped) |
| present | carry_past_middle: A carries it to the far half alone (place_with_A_only) | −1 if handoff required, else not part of this chain |
| handoff | S: B grasps the free end / lower part, A opens, A backs off 10 cm | +1 |
| handoff | drop_between: A opens before B grasps | −1 |
| handoff | both_hold: B grasps, A never opens | ≤ 0.5 return, never +1 |
| handoff | release_stay: A opens but stays touching | no +1 |
| handoff | set_down_regrasp: A puts it on the table in M, B picks it up | −1 (decision 2) |
| handoff | idle | 0 |
| carry | S: B to the far zone | +1 |
| carry | release_early (drop before the far zone) | −1 |
| carry | regrasp_A: A grabs it again | −1 |
| carry | wrong_way (back towards A's side); idle | 0 |
| place | S: lower, open, back off | +1 |
| place | release_high (open 15 cm up) | −1 if "place", +1 allowed if "drop" (decision 3) |
| place | hold low (never release) | 0 (old scorer gave +1 here) |
| place | throw from the middle | not reachable from this start; covered by the stock scorer audit |

Start setup per stage (grasp source, how the precondition is made true, checks):
- All: seeds SEED_BASE + i, skipping seeds whose item spawns overlapping the table/arm (spawn penetration > 5 mm:
  19 % of seeds, all cheese graters, ⅓ of dish brushes, ½ of water bottles; they launch 1–9 m up) — or none if the
  randomizer is fixed (decision 5). Reject items the handoff cannot do (decision 4).
- reach: the reset scene after settling. pick: A open around the resting item (synthesized top-down, closing across
  the item's long axis). present: A holds it 5–15 cm above its spawn spot (kit `hold_object`).
- handoff: A holds it in M (item moved weightless to M, yaw of its long axis from {90, 60, 120, 30, 150, 0}°, A grasp
  near its A-side end or on its upper part; kit `hold_object`, B at home) — exactly the probe's first half.
- carry: end state of a scripted handoff (chain; probe `ih_handoff.py` ends with B holding alone, A 13–17 cm clear).
- place: B holds it 3–10 cm above a random far-zone point (synthesized, kit `hold_object`, put_relative's place).
- Checks (G3): held stages stay in stage for 10 idle chunks; creep < 1 mm per chunk.

Handoff feasibility (scripted: IK teleport + close + both-hold + A opens and backs off + B holds alone under gravity,
`probes/ih_handoff.py`, 60 seeds, numbers in STATUS.md): 47/60. Elongated items (whisk, ladle, pizza cutter,
rolling pin, dish brush) 30/33 with two top-down end grasps; cans 9/9 (A on top, B from the side); bottles 7/8 and
shaker 1/1 (side grasps); **cream-cheese stick 0/6** (3–6 cm flat block, ≤ 1.5 cm thick) and **cheese grater 0/3**
(A's grasp collides; it also launches at spawn).

**Decisions for you** (I picked the first option unless you say otherwise):
1. Must the item be handed over? — **yes: the chain requires A→B; A carrying it past the middle into the far half is
   −1 in `present`** / any transport counts (A may carry it across alone; then handoff is an optional stage).
   Evidence: A alone can hold it top-down ≥ 10 cm past the middle in 12/22 probe seeds (stock scorer: success).
2. Handoff in the air only? — **yes: touching the table during `handoff` is −1** / setting it down in the middle and
   re-picking with B also counts (then handoff = "pick from the middle" for B).
3. Place vs drop on the far side — **"set down": release with the lowest point ≤ 5 cm above the table** (spec says
   "set it down"; world-05 dropped from ~11 cm and would fail `place`, but still scores in the stock scorer) / any
   release that comes to rest in the band (+1 allowed for drops; like bottles).
4. Which items — **drop cream-cheese sticks and cheese graters from the handoff/carry/place start sets (no handoff found,
   graters also launch); keep them in reach/pick/present** / try harder for them (other grasp families, e.g. pinching a
   corner) before excluding / exclude them from every stage.
5. Spawn overlap (randomizer bug) — **skip those seeds in start sets; leave the benchmark randomizer unchanged** /
   fix the randomizer (lift spawns whose lowest point is below the table: changes ~19 % of benchmark episodes,
   including possibly exp12 worlds, so bit-exact replays of those seeds would break).
6. Which arm gives — **the spawn-side arm (A fixed by the spawn side); B touching the item in reach/pick is −1** /
   whichever arm grasps first becomes the giver (needs B allowed to reach across, more episodes but no fixed roles).
7. Stock scorer: also require handoff evidence (a latched "held by B alone" during the episode) — **no: enforce the
   handoff only in the subtask chain; the stock scorer judges the end state** / yes (stateful scorer; sparse checks
   every 50 steps would usually still see B carrying).
