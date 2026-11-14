# sweep — stage table

**G1 (user, 2026-11-14): decision 1 = hand-pick chain H first** (section "Alternative chain H" below), exactly the
approved bottles semantics. The tool chain (brush + dustpan) is **deferred — not built**. Defaults kept for chain H:
6 scraps on the floor = −1; 7 stages fail at > 5 cm / > 20° bin disturbance, scorer keeps 45°. Implemented and
verified: see `STATUS.md`, `design.md`. The tool-chain draft below is kept for a later round.

Task `sweep_away_paper_scraps_from_table`, short `sweep`. Template E (tool use), adapted to what the scene
provides. Baseline: exp12 canon 14B **10/60 recorded → 7/60 with the fixed scorer** (3 were scraps on the table against
the bin base; see STATUS.md).

## What the scene provides (measured, `probes/sw_reset.py`, `sw_geom.py`, `sw_tools.py`)
- **Scraps**: 2–4 active paper balls per episode (40 resets: 2:10, 3:11, 4:19), Ø ≈ 3.5 cm (scale 0.85–1.15),
  2–4 g, friction 0.15 (they roll). Clustered at x ≈ 0.43–0.60, y ≈ −0.23…0.20, ≥ 4 cm apart.
- **Bin**: tapered can, 20 cm tall, rim z ≈ 0.94–0.95; inner radius 6.9 cm at the floor, 8.5–9 cm at the rim
  (scale 0.95–1.05); 150 g; x 0.58–0.70, y −0.17…0.20. **Scraps cannot be swept into it across the table** —
  they must be lifted over a 20 cm rim.
- **Brush**: flat hand brush, 21 cm, 31 g, lying on the table; grip box 8 × 1.4 × 2 cm at one end.
- **Dustpan**: 40 g, pan ≈ 19 cm wide, walls ≤ 5.7 cm, floor on the table; handle grip box 2.2 × 1.4 × 6.6 cm
  ≈ 10 cm behind the pan centre; opening on the side away from the handle (to be confirmed in phase 2).
- **Robot**: two YAM arms at (0.25, ±0.31). Bottles found straight-down grasps above z ≈ 1.05 out of reach; a
  leaning grasp reaches — dumping over a 0.95 rim needs the same.

So "sweep" means: brush the scraps into the dustpan, lift the dustpan over the bin, tip it.
**How the baseline actually succeeds** (`probes/sw_tooluse.py`, all 10 recorded exp12 successes replayed
bit-exactly): every scrap that ended in the bin was touched by a gripper and lifted above the rim (max z
0.98–1.12), i.e. **picked by hand and dropped in**. The policy often grasps the brush (10/10 worlds) and sometimes
the dustpan (6/10), but scrap–dustpan contacts are rare and no scrap was carried in the pan. See decision 1.

## Proposed chain (tool route, one episode = all scraps still outside the bin)

Notation: *held(tool)* = both fingers of one gripper touch that tool for ≥ 5 actions; *in pan* = scrap centre inside
the pan's inner hull in the dustpan frame, below its wall top, touching only dustpan/other scraps; *lip down* =
the pan's front edge touches the table and the pan floor is within 20° of level; *over opening* = every in-pan
scrap's xy within r_top − 2 cm of the bin axis and the pan's lowest point ≥ rim + 1 cm. Arm A = the arm on the
dustpan's side (y sign), arm B = the other.

| stage | prompt (unique) | precondition (= start) | goal (= next precondition) | progress (span, eps) | failures (−1) |
|---|---|---|---|---|---|
| grasp_dustpan | pick up the dustpan by its handle | reset (none of below) | held(dustpan) by arm A, lip down | −dist(arm A grasp site, handle grip centre) + 0.25·fingers (0.30 m, 5 mm) | dustpan flipped (> 60°) or off table; bin tipped/moved; scrap off table |
| grasp_brush | pick up the brush with the other hand | dustpan held by A, brush not held | held(brush) by arm B, lifted ≥ 2 cm; dustpan still held | −dist(arm B grasp site, brush grip) + 0.25·fingers + 0.5·lift/2 cm (0.30 m, 5 mm) | dustpan released; brush knocked off table; bin tipped/moved; scrap off table |
| place_dustpan | set the dustpan down next to the paper scraps | both tools held, nearest unswept scrap > 8 cm from the lip or outside the opening cone | lip down, nearest unswept scrap ≤ 8 cm in front of the lip, inside ±45° of the opening direction | −dist(lip centre, target scrap) − 0.1·opening-angle error (0.40 m, 5 mm) | either tool released; scrap off table; bin tipped/moved |
| sweep | sweep the paper scraps into the dustpan | dustpan placed (above), brush held | every scrap not in the bin is in pan and at rest; both tools still held | −Σᵢ min(dist(scrapᵢ, lip centre), 0.30)/n, per-scrap +1/n jump when it enters the pan (n = scraps outside bin) | either tool released; scrap off table/floor; scrap pushed into the bin region by hand (n/a: both hands busy); bin tipped/moved; in-pan scrap spilled back out after leaving the lip line by > 5 cm |
| carry | carry the dustpan over the bin | all unbinned scraps in pan, dustpan held | over opening, scraps still in pan | −3-D gap of pan centre to the over-opening box (0.40 m, 5 mm) | scrap spills out of the pan (lands on table/floor); dustpan released; bin tipped/moved |
| dump | empty the dustpan into the bin | over opening, scraps in pan | fixed stock scorer: all active scraps in the upright bin, at rest | −Σᵢ (height of scrapᵢ above bin floor, clipped 0.25 m)/n + pan tilt toward the opening (0.25 m, 5 mm) | scrap lands outside the bin; bin tipped/moved; dustpan dropped into/onto the bin; knocked a binned scrap out |

Brush release: allowed any time after the sweep goal (dropping it on the table is free; dropping it during
place/sweep is −1). The final dump goal is the fixed stock success, so the chain ends exactly at task success.

## Thresholds and where they come from
- bin "tipped" for the **scorer** = axis > 45° from vertical (committed fix; a tipped bin rests at 84°, an upright one
  at 0°, a policy-pushed one reached 20° — DreamZero exp7 world 00).
- bin "moved" for **stages** = > 5 cm xy or > 20° tilt (bottles default; stricter than the scorer).
- in-bin / bin wall: tapered radius from `gc_inner` per scale (committed fix).
- lip-to-scrap 8 cm, opening cone ±45°, 20° pan level, rim clearance 1 cm, over-opening r_top − 2 cm: **guesses to
  measure in phase 2** (scripted sweeps / dustpan pose sampling), like mug "near" 4 → 10 cm.
- held = both fingers ≥ 5 actions; scraps at rest = speed < 1 cm/s (bottles).
- 32-action chunks, progress reward, no frame history at mid-task starts (as approved for the other tasks).

## Scripted behaviours (phase 4)
| stage | behaviour | expect |
|---|---|---|
| grasp_dustpan | S: top-down grasp of the handle grip | +1 |
| | grasp the pan wall instead of the handle | 0 (not held(handle)) — or +1 if decision 5 allows any grasp |
| | push the dustpan with open jaws | 0, −1 if flipped/off table |
| | idle / move away | 0 / ≤ 0 |
| grasp_brush | S: top-down grasp of the brush grip, lift 3 cm | +1 |
| | release the dustpan while reaching | −1 |
| | push the brush off the table | −1 |
| | idle | 0 |
| place_dustpan | S: slide/set the lip in front of the nearest scrap, opening toward it | +1 |
| | lift the dustpan and keep it up | 0 |
| | drop the brush | −1 |
| | idle | 0 |
| sweep | S: brush strokes from behind each scrap toward the lip | +1 |
| | brush scraps sideways / away | < 0 shaping, no +1 |
| | sweep scraps off the table edge | −1 |
| | push the bin over with the brush | −1, never +1 |
| | drop the brush and push by hand (decision 1) | −1 under the tool route |
| | idle | 0 |
| carry | S: lift and move the pan over the opening, pan level | +1 |
| | tilt early (spill on the table) | −1 |
| | release the dustpan | −1 |
| | idle | 0 |
| dump | S: tilt the pan toward the opening, hold, set it back | +1 |
| | dump beside the bin | −1 |
| | push the bin over / drop the dustpan into it | −1 |
| | idle (held over the opening) | 0 |

## Start setup per stage
- grasp_dustpan: reset (sources: seeds `SEED_BASE + i`).
- grasp_brush: sampled grasp of the dustpan handle grip box (IK teleport, close, gravity hold check), dustpan
  resting lip-down; chain from the grasp_dustpan solver when it works.
- place_dustpan: both tools held (sampled grasps), dustpan anywhere ≥ 10 cm from the scraps; chain otherwise.
- sweep: dustpan teleported lip-down 3–8 cm in front of the nearest scrap, opening toward it, then grasped;
  brush held 2–5 cm above the table behind the scraps; settle; reject if any scrap moved > 1 cm.
- carry: 2–4 scraps dropped into the held dustpan's pan and settled (sampled), dustpan held lip-down or lifted ≤ 5 cm.
- dump: chained from the carry solver (bottles: synthesized holds above the rim reached 4/20); sampled leaning
  holds over the opening as fallback, scraps settled in pan, held-idle check (≥ 90 % still in pan after 10 chunks).
- Expect STOP candidates: place_dustpan/sweep (bimanual coordination, rolling 3 g balls), dump (reach above the
  rim with a 15 cm lever). Stop per stage; earlier stages still ship.

## Alternative chain H (hand pick, template A per scrap) — if decision 1 allows hands
reach → pick → carry → drop per target scrap (nearest unbinned scrap to a gripper), exactly the approved bottles
table: drop from above the opening allowed, must settle inside; bin moved/tipped, scrap off table, knocking a binned
scrap out are failures. Cheap (reuses the bottles code paths) and matches how the baseline policy succeeds.

**Decisions for you** (I picked the first option unless you say otherwise):
1. **Route** — tool chain above (brush + dustpan; hands-only pushing/picking is not a sweep) / hand-pick chain H /
   both chains as separate subtask sets (H first, it is cheap and matches the baseline's behaviour).
2. **Dustpan handling** — held by arm A during place/sweep (bimanual) / left resting on the table while one arm
   brushes (fewer stages; the 40 g pan may slide when brushed into).
3. **Progress unit** — whole-table: one episode sweeps all unbinned scraps, Σ-distance shaping + 1/n per scrap
   in pan / per-scrap: one target scrap per episode (nearest), goal = that scrap in pan, others only via failures.
4. **What counts as swept** — resting in the dustpan pan / anywhere off the table top into the dustpan *or* bin
   (e.g. brushed straight into a bin standing on the floor — not possible in this scene, so only matters later).
5. **Grip location** — only the handle grip boxes count as "held" / any two-finger grasp on the tool.
6. **Scraps on the floor** — failure (−1) in every stage, since the stock scorer can never count them again /
   allowed ("away from the table" read literally) — would need a scorer change, not recommended.
7. **Bin disturbance** — stages fail at > 5 cm or > 20° (stricter than the scorer's 45° upright rule) / only the
   scorer's 45°. Also: is 45° the right scorer threshold? (a bin can be pushed 20° without spilling.)
8. **Dump release** — tipping the pan from above the rim (scraps fall) is allowed, must settle inside / must lower
   the pan into the opening first. The dustpan may touch the rim; dropping the dustpan into the bin is −1.
9. **Brush after sweep** — may be dropped on the table freely after the sweep goal / must be put back down gently
   (no throw) / must stay held to the end.
10. **place_dustpan as its own stage** — yes (separate, easier credit) / merge into sweep.
