# Real2Gym "put it in a container" tasks: proposed stages (for approval)

First three tasks: `put_the_screwdriver_in_the_bin`, `place_and_organize_the_fake_fruits_in_the_fruit_bowl`,
`organize_the_makeup`. The same table is meant for every Real2Gym task whose goal kind is `in` (29 goals in the
catalog); per-task differences are data from `real2gym_catalog.py`, not code.

Env ids: `r2g_<task>__<stage>`. One episode = one **target object** and one **arm**, both taken from the
existing spec (`real2gym_spec.resolve_spec`, as in `r2g_<task>__obj_<class>`): the target is drawn by the seed among
the task's valid goal objects, the arm is the one on the target's side of the table. Template A (reach, pick, carry,
place). Final goal = the `__obj` env's own check, unchanged.

| stage | prompt (live, from the spec's phrases) | in it when (= start) | goal (= next start) | failures (−1) | progress (span) |
|---|---|---|---|---|---|
| reach | Move the {arm} arm to the {object phrase} {position}. | none of the below | pinch point of the named arm ≤ 5 cm from the target's centre of mass, or one of its finger pads on the target | target pushed > 8 cm while not pinched; target off the table; container moved or tipped | −distance pinch point → centre of mass (0.45 m) |
| pick | Grasp the {object phrase} with the {arm} arm and lift it {lift clause}. | near (≤ 5 cm, leaves at 7 cm) or pinched, but not yet lifted clear | both pads of the named arm on it, lifted ≥ 3 cm, touching nothing else (the `__pick` check, without its 0.5 s hold) | pushed > 8 cm while not pinched; off the table; container moved or tipped | 0.3 · pinched + 0.7 · lift / 3 cm (1.0), as the `__pick` env |
| carry | Carry the {object phrase} over the {container phrase}. | pinched and lifted clear, centre of mass not over the container | still pinched and clear, centre of mass inside the container's footprint (2 cm inside the edge) and above its floor | released or lost outside the container; container moved or tipped; a placed object knocked out | −horizontal gap from the centre of mass to the footprint (0.40 m) |
| place | {the `__obj` instruction's goal clause}, e.g. "Release it into the white rectangular bin." / "Place it upright in the blue organizer tray." | over the container, pinched or touching it | the `__obj` check: in the container's box, resting on it, not on the table, tilt in range where the task has one, no finger on it, speed ≤ 5 cm/s, for 0.5 s | lands outside the container; container moved or tipped; a placed object knocked out | −height of the lowest point above the container floor (0.20 m), + 0.05 once no finger touches it |

## Thresholds and where they come from
- lift 3 cm, hold 0.5 s, pinch time 0.2 s, rest speed 5 cm/s, pushed 8 cm, table height 0.70 m: the existing
  `Real2GymEvaluator` (`task_eval/real2gym_tasks.py`), unchanged.
- near = 5 cm from the **pinch point**, not the grasp site: at the scripted grasp pose the grasp site is 4.7–5.2 cm
  from the centre of mass (16 grasps, three tasks) because the pad centre is 4.4 cm ahead of the site.
- reach span 0.45 m: grasp site to target at reset, 0.445–0.457 m (screwdriver, 8 seeds).
- carry span 0.40 m: target to container at reset 0.16–0.38 m (27 resets, three tasks).
- container box (catalog): bin 23 × 16 × 3.6 cm, accepted up to 6 cm above the top; bowl 25 × 25 × 5 cm, 6 cm;
  organizer tray 17 × 30 × 1.7 cm, 10 cm, tilt ≤ 30° ("upright").
- container moved = > 5 cm or tipped > 20°: bottles' values. All three containers are free bodies and can be pushed.
- labels debounced over 5 actions; "near" 5 / 7 cm and footprint 2 / 0 cm hysteresis: stagekit defaults.

## Start setup per stage
- reach: the env's own reset (already checked settled and not decided by `start_check.py`).
- pick: named arm teleported open around the resting target (top-down pinch across its short axis), as put_relative.
- carry: held 5–15 cm above its spot (`stagekit.hold_object`), plus the 0.2 s pinch time the check needs.
- place: chained from the carry solver's end state (held over the container), as bottles.
- Every held start must still be in its stage after 10 idle chunks. Feasibility run: the screwdriver slipped in
  3 of 8 carry holds, and in 2 of 6 place holds it fell into the bin and scored with no action; such starts are dropped.
- Every start row stores the evaluator's pinch time: it lives outside the sim state and the final check needs it.

## Scripted behaviours (phase 4)
| stage | behaviour | expect |
|---|---|---|
| reach | move to the target / move away / idle / reach with the other arm | +1 / no +1 / 0 / no +1 |
| pick | grasp and lift / close without lifting / push / idle / grasp with the other arm | +1 / no +1 / −1 if > 8 cm / 0 / no +1 |
| carry | carry over the container / carry the wrong way / open early / idle / push the container under the object | +1 / no +1 / −1 / 0 / −1 |
| place | lower and release / release beside the container / hold still / idle / drop from high above | +1 / −1 / no +1 / 0 / +1 only where the prompt says "release" |

## Feasibility numbers (2026-11-16, `probes/proto_stages.py`, scripted solver of the feasibility prototype)
| task (object) | seeds | whole solver | starts built (reach / pick / carry / place) | exact restore |
|---|---|---|---|---|
| screwdriver → bin | 8 | 8 / 8 | 8 / 8 / 8 / 6 | all |
| orange peach → bowl | 8 | 6 / 8 | 8 / 8 / 8 / 6 | all |
| dark green jar → tray | 6 | 3 / 6 | 6 / 6 / 6 / 4 | all |

## Decisions for you (I picked the first option unless you say otherwise)
1. Env granularity — one env per task and stage, target drawn by the seed among all valid goal objects /
   one env per object class and stage (as `__obj_<class>`; about 6× more env ids).
2. Named arm — only the arm the instruction names earns progress and +1 (as the `__pick` / `__obj` checks);
   the other arm touching is recorded, not punished / −1 for a wrong-arm grasp.
3. Place — the stock check decides: dropping from above counts if the object ends in the container at rest (and
   upright where the task says so) / require lowering to within a few cm before release.
4. Other objects disturbed — recorded only, except the container moved or a placed object knocked out (−1) /
   −1 for any object moved > 5 cm.
5. Container on the far side of the named arm — keep only starts the named arm can reach (as put_relative) /
   add a handoff stage now.
6. Screwdriver slipping in the hold — drop slipping starts and keep the object's physics as imported /
   raise finger friction for this task (changes what the `__obj` env was evaluated with).
