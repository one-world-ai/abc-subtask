# open_drawer / close_drawer: proposed stages (APPROVED by the user 2026-11-14, all default options)

Target per episode: the drawer named in the live prompt (`metadata.target_drawer`; top/middle/bottom). Template C.
Baseline: none (no rollouts, no demos). Arm: either (handles sit at y −0.23…0.00, so in practice mostly the right).
One chain per task; both reuse the stock `DrawerOpenCloseEvaluator` for the final goal.

Notation: **bar** = crossbar centre = `*_handle` site + 1.9 cm along the pull axis; **pinch** = grasp site +
R·(0.044, 0, −0.005) (finger-pad centre, overlay measurement); **q** = target slide joint (m, + = open);
**others** = max q of the two other drawers; **on handle** = finger geom of one gripper in contact with the target's
stem/bar geoms in any of the last 3 control steps (contacts chatter, see STATUS); **hand on drawer** = any robot geom
touching any target-drawer geom in the last 3 steps.

## open_drawer (4 stages)
| stage | prompt (live) | in it when (= start) | goal (= next start) | progress (span, eps) | failures (−1) |
|---|---|---|---|---|---|
| reach | "move a gripper to the {target} drawer handle" | none of below | pinch ≤ 3 cm from bar, or a finger on the handle | −dist(pinch, bar) (0.45 m, 5 mm) | others > 5 cm |
| grasp | "grasp the {target} drawer handle" | near, not grasped | both fingers of one gripper on the handle, gripper command < 0.5 (closing) | 0.25·fingers on handle + 0.5·closure·[both on] (1.0, 0.05) | others > 5 cm; back to reach (hand > 5 cm from bar) |
| pull | "pull the {target} drawer open" | grasped (or q ≥ 2 cm with hand on drawer, see D2), q < 12 cm | q ≥ 12 cm (stock success holds) | q / 12 cm, clipped (stock progress) (1.0, 0.01) | others > 5 cm; hand off the drawer for 5 steps with q < 12 cm → back to reach |
| release | "let go of the {target} drawer" | q ≥ 12 cm, hand on drawer | stock success **and** no robot contact with the target for 5 steps **and** pinch ≥ 5 cm from bar | min(dist(pinch, bar), 5 cm)/5 cm (1.0, 0.05) | q < 11 cm (pushed back); others > 5 cm |

## close_drawer (3 stages)
| stage | prompt (live) | in it when (= start) | goal (= next start) | progress (span, eps) | failures (−1) |
|---|---|---|---|---|---|
| reach | "move a gripper to the front of the {target} drawer" | none of below | pinch ≤ 3 cm from bar, or hand on drawer | −dist(pinch, bar) (0.45 m, 5 mm) | others > 5 cm; target pulled > 3 cm past its start |
| push | "push the {target} drawer closed" | hand on drawer (or near), q > 1.5 cm | q ≤ 1.5 cm (stock success holds) | (start − q)/(start − 1.5 cm) (stock progress) (1.0, 0.01) | others > 5 cm; target pulled > 3 cm past its start |
| release | "move the gripper away from the {target} drawer" | q ≤ 1.5 cm, hand on drawer | stock success **and** no robot contact with the target for 5 steps **and** pinch ≥ 5 cm from bar | min(dist(pinch, bar), 5 cm)/5 cm (1.0, 0.05) | q > 2.5 cm (pulled back open); others > 5 cm |

No grasp stage for close: pushing the front (fingertips, palm) is the natural strategy and the overlay's own solver;
a grasp-and-push also passes (D3).

Thresholds and where they come from:
- open 12 cm / closed 1.5 cm / others 5 cm — stock evaluator options (eval set v5); stage goals reuse them verbatim.
- release-stage hysteresis: open exit 11 cm, closed exit 2.5 cm — decision (1 cm band; the drawer is quasi-static,
  so these only fire if the hand drags it back).
- near 3 cm (exit 5 cm) — template C; overlay grasp puts the pinch 0.4 cm from the bar. Re-measure from solver grasps.
- span 0.45 m for reach — home grasp-site → handle 0.30–0.48 m (right arm), 0.45–0.64 m (left) (`geom.log`).
- contact window 3 steps — close push at 8 cm/s had contact at only 2 of 6 step boundaries (`dr_close_touch.py`).
- hold (debounce) 5 actions, as mug/bottles.
- "pulled > 3 cm past start" (close) — decision; start 14–19 cm, travel ends at 20 cm (+2 cm soft limit).

Scripted behaviours to run (phase 4):
| task / stage | behaviour | expect |
|---|---|---|
| open reach | overlay pre-grasp (pinch 10 cm in front of the bar, then to the bar) | +1 |
| open reach | move away / idle | 0 |
| open reach | go to the wrong drawer's handle | 0 |
| open reach | pull the wrong drawer open | −1 |
| open grasp | close on the bar (overlay 15/30/22° approach) | +1 |
| open grasp | close in the air 4 cm off the bar / idle | 0 |
| open grasp | hook pull without closing on the bar (D2) | +1 if hooks count, else 0 |
| open pull | pull 17 cm | +1 |
| open pull | pull 10 cm and hold | 0 (progress ≈ 0.83) |
| open pull | let go and retreat early | −1 (back to reach) |
| open pull | pull the neighbour too (wrong drawer) | −1 |
| open release | open the gripper, back off 10 cm | +1 |
| open release | keep holding / idle | 0 |
| open release | push the drawer back in while letting go | −1 |
| close reach | fingertips to the bar face | +1 |
| close reach | move away / idle | 0 |
| close reach | grab and pull the target further open | −1 |
| close push | push in by opening + 1.2 cm (overlay solver) | +1 |
| close push | push 30 % and hold | 0 (progress ≈ 0.3) |
| close push | push then pull back open | 0 / −1 (> 3 cm past start) |
| close push | open a neighbour while pushing | −1 |
| close release | back off 10 cm | +1 |
| close release | stay on the drawer / idle | 0 |
| close release | pull it open while leaving | −1 |

Start setup per stage (100 per stage, seeds SEED_BASE + i; skip seeds that crash):
- reach: the reset scene.
- open grasp: open gripper with the pinch at the bar (overlay approach, IK teleport; arm = the one whose IK error is
  smaller), no contact; check classify = grasp.
- open pull: chain from the grasp solver's end states; fallback: IK teleport + close on the fixed handle (10 actions),
  check both fingers on the handle and q < 1 cm.
- open release: set q ∈ U(12.5, 18) cm directly (1-DOF, quasi-static, no velocity) and grip the bar as above; check
  stock success and hand on drawer. Also chain from pull-solver ends.
- close reach: reset (target 14–19 cm open).
- close push: fingertips on the bar face (overlay pose at the reset opening), contact checked.
- close release: set q ∈ U(0, 1.4) cm and place the fingertips on the bar face; check contact. Also chain from the
  push solver.
- Held checks (G3 10 idle chunks): open pull/release starts must keep the handle grip and q within 1 cm; close
  release starts must stay ≤ 1.5 cm. Expect easy: no gravity along the slide, drawer quasi-static.
- Expected trouble: the **bottom drawer** (handle 4 cm above the table): overlay close solver 0/7, open 5/7, hook
  1/7. If its starts or solver fail G3/G4, either stop per target drawer or bias the solver to a steeper approach.

**Decisions for you** (I picked the first option unless you say otherwise):
1. **Release stage.** Add a final `release` stage (stock success + hand off + 5 cm away) — **yes** / no, end the
   chain at the stock success (the eval stops there with the hand on the handle).
2. **No-grasp opening (hook / fingertip behind the bar).** Counts: reaching q ≥ 12 cm by any contact gives the pull
   goal (+1), and the pull stage also starts from hook states — **yes, count it** (stock task doesn't require a grasp;
   hook works 12/20 scripted) / no, the pull goal requires both fingers on the handle at the crossing (hook → 0).
3. **Close by pushing.** Any contact counts (fingertips, palm, grasp-and-push); no grasp stage — **yes** / add a grasp
   stage for close too.
4. **Wrong drawer.** −1 as soon as another drawer exceeds 5 cm (stock `other_max_m`) — **5 cm (stock)** / stricter
   2 cm.
5. **Close: pulling the target further open.** −1 if q > start + 3 cm — **yes** / no failure, negative progress only.
6. **Target choice.** The prompted drawer only; prompts embed top/middle/bottom — **yes** / train one drawer
   (e.g. middle) first since bottom is hardest.
7. **Bottom drawer coverage.** Keep all three targets and report per-target gate numbers; if the bottom drawer fails
   G3/G4 after repairs, STOP it per target and ship top/middle — **yes** / exclude bottom from the start.
8. **Overlay scope.** All seven overlay tasks are registered in this branch (modules are shared) — **keep** / strip to
   the drawer tasks only (requires editing the overlay modules).
