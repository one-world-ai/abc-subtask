---
name: abc-scorer-audit
description: Audit an ABC sim task's success scorer for false positives and false negatives before it is used as a reward or stage goal — reset probes, hand-made poses, random physical settles classified geometrically, CPU renders, and bit-exact re-scoring of recorded policy rollouts. Use when asked to check, verify or fix an ABC evaluator (abc_sim/task_eval/*.py), when a baseline looks suspicious, or as phase 0 of abc-subtask-reward.
---

# ABC scorer audit

A scorer is trusted only after it (1) gives 0 on resets, (2) rejects the cheap ways to look successful, (3) accepts
real successes, and (4) its baseline numbers survive re-scoring of recorded rollouts. Worked example: the mug-rack
audit (2026-11-13) — band check → contacts rule → handle rule; exp12 8/10 became 0/10.

Python: `MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python <script>` on a CPU node.
Put probe scripts in the session scratchpad; promote the final cases into `tests/test_<short>_scorer.py`.

## Steps
1. **Read the evaluator** and write its success rule in one sentence. List what it does *not* check (contacts with
   the robot? object at rest? which frame is the goal region in? orientation?). Each gap is a probe.
2. **Reset probe**: 20–40 seeds → reward 0, success False. Any non-zero is a bug (e.g. mug standing near the tree).
3. **Hand-made poses** (set qpos, `mj_forward`, optionally settle): object held by the gripper at the goal; object
   resting next to / leaning on the outside of the goal; goal container tipped over; object floating in the region.
4. **Random physical settles** (the probe that finds what you didn't think of): drop the object from many random
   poses near the goal (`scripts/settle_search.py`), keep the ones that rest touching only the goal fixture, and
   classify each one *geometrically* (where is the supporting contact in the object's own frame?). Report counts per
   class and how many the scorer accepts.
5. **Render** a sample of each class on CPU (Mesa; see abc-subtask-reward `references/gotchas.md`), object painted
   bright. Look at them before trusting a geometric classifier.
6. **Fix** the scorer with the narrowest physical rule that separates the classes, measured per object model from
   its collision mesh (never hard-code one model's numbers). Keep the stock behaviour on true successes.
7. **Regression tests**: resets, each rejected class, at least two accepted real successes (also after settling),
   and a per-model check that the measured geometry exists (e.g. every mug model has a handle).
8. **Re-score recorded rollouts** bit-exactly with `scripts/rescore_replay.py` (exp12 worlds:
   `../simulation/outputs/openwam/experiment12/step-57000/tasks/<task>/world-XX`). Report old vs new success, and note
   episodes that were cut short by an old false positive (their new 0 is not a policy verdict).

## Output
In the task's ledger (or reply): the rule, the gaps found, probe counts table (class × accepted before/after),
test file, re-scored baseline. Mention which other checkouts still have the old scorer (`../simulation/abc`,
RoboEnv patch) — don't sync them unless asked.
