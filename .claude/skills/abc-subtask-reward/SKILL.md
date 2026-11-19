---
name: abc-subtask-reward
description: Build single-stage RL subtasks for one ABC sim task — split it into physics-defined stages, write the stage tests + per-chunk reward, build start states for every stage, check the reward with scripted IK policies (solvers and reward hacks), and stop the task with a written reason when a gate fails. Use when the user asks to add subtask/substage/staged rewards for an ABC task, to continue a task in subtask_rewards/, or to run the pipeline on the next queued task. Mug flip is the finished reference.
---

# ABC subtask rewards: one task through the pipeline

Goal: for task `<task>`, deliver `<task>__<stage>` env tasks that start from a saved state where the stage's
precondition holds, and score each 32-action chunk with +1 (stage goal) / −1 (failure) / potential-based shaping in
between. The reference implementation is mug flip — read `references/mug-codemap.md` before writing any code.

Everything is decided from simulator state (poses, contacts, joint positions), never from vision or LLM subtask labels.
Goal of stage k = precondition of stage k+1, so trained subtasks chain back into the whole task.

## Where things live

| what | path |
|---|---|
| sim code (branch `subtask-r`) | `abc_sim/task_eval/` — shared kit `stagekit/`, per task `<short>_stages.py`, `<short>_starts.py`, `<short>_subtask.py` (like mug_flip_*) |
| other repos used (siblings of this one under policy-finetuning/) | `../simulation` (tooling + venv, branch `torch`), `../envs/mesa-cpu`, `../mug_substages.md` |
| task specs / registry | generated: set `TASK, ENV_TASK, NAME` on the evaluator class (`abc_sim/task_eval/subtasks.py`); never edit `task_specs.py` / `registry.py` for a subtask |
| user API | `env.reset_to_stage`, `env.subtask`, `env.advance_stage`, `abc_sim.load_starts` / `stagekit.save_starts` (`subtask_rewards/QUICKSTART.md`) |
| tests | `tests/test_<short>_subtasks.py` |
| tooling (replay, starts, scripted checks, renders) | `../simulation/experiments/subtask_rewards/<short>/` (mug: `../simulation/experiments/mug_stages/`) |
| outputs | `../simulation/outputs/subtask_rewards/<short>/` |
| **ledger + design doc** | `subtask_rewards/<short>/STATUS.md`, `subtask_rewards/<short>/design.md` (mug: `../mug_substages.md`) |
| task queue | `subtask_rewards/README.md` |

Python: `MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python ...` on a CPU node.
Read `references/gotchas.md` once per session.

## Real2Gym tasks (`r2g_<task>`)
Yuyang's scenes ported onto the ABC table (`abc_sim/real2gym_catalog.py`). "Put objects in / on a container" tasks
reuse one stage module: add the task to `R2G_STAGE_TASKS` in `abc_sim/task_eval/r2g_subtask.py`, then run
`subtask_rewards/r2g/tools/{build_starts,scripted_rollouts,gates}.py` (case B in `subtask_rewards/ADDING_A_TASK.md`)
and record it in `subtask_rewards/r2g/STATUS.md`. Other task kinds follow the phases below like an ABC task. A scene
that isn't in the catalog yet must be ported first (case C): stop and tell the user.

## Shared kit
`abc_sim/task_eval/stagekit/` (extracted 2026-11-13, see `references/contract.md`). Changing the kit: the mug tests,
`tests/test_stage_starts.py` and a 3-per-stage scripted run must stay bit-identical.

## The pipeline (one task)

Open or create `subtask_rewards/<short>/STATUS.md` from `templates/STATUS.md`. After every phase, write its result,
evidence paths and the decision there. **Resume from the ledger**: never redo a phase marked passed unless the code
it depends on changed.

Each phase has a gate (`references/gates.md`). A failed gate gets at most **3 repair rounds** (fix code, thresholds,
start setup or solver — never loosen the gate itself). Still failing → mark the stage or task **STOPPED** with the
reason and evidence, and move on. Stopping is a valid, useful outcome; say so plainly.

### Phase 0 — scene and scorer
Run the `abc-scorer-audit` skill on the task. Also check that the scene's assets are prepared
(`python prepare.py --sim-task <task>`) and record the task's baseline (exp12 tables in
`../simulation/experiments/experiment12/RESULT.md`, or "none").
Gate G0. A scorer false positive must be fixed before stages are defined — the stage goals reuse it.

### Phase 1 — stage table (the only human gate)
1. Read the task's evaluator, randomizer (`abc_sim/randomization/tasks/`) and, if any, recorded rollouts.
2. Pick the template(s) from `references/templates.md`. Per object, one chain of stages.
3. Fill `templates/stage_table.md`: stage, prompt (unique string), precondition (= start), goal, failures,
   progress number, span, eps, hysteresis, and the scripted behaviours you will test (solver + hacks).
4. List the **decisions** the table makes on the user's behalf (e.g. "placed, not dropped"; "wrong bin = failure or
   0?"; target choice in multi-object scenes) and ask the user to approve the table. Do not write stage code before
   the approval is in the ledger. If the user said to run unattended, record "auto-approved" and the decisions.

### Phase 2 — stages, tracker, reward
Write `<short>_stages.py` / `<short>_starts.py` / `<short>_subtask.py` against the kit (`references/contract.md`).
Register `<task>__<stage>` specs (description must say "start with env.reset_to_stage"). Unit tests (gate G2) on
scripted motions and synthetic states. Use `reward_mode="progress"` by default.

### Phase 3 — start states
Grasp sources, in order of preference:
1. **Chain from the previous stage's solver** (end state of a successful stage-k solver run is a stage-k+1 start).
2. **Recorded policy rollouts** — replay bit-exactly, cut at stage entries, extract grasps (mug: `replay_stages.py`,
   `extract_grasps.py`). Only if the baseline has some success on that stage.
3. **Sampled grasps** on the collision mesh (antipodal / top-down), IK teleport, close on the weightless object, hold
   check under gravity.

Build 100 starts per stage on seeds `SEED_BASE + i` (fixed per task, recorded in the ledger). Gate G3.

### Phase 4 — scripted policies
Write behaviours per stage (`references/templates.md` lists the hacks for each template), run them from the saved
starts exactly like an RL loop (32-action chunks, `chunk_reward()` after each), write `summary.json`. Gates G4 + G5.
If the solver fails, first look at *why* (IK unreachable, slip, collision) — a planning failure is a solver bug,
a slip with a good plan is evidence the stage is too hard.

### Phase 5 — report
Write `design.md` in the style of `../mug_substages.md` (stages, reward with worked examples, starts, checks, codemap,
open questions). Renders and a Space tab are optional: CPU rendering works (`references/gotchas.md`); publishing to
`OneWorld-AI/subtask-reward-visualization` needs the user's org token, so ask.

## Running several tasks
One task per session or per worktree agent; the queue and statuses are in `subtask_rewards/README.md`. Tasks don't
share code beyond `stagekit/`, so worktrees merge cleanly if each registers its own specs in a separate block.
Update the README status line when a task changes state.

## Done means
Every stage is PASSED or STOPPED with a reason; tests pass; ledger and design doc are written; README updated.
Report to the user: per stage — starts valid, solver success, best hack return vs solver return, and stops.
