# Adding subtask rewards for a new task

How to use the rewards for RL: [QUICKSTART.md](QUICKSTART.md). This page is about making new ones.

## What "done" means

A new task is split into stages (e.g. reach → pick → carry → place). Each stage gets:
1. a **stage table**: its start condition, goal, failures and progress measure, approved by a person;
2. **code**: a stage evaluator that scores every 32-action chunk +1 / −1 / progress shaping;
3. **a deterministic fresh-reset generator** that builds legal starts for ≥ 80 of 100 evaluation seeds;
4. **evidence the reward is right**: a scripted solver that earns +1, and scripted "hacks" (idle, push the object,
   use the wrong arm, …) that don't.

Everything is decided from simulator state (poses, contacts, joint angles), never from images or language labels.
The goal of stage k is the start condition of stage k+1, so trained stages chain back into the whole task.

## The pipeline and its gates

Each phase ends with a gate: a script result, not a judgement (thresholds in
`.claude/skills/abc-subtask-reward/references/gates.md`). A failing gate gets at most 3 rounds of real fixes, never a
looser threshold. After that the stage is marked **STOPPED** with the reason. Stopping is a valid outcome; the
task still ships its passing stages.

| phase | what happens | gate |
|---|---|---|
| 0 scene and scorer | check the task's own success check: 0 on resets, rejects fake successes, accepts real ones | **G0** |
| 1 stage table | write the stages and list the decisions it makes (e.g. "placed, not dropped") | **G1: a person approves** |
| 2 code + tests | stage evaluator, unit tests (starts classify, idle earns ~0, every failure fires, restore is exact) | **G2** |
| 3 start states | 100 seeds per stage, report rejection reasons and diversity; ≥ 80 valid, ≤ 10 s each, still in the stage after 10 idle chunks | **G3** |
| 4 scripted checks | solver earns +1 on ≥ 40 % of starts; hacks behave as expected, none gets +1 on > 10 %; solver beats the best hack by ≥ 0.5 return | **G4, G5** |
| 5 report | `design.md`, `STATUS.md`, seed audit, optional videos on the Space | |

Every task keeps a ledger, `subtask_rewards/<short>/STATUS.md`, with the result of each phase and its numbers. Work
resumes from it.

## Three cases

### Case A: a task of the ABC benchmark

Reference: mug flip (`abc_sim/task_eval/mug_flip_*.py`). The cleanest small example is bottles.

1. If the task's scene isn't in ABC yet, add it first: scene XML, randomizer (`abc_sim/randomization/tasks/`),
   success check (`abc_sim/task_eval/`), and one entry each in `abc_sim/task_specs.py`, `task_registry.py` and the
   randomizer / evaluator registries. The drawer, fruit-on-plate and whiteboard tasks came in this way, from
   `openwam-sim-eval/abc_tasks`.
2. Write three files next to the others, built on the shared kit `abc_sim/task_eval/stagekit/`:
   - `<short>_stages.py`: stage names, labels, prompts, the scene/tracker that measures the objects;
   - `<short>_starts.py`: `build_start` for each stage (place the object in the gripper, check that it holds);
   - `<short>_subtask.py`: the evaluator class. It **registers itself**: no shared file to edit.

   ```python
   class BottleSubtaskEvaluator(SubtaskEvaluator):
       TASK, ENV_TASK, NAME = "put_plastic_bottles_in_bin", "put_bottles", "bottles_subtask"
       STAGES, LABELS, FALLEN, STAGE_PROMPTS = STAGES, LABELS, FALLEN, STAGE_PROMPTS
       ...   # make_scene, make_tracker, make_stock, pick_target, build_start
   ```
   From the class, `abc_sim/task_eval/subtasks.py` generates one env per `STAGE_PROMPTS` key:
   `put_plastic_bottles_in_bin__reach`, `__pick`, `__carry`, `__drop`.
3. `tests/test_<short>_subtasks.py` (plain script, no pytest).
4. Start builder and scripted solver/hacks: copy a task's `subtask_rewards/<short>/tools/` (clutter, drawer) as the
   starting point.

Stage templates for common task shapes are in `.claude/skills/abc-subtask-reward/references/templates.md`:
A pick-and-place, B reorient in hand, C articulated (drawers), D handoff between arms, E tool use, F insertion /
hanging.

### Case B: a Real2Gym task that is already in the repo

36 of Yuyang's scenes are ported (`abc_sim/real2gym_catalog.py`, env ids `r2g_<task>`), 10 of them have stages. For
another **"put objects in / on a container"** task, the stage code already exists. It's one generic module for
all of them (`abc_sim/task_eval/r2g_subtask.py`, `r2g_stages.py`, `r2g_starts.py`).

1. Add the task to `R2G_STAGE_TASKS` in `abc_sim/task_eval/r2g_subtask.py`, with any object a scripted pinch can't
   hold in the exclude tuple:
   ```python
   "organize_the_condiment_bottles": ("bottle03_grinder",),
   ```
   That registers `r2g_<task>__reach`, `__pick`, `__carry`, `__place`.
2. Call the stage generator directly for each evaluation seed. Run the solver and hacks on the same seeds and
   report G2–G5. The historical Real2Gym scripts take a directory of saved rows. Adapt the harness to regenerate
   each episode when evaluating a new stage. Do not create intermediate start files for those scripts.
3. Record the result in `subtask_rewards/r2g/STATUS.md` and the usable stages in `QUICKSTART.md`.

Known limits of the shared design: carry/place mean "held over the container, above its top", which fits low open
containers (bin, bowl, tray). Tall or side-entry containers (shelf, counter, wine rack) mostly fail place; they need
their own stage table. Tasks of another kind (wipe, hang, open, pour) need their own stages too, as in case A.

### Case C: a new Yuyang scene that isn't ported yet

First the scene has to become a Real2Gym task:
- its object meshes go into the Real2Gym model library (HF dataset `OneWorld-AI/abc-rl-envs`, `abc_assets_v4/<task>/`);
- an entry goes into `abc_sim/real2gym_catalog.py` (`TASKS`: goals, budget, physics).

This port was done by Ansh, in `one-world-ai/openwam-sim-eval` (`abc_tasks/rl_suite`). The tool that converts a
scene into `abc_assets_v4` isn't in this repo; ask Ansh before porting a new one. After the port, follow case B or A.

## Using the agent skills

The repo ships two Claude Code skills in `.claude/skills/`. Opened in this repo, Claude Code loads them by itself
when you ask for subtask rewards:

> add subtask rewards for `<task>` &nbsp;·&nbsp; continue the subtask rewards for `<task>` &nbsp;·&nbsp; run the pipeline on the next queued task

| skill | what it tells the agent |
|---|---|
| `abc-subtask-reward` | runs phases 0–5 above for one task. It reads the mug-flip code map first, picks a stage template, writes the stage table and **stops for your approval (G1)**, then writes the code and tests, builds starts, runs the scripted checks, and applies the gates and the 3-repair-round stop rule. It writes everything into `subtask_rewards/<short>/STATUS.md` so a later session can resume |
| `abc-scorer-audit` | phase 0: checks a task's success function for false positives (reset probes, hand-made poses, random settles, re-scoring recorded rollouts) and fixes it with the narrowest physical rule |

What's in the skill folder:
- `SKILL.md`: the phases and where each file goes
- `references/gates.md`: the gate thresholds and the stop rule
- `references/templates.md`: stage templates A–F
- `references/contract.md`: the stagekit API a stage must implement
- `references/mug-codemap.md`: the reference implementation, file by file
- `references/gotchas.md`: known traps, e.g. grasp-site offset, restore order, CPU rendering
- `templates/`: blank `STATUS.md` and `stage_table.md`

The task queue and the status of every task are in [README.md](README.md).

**Known gaps** in the skills:
- They describe case A only. For Real2Gym, point the agent at case B above and at `subtask_rewards/r2g/`.
- Some steps still call helper scripts in the separate `simulation` repo: recorded-rollout replay, CPU renders, and
  the Space pages. Without that repo next to this one, skip those steps; the gates don't need them.

## Publishing a finished task

1. Tests pass: `python tests/test_<short>_subtasks.py`, plus `tests/test_subtask_api.py` (update its task list and
   stage count).
2. Publish the generator, runtime and asset versions, distribution bounds, training and held-out seed ranges,
   rejection counts and deterministic replay results. Require only the stage ID, seed and normal simulation
   dependencies for a fresh reset. Do not save or upload concrete starts.
3. Add the task to the usable-stage table in `QUICKSTART.md` and to the queue in `README.md`.
4. Optional: videos on the Space `OneWorld-AI/subtask-reward-visualization` (needs OneWorld-AI write access).
