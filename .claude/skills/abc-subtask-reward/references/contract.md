# Shared kit (`abc_sim/task_eval/stagekit/`) and the task module contract

Extracted from mug flip on 2026-11-13; mug flip now runs on it (tests, saved starts and scripted rollouts are
bit-identical to before). Read the kit's docstrings; this is the map.

## Kit
| module | what |
|---|---|
| `scene.py` | `ObjectScene(model, [(name, free_joint_id), ...])` — `observe(data, rest)` → per-object `lift, hits (m×4 fingers), fingers, grasp_arm, grasped, touched, dist (nearest grasp site → CoM), near_arm`; `heights`, `finger_hits`, `object_T`; `qpos_adr`, `dof_adr`, `bodies`, `owner` (geom → object/finger). `free_joints_matching(model, r"kind_(\d+)_jnt")`, `subtree`, `geoms_of`, `is_desc` |
| `tracker.py` | `StageTracker(scene, classify, cfg, start_label=0, settle=3)` — debounced labels (`cfg.hold`), `reset(data, rest_height, labels)`, `update(data)` |
| `subtask.py` | `SubtaskEvaluator` — begin / per-step latched failures / `evaluate_model_data` / `chunk_reward` (progress, sign, best). Subclass sets `STAGES, LABELS, FALLEN, STAGE_PROMPTS, TARGET_KEY` and hooks `make_config, make_scene, make_tracker, make_stock, progress, step_failure, extra_metrics, pick_target, sync_stock_to_episode, build_start` |
| `holding.py` | `place_gripper`, `hold_object` (teleport 1 cm wider → close on weightless object → hold under gravity → checks), `grasp_from_state` (keys `T_obj_link6`, `obj_quat`; mug keys accepted), `hold_action`, `touches_other`, `set_object_pose`, `object_variant` |
| `robot.py` | `ARMS, GRIP_MAX, STATE_SPEC, arm_ids, ik_link6` |
| `geometry.py` | `T, inv, quat_of, axis_angle_quat, qmul` |

`env.reset_to_stage(seed, start=None, rng=None)` works for any `SubtaskEvaluator` task: builds via
`ev.build_start(env, seed, rng)` or restores `start` (target under `target` or `ev.TARGET_KEY`), then `ev.begin`.
Scripted-motion helpers (Cartesian paths, IK paths, run loop) are tooling, not kit:
`../simulation/experiments/mug_stages/scripted_rollouts.py` — copy and adapt per task.

## Task module — `abc_sim/task_eval/<short>_stages.py` (+ `<short>_starts.py`, `<short>_subtask.py`)
```python
STAGES = ("reach", "pick", "carry", "place", "done")   # trainable stages, then terminal label(s)
FALLEN = len(STAGES); LABELS = STAGES + ("fallen",)
STAGE_PROMPTS = {...}                                    # unique strings: task prompts are lookup keys
@dataclass(frozen=True)
class StageConfig: hold: int = 5; span: dict; eps: dict; ...   # each threshold: comment with where it came from
class <Task>Scene(ObjectScene): observe(...) adds task measurements (goal distance, orientation, ...)
def classify(obs, prev, cfg) -> labels                   # sim state only + hysteresis
def progress(stage, obs) -> np.ndarray
class <Task>SubtaskEvaluator(SubtaskEvaluator): hooks above; register in task_eval/registry.py
def build_start(env, seed, stage, library, rng) -> (start | None, log)    # start: seed, stage, target, arm, state, rest_height, labels
```
Specs: `<task>__<stage>` in `task_specs.py` with `evaluator_name=<task>_subtask`, `evaluator_options=(("stage", s),)`,
description "… start with env.reset_to_stage". Rules: goal of stage k ⇔ classify ≥ k+1; the final goal reuses the
stock evaluator's per-object success.
