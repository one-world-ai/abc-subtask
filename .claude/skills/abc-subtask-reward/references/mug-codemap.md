# Mug flip: the reference implementation

Full design + reasoning: `../mug_substages.md` (policy-finetuning root, next to this repo). Read its "Idea", "Stages", "Reward" and
"Checking the reward" sections before starting a new task.

| file | what to copy |
|---|---|
| `abc_sim/task_eval/mug_flip_stages.py` | `StageConfig` (thresholds with measured reasons), `MugScene.observe` (contacts → fingers/grasped/touched, lift vs rest height, grasp-site distance), `classify` with hysteresis, `progress`, `MugStageTracker` (debounce, saved rest heights) |
| `abc_sim/task_eval/mug_flip_subtask.py` | evaluator: `begin`, per-step latched failures (dropped, regressed, off table), `chunk_reward` modes, potential-based shaping |
| `abc_sim/task_eval/mug_flip_starts.py` | `ik_link6` (damped LS on link_6 pose), `place_gripper`, `hold_mug` (weightless close 10 actions, gravity hold 30, checks), `build_start` per stage, grasp library I/O |
| `abc_sim/env.py` | `reset_to_snapshot` (forget_arm_state → reset(seed) → evaluate once → mj_setState), `reset_to_stage` |
| `abc_sim/task_specs.py`, `task_eval/registry.py` | `turn_mug_right_side_up__<stage>` specs with `evaluator_options` |
| `tests/test_mug_flip_subtasks.py`, `test_stage_starts.py` | the G2 test set |
| `../simulation/experiments/mug_stages/replay_stages.py` | bit-exact replay of exp12 server traces (state check at every model call) |
| same dir: `extract_grasps.py`, `build_starts.py` | grasp mining, 100 starts per stage (`SEED_BASE` 9e8) |
| same dir: `scripted_rollouts.py` | behaviours + EXPECT + summary.json (G4/G5) |
| same dir: `render_starts.py`, `render_rollouts.py`, `build_stage_pages.py`, `../simulation/experiments/subtask_reward_space/make_hf_space.py` | review renders and the Space |

Recorded rollouts for other tasks (exp12, OpenWAM canon 14B, one world per job, server-trace.jsonl + reward_trace):
`../simulation/outputs/openwam/experiment12/step-57000/tasks/<task>/world-XX/` (10 worlds) and the 60-world runs for
bottles/mug/sweep/plates/chess (paths in `../simulation/experiments/experiment12/experiment.json`).
Example replay-and-rescore script for any task: see `abc-scorer-audit` → `scripts/rescore_replay.py`.

Lessons from mug (carry over):
- Thresholds from measurement, not guesses (grasp site is 5–10 cm from a held mug's CoM → near = 10 cm).
- Recorded policies often do the task differently from the "clean" story (46/57 mugs set down tilted and settled;
  63 mid-air handoffs). Look at the recordings before fixing the stage table.
- Held starts can slip slowly — the 30-action hold check missed slips within ~10 chunks (G3 now checks this).
