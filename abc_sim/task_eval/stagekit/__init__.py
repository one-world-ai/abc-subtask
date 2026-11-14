"""Task-independent pieces of the single-stage RL subtasks (extracted from the mug-flip subtasks).

A task's subtasks are made of:
  a scene       ``ObjectScene`` subclass: per-object measurements from MjData (contacts with each finger, lift above
                rest height, grasp-site distance, plus task-specific ones)
  stage tests   ``classify(obs, prev_labels, cfg)`` -> stage index per object, with hysteresis
  a tracker     ``StageTracker``: debounced stage labels over an episode
  an evaluator  ``SubtaskEvaluator`` subclass: one stage on one target object; latched failures; ``chunk_reward``
  starts        ``holding``: IK teleport, closing on a weightless object, hold checks; the task builds its own starts

Reference task: ``abc_sim.task_eval.mug_flip_{stages,starts,subtask}``. Pipeline and gates:
``.claude/skills/abc-subtask-reward``.
"""

from abc_sim.task_eval.stagekit.geometry import T, axis_angle_quat, inv, qmul, quat_of
from abc_sim.task_eval.stagekit.holding import (grasp_from_state, hold_action, hold_object, object_variant,
                                                place_gripper, set_object_pose, touches_other)
from abc_sim.task_eval.stagekit.robot import ARMS, GRIP_MAX, STATE_SPEC, ArmIds, actuator_limits, arm_ids, ik_link6
from abc_sim.task_eval.stagekit.scene import ObjectScene, free_joints_matching, geoms_of, is_desc, subtree
from abc_sim.task_eval.stagekit.subtask import SubtaskEvaluator
from abc_sim.task_eval.stagekit.tracker import StageTracker

__all__ = [
    "ARMS", "GRIP_MAX", "STATE_SPEC", "ArmIds", "actuator_limits", "ObjectScene", "StageTracker", "SubtaskEvaluator", "T", "arm_ids",
    "axis_angle_quat", "free_joints_matching", "geoms_of", "grasp_from_state", "hold_action", "hold_object",
    "ik_link6", "inv", "is_desc", "object_variant", "place_gripper", "qmul", "quat_of", "set_object_pose", "subtree",
    "touches_other",
]
