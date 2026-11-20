"""Pour from a held source mug and retain beads in the receiving cup."""
import numpy as np

from abc_sim.task_eval.pour import PourBeadsInContainerEvaluator
from abc_sim.task_eval.stagekit.held_stage import HeldStage


class PourHeld(HeldStage):
    TASK, ENV_TASK, NAME = "pouring", "pour", "pour_held_subtask"
    STAGE_PROMPTS = {"pour": "Tilt the held mug and pour the beads into the receiving cup. Keep the beads inside."}
    BUDGET_SECONDS = 18.0
    object_pattern = r"mug_(1)_jnt"

    def make_stock(self, model):
        self.source = PourBeadsInContainerEvaluator(model=model, spec=self.spec, target_joint_name="mug_1_jnt")
        return PourBeadsInContainerEvaluator(model=model, spec=self.spec)

    def initial_arrangement(self, data, target):
        source = self.source.evaluate_qpos_batch(data.qpos[None])
        receiver = self.stock.evaluate_qpos_batch(data.qpos[None])
        return (source.metrics["num_beads_in_container"][0] == len(self.source.bead_names)
                and receiver.metrics["num_beads_in_container"][0] == 0)

    def init_events(self, data):
        self.tilted = False
        self.last_grasp = data.time

    def observe_goal(self, data):
        """Require a held tilt followed by enough beads settling inside the live receiver."""
        held = bool(self.scene.contact_obs(data)["grasped"][0])
        upright = data.xmat[self.scene.bodies[0], 8]
        self.tilted |= held and upright < 0.75
        result = self.stock.evaluate_qpos_batch(data.qpos[None])
        inside = result.metrics["bead_in_container_mask"][0]
        settled = sum(np.linalg.norm(data.jnt(name + "_jnt").qvel[:3]) < 0.1
                      for name, ok in zip(self.stock.bead_names, inside) if ok)
        if held:
            self.last_grasp = data.time
        elif data.time - self.last_grasp >= 0.12:
            self._failed = "released source mug"
        return (held and self.tilted and settled >= self.stock.success_count,
                dict(beads_inside=int(inside.sum()), settled=int(settled), tilted=bool(self.tilted)))
