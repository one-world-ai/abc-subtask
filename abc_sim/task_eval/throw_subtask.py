"""Throw one held bottle from outside the opening and let it settle in the bin."""
import numpy as np

from abc_sim.task_eval.bottles import BottlesInBinEvaluator
from abc_sim.task_eval.stagekit.held_stage import HeldStage
from abc_sim.task_eval.stagekit.scene import subtree


class ThrowHeld(HeldStage):
    TASK, ENV_TASK, NAME = "throw_plastic_bottles_in_bin", "bottles", "throw_held_subtask"
    STAGE_PROMPTS = {"throw": "Throw the held bottle from outside the bin opening so it lands and stays inside."}
    object_pattern = r"bottle_(\d+)_joint"

    def make_stock(self, model):
        self.bin_bodies = subtree(model, model.body("bin_container").id)
        return BottlesInBinEvaluator(model=model, spec=self.spec)

    def local_center(self, data, k):
        """Locate the bottle mass center in the moving bin frame."""
        body = self.scene.bodies[k]
        center = data.xpos[body] + data.xmat[body].reshape(3, 3) @ self.stock._bottle_center_offsets[k]
        bin_body = data.body("bin_container")
        return bin_body.xmat.reshape(3, 3).T @ (center - bin_body.xpos)

    def initial_arrangement(self, data, target):
        return np.linalg.norm(self.local_center(data, target)[:2]) > self.stock._bin_apothem

    def init_events(self, data):
        self.flight = False
        self.entered = False
        self.previous = self.local_center(data, self.target)
        self.bin_origin = data.body("bin_container").xpos.copy()

    def observe_goal(self, data):
        """Track unsupported flight outside the opening, rim crossing, and a settled landing."""
        k = self.target
        p = self.local_center(data, k)
        outside = np.linalg.norm(p[:2]) > self.stock._bin_apothem
        touched = self.scene.contact_obs(data)["touched"][k]
        contacts = [c for c in data.contact if any(tuple(self.scene.owner[g]) == (0, k) for g in c.geom)]
        bin_contact = any(any(self.model.geom_bodyid[g] in self.bin_bodies for g in c.geom) for c in contacts)
        if np.linalg.norm(data.body("bin_container").xpos - self.bin_origin) > 0.03 or data.body("bin_container").xmat[8] < 0.95:
            self._failed = "bin displaced"
        if self.flight and touched:
            self._failed = "bottle touched again"
        if not touched and not self.flight:
            if outside and not contacts:
                self.flight = True
            else:
                self._failed = "released without outside flight"
        if self.flight and not self.entered:
            rim = self.stock._bin_top_z
            if not outside and self.previous[2] >= rim and p[2] <= rim:
                self.entered = True
            elif contacts and outside:
                self._failed = "landed outside"
        self.previous = p
        vel = data.qvel[self.scene.dof_adr[k]:self.scene.dof_adr[k]+6]
        inside = self.stock.evaluate_qpos_batch(data.qpos[None]).metrics["bottle_in_bin_mask"][0, k]
        settled = np.linalg.norm(vel[:3]) < 0.05 and np.linalg.norm(vel[3:]) < 0.5
        return (self.entered and inside and settled and bin_contact and not touched,
                dict(outside_flight=self.flight, entered=self.entered, in_bin=bool(inside)))
