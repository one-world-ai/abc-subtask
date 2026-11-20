"""Replay ABC sim_224 episode poses through the original pouring or throw evaluator.

Run from the repo root: python -m scripts.replay_abc_goal /path/to/episode_...
Frame numbers in the output are zero-based.
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from abc_sim.task_eval.registry import make_task_evaluator


def episode_model(episode, meta):
    """Load the recorded scene with locally installed ABC meshes."""
    xml = ET.parse(episode / meta["scene_xml_file"])
    assets = Path(__file__).resolve().parents[1] / "abc_sim/models/assets"
    compiler = xml.find("compiler")
    for attr in ("meshdir", "texturedir"):
        compiler.set(attr, str(assets))
    return mujoco.MjModel.from_xml_string(ET.tostring(xml.getroot(), encoding="unicode"))


def replay(episode: Path) -> dict:
    """Measure instantaneous goal membership and reversals on one recorded scene."""
    meta = json.loads((episode / "episode_metadata.json").read_text())
    model = episode_model(episode, meta)
    task = meta["task_name"]
    count_key = {"sim_pouring_beads": "num_beads_in_container",
                 "sim_throw_plastic_bottles_in_bin": "num_bottles_in_bin"}[task]
    evaluator = make_task_evaluator(model, task)
    poses = np.load(episode / "scene_qpos.npy", allow_pickle=False)
    if poses.shape != (meta["num_steps"], model.nq):
        raise ValueError(f"Unexpected qpos shape {poses.shape} for {episode}")
    result = evaluator.evaluate_qpos_batch(poses)
    counts = result.metrics[count_key]
    successes = np.flatnonzero(result.success)
    return dict(episode=episode.name, task=task, frames=len(poses), start=int(counts[0]),
                peak=int(counts.max()), final=int(counts[-1]), required=int(evaluator.success_count),
                first_success=int(successes[0]) if successes.size else None,
                lost_success=int(np.sum(result.success[:-1] & ~result.success[1:])),
                final_success=bool(result.success[-1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path, nargs="+")
    for path in parser.parse_args().episodes:
        print(json.dumps(replay(path)))
