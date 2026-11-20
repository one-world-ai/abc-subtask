"""Audit goal meshes, finger-only collision filters and possible unsupported carries on donor traces."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

import abc_sim
from abc_minimal.episode_io import load_episode
from abc_sim.task_eval.stagekit.scene import subtree
from abc_sim.task_eval.stagekit.starts import read_starts


def mesh_triangles(model, data, g):
    """Transform compiled mesh vertices and faces to the live world frame."""
    mesh = model.geom_dataid[g]
    start, n = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[start:start+n] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
    start, n = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
    return vertices, model.mesh_face[start:start+n]


def coverage(env, seed):
    """Estimate visible mesh area contained by collision hulls with 2 mm plane tolerance."""
    m, d, ev = env.model, env.data, env.subtask
    rng = np.random.default_rng(seed)
    items = {}
    for j in ev.goal_joints:
        gs = np.flatnonzero(np.isin(m.geom_bodyid, list(subtree(m, m.jnt_bodyid[j]))))
        coll = [g for g in gs if m.geom_contype[g] or m.geom_conaffinity[g]]
        vis = [g for g in gs if g not in coll and m.geom_group[g] <= 2 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        item = {"collision_geoms": len(coll), "separate_visual_meshes": len(vis)}
        # Check collision bit masks against other bodies, excluding bodies of the goal object itself.
        other = np.setdiff1d(np.arange(m.ngeom), gs)
        finger_only = []
        for g in coll:
            mask = (m.geom_contype[g] & m.geom_conaffinity[other]) | (m.geom_conaffinity[g] & m.geom_contype[other])
            partners = other[mask != 0]
            if partners.size and (ev.scene.owner[partners, 0] == 1).all():
                finger_only.append(int(g))
        item["finger_only_filter_geoms"] = finger_only
        if vis and all(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH for g in coll):
            triangles = []
            for g in vis:
                vertices, faces = mesh_triangles(m, d, g)
                triangles.extend(vertices[faces])
            a, b, c = np.asarray(triangles).transpose(1, 0, 2)
            area = np.linalg.norm(np.cross(b-a, c-a), axis=1)
            idx = rng.choice(len(a), size=2048, p=area/area.sum())
            u, v = rng.random((2, 2048, 1))
            u = np.sqrt(u)
            points = (1-u)*a[idx] + u*(1-v)*b[idx] + u*v*c[idx]
            covered = np.zeros(len(points), dtype=bool)
            for g in coll:
                vertices, _ = mesh_triangles(m, d, g)
                planes = ConvexHull(vertices).equations
                covered |= (points @ planes[:, :3].T + planes[:, 3] <= 0.002).all(axis=1)
            item["covered_fraction"] = float(covered.mean())
        items[m.jnt(j).name] = item
    return items


def sticky_segments(env, row, episode):
    """Flag runs of ≥4 donor frames moving with an open or non-pinching gripper."""
    ev, d, m = env.subtask, env.data, env.model
    k = row["target"]
    runs, length = [], 0
    previous = None
    poses = np.load(episode / "scene_qpos.npy")
    _, _, commands = load_episode(episode)
    for i, pose in enumerate(poses):
        d.qpos[:] = pose
        mujoco.mj_forward(m, d)
        obs = ev.scene.contact_obs(d)
        arm = int(obs["near_arm"][k])
        pos = d.xpos[ev.scene.bodies[k]].copy()
        grip = d.site_xpos[ev.scene.sites[arm]].copy()
        commanded_open = commands[i, arm*7+6] >= 0.95
        suspect = False
        if previous is not None and previous[0] == arm and obs["dist"][k] < 0.075:
            move = pos - previous[1]
            suspect = (pos[2] - row["rest_height"][k] >= 0.03 and
                       (commanded_open or not obs["grasped"][k]) and np.linalg.norm(move) >= 0.002
                       and np.linalg.norm(move - (grip - previous[2])) <= 0.003)
        if suspect:
            length += 1
        else:
            if length >= 4:
                runs.append([i-length, i-1])
            length = 0
        previous = arm, pos, grip
    if length >= 4:
        runs.append([len(poses)-length, len(poses)-1])
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--episodes", nargs="+", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {p.name: p for p in args.episodes}
    audits = []
    for file in sorted(args.root.glob("*/*.npz")):
        env = abc_sim.make_env(task=file.parent.name+"__"+file.stem, render_cameras=False)
        for row in read_starts(file):
            env.reset_to_stage(row["seed"], row)
            path = paths[str(row["episode"])]
            audits.append(dict(episode=path.name, seed=row["seed"], task=file.stem,
                               geometry=coverage(env, row["seed"]), sticky_candidates=sticky_segments(env, row, path)))
        env.close()
    args.output.write_text(json.dumps(audits, indent=2) + "\n")
