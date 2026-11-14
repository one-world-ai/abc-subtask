"""Phase 1: tool geometry at reset (seed 0, settled 1 s): brush and dustpan collision extents in world, dustpan lip
height above the table, pan floor, handle grip boxes; reachability hints (positions relative to the arm bases)."""
import numpy as np, mujoco, abc_sim
print(abc_sim.__file__)
env = abc_sim.make_env(task="sweep_away_paper_scraps_from_table", render_cameras=False)
env.reset(seed=0); m, d = env.model, env.data
for _ in range(500): mujoco.mj_step(m, d)
def world_pts(g):
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g]); s = int(m.mesh_vertadr[mid]); n = int(m.mesh_vertnum[mid]); v = np.asarray(m.mesh_vert[s:s + n], float)
    else:
        e = m.geom_size[g]; v = np.array([[sx * e[0], sy * e[1], sz * e[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
for body in ("brush", "dustpan", "bin_container", "trash_1"):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]]
    P = np.concatenate([world_pts(g) for g in gs])
    print(f"{body}: pos {np.round(d.xpos[b], 3)} mass {m.body_subtreemass[b]:.3f} world AABB {np.round(P.min(0), 3)} .. {np.round(P.max(0), 3)}")
    for g in gs:
        if m.geom(g).name: print(f"   named geom {m.geom(g).name}: centre {np.round(d.geom_xpos[g], 3)} size {np.round(m.geom_size[g], 3)}")
    if body == "dustpan":
        # height profile of the pan's collision hull along its long horizontal axis (lip -> handle)
        c = P[:, :2].mean(0); X = P[:, :2] - c; w, V = np.linalg.eigh(X.T @ X); ax = V[:, -1]
        t = X @ ax
        for lo in np.linspace(t.min(), t.max(), 9)[:-1]:
            sel = (t >= lo) & (t < lo + (t.max() - t.min()) / 8)
            print(f"   along {lo:+.3f}: min z {P[sel, 2].min():.4f} max z {P[sel, 2].max():.4f} width {np.ptp(X[sel] @ V[:, 0]):.3f}")
for arm in ("left_arm", "right_arm"):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, arm); print(arm, "base", np.round(d.xpos[b], 3))
