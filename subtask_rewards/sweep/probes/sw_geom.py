"""Phase 0: bin geometry in its own frame (local +y is up): inner visual mesh vs collision hull radius per height band."""
import numpy as np, mujoco, abc_sim
print(abc_sim.__file__)
env = abc_sim.make_env(task="sweep_away_paper_scraps_from_table", render_cameras=False)
env.reset(seed=2); m = env.model
b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bin_container")
def pts(g):
    mid = int(m.geom_dataid[g]); s = int(m.mesh_vertadr[mid]); n = int(m.mesh_vertnum[mid])
    v = np.asarray(m.mesh_vert[s:s+n], float); R = np.empty(9); mujoco.mju_quat2Mat(R, m.geom_quat[g])
    return v @ R.reshape(3, 3).T + m.geom_pos[g]
inner = pts(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "gc_inner"))
col = np.concatenate([pts(g) for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH])
print("inner y range", inner[:, 1].min().round(4), inner[:, 1].max().round(4), "col y range", col[:, 1].min().round(4), col[:, 1].max().round(4))
for lo in np.arange(-0.09, 0.13, 0.02):
    sel_i = inner[(inner[:, 1] >= lo) & (inner[:, 1] < lo + 0.02)]; sel_c = col[(col[:, 1] >= lo) & (col[:, 1] < lo + 0.02)]
    ri = np.linalg.norm(sel_i[:, [0, 2]], axis=1) if len(sel_i) else np.array([np.nan])
    rc = np.linalg.norm(sel_c[:, [0, 2]], axis=1) if len(sel_c) else np.array([np.nan])
    print(f"y [{lo:+.2f},{lo+0.02:+.2f}) inner r min/max {np.nanmin(ri):.4f}/{np.nanmax(ri):.4f}  col r min/max {np.nanmin(rc):.4f}/{np.nanmax(rc):.4f}")
# trash collision size
t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trash_3")
tp = np.concatenate([pts(g) for g in range(m.ngeom) if m.geom_bodyid[g] == t and m.geom_contype[g]])
print("trash_3 extents (body frame)", tp.min(0).round(4), tp.max(0).round(4))
print("bin mass", m.body_subtreemass[b], "bin body pos z", env.data.xpos[b][2])
