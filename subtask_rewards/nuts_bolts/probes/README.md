# Nuts/bolts audit evidence

Run from the ABC checkout with the sibling simulation venv. Verify the printed `abc_sim.__file__` belongs to this
checkout. `MUJOCO_GL=disable PYTHONPATH=$PWD ../simulation/.venv/bin/python ...` is the CPU prefix.

- `audit.py`: 40 resets, seven artificial complete/mixed/unsupported/tipped layouts, and 180 drops (three seeds,
  two part types, 30 trials each). Fresh MjData, current arm-position controls, warnings and elapsed-time checks.
  Results are `audit.json`; `summary.json` contains the compact counts. `audit_round0.*` are superseded, untrusted
  methodology evidence; do not aggregate those drops into the final counts.
- `held_native_pad.py`: one successful real two-finger held nut and bolt inside the stock goal boxes. Correctly maps
  the grasp site's rotated local frame into link6 before positioning the gripper. `held.py` / `held_pad_offset.py`
  retain the preceding unsuccessful naive and wrong-frame probes; those are not evidence of impossibility.
- `replay_all.py`: full independent bit-exact replay/re-score of all ten existing exp12 worlds; successful completed
  outputs appear under `replay/world-XX.json`. The helper stops a world at any server-trace state mismatch and records
  its process exit. `replay/exits.json` is written only when all subprocesses finish.
- `render.py`: saved `.npz` poses, with nuts pink, bolts blue and enclosing walls hidden. Run using Mesa EGL variables
  from the subtask skill. Images of synthetic `floating_spec` / tipped layouts demonstrate a raw-score limitation,
  not physical successes. Physical sample `settle_0_bolt_1` is a floor-supported bolt, `settle_1_nut_0` a floor-supported
  nut, `settle_1_bolt_0` a rejected wall/rim rest. Invalid numerical samples (e.g. `settle_0_nut_0`) are retained but
  excluded from every at-rest accuracy count; `audit.json` has the per-sample `valid` flag.
- `tests.log`: four scorer regression checks from `tests/test_nuts_bolts_scorer.py`.

The drop classifier uses a separate support signal: contact with the assigned bin's floor, wall/rim-only contact, or
outside/table/other contact; it is not solely a second computation of the scorer's own membership test. In the 149
numerically valid, low-speed settles: 64 floor-supported accepted, 27 wall/rim-only rejected, 58 outside rejected.
Only one selected part per type per randomized seed is dropped; this is a focused geometry audit, not exhaustive
coverage of all part combinations, long-horizon dynamics or grasp retention.
