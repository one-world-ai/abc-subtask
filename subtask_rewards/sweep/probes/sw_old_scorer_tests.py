"""Phase 0 evidence: run tests/test_sweep_scorer.py against the pre-fix scorer (git <rev>:abc_sim/task_eval/sweep.py).

    git show 49fb9ad:abc_sim/task_eval/sweep.py > /tmp/old_sweep.py
    sw_old_scorer_tests.py /tmp/old_sweep.py      # expect the 5 false-positive tests to FAIL, 3 to PASS
"""
import importlib.util, runpy, sys
import numpy as np
import abc_sim, abc_sim.task_eval.registry as reg
print(abc_sim.__file__)
spec = importlib.util.spec_from_file_location("old_sweep", sys.argv[1]); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
class Old(old.SweepAwayEvaluator):  # old rule; expose the new metric names so the tests reach their assertions
    @property
    def _bin_radius_bottom(self): return self._bin_radius - 1e-3
    @property
    def _bin_radius_top(self): return self._bin_radius
    def evaluate_qpos_batch(self, q):
        r = super().evaluate_qpos_batch(q); a = self._bin_qpos_adr
        w, x, y, z = np.asarray(q[0, a + 3:a + 7], float) / np.linalg.norm(q[0, a + 3:a + 7])
        r.metrics["bin_up_z"] = np.array([2 * (y * z + x * w)]); r.metrics["bin_upright"] = np.array([True]); return r
reg._TASK_EVALUATORS["sweep_away"] = Old
ns = runpy.run_path("tests/test_sweep_scorer.py", run_name="not_main")
for name in sorted(k for k in ns if k.startswith("test_")):
    try: ns[name](); print("PASS", name)
    except AssertionError as e: print("FAIL", name, e)
