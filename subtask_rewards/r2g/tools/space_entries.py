"""Real2Gym entries for the Space `OneWorld-AI/subtask-reward-visualization` (group "r2g" = real -> gym).

  python space_entries.py VALIDATION_JSON SIMULATION_DIR
Builds each task's page (simulation/experiments/subtask_reward_space/build_task_page.py; videos from
slurm/torch/r2g_render.bash in SIMULATION_DIR/outputs/subtask_reward_space_src/r2g_<task>) and rewrites the r2g entries
of catalog.json and outputs/subtask_reward_space_src/tasks.json. Stage status comes from the gate table (gates.py).
Then: python experiments/subtask_reward_space/make_hf_space.py OUT && hf upload ... (see make_hf_space.py).
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ("reach", "pick", "carry", "place")
TITLES = {
    "put_the_screwdriver_in_the_bin": "Screwdriver in bin",
    "place_and_organize_the_fake_fruits_in_the_fruit_bowl": "Fake fruits in bowl",
    "organize_the_makeup": "Organize makeup",
    "organize_the_condiment_bottles": "Organize condiment bottles",
    "sort_the_hair_cutting_tools": "Sort hair-cutting tools",
    "sort_the_tools_into_containers": "Sort tools into containers",
    "sort_the_stationery_into_containers": "Sort stationery into containers",
    "place_and_organize_the_beverage_onto_the_shelf": "Beverages onto shelf",
    "place_and_organize_the_canned_foods_onto_the_counter": "Canned foods onto counter",
    "put_the_wine_bottle_on_the_wine_rack": "Wine bottle onto rack",
}
# Hand-written notes for stages that did not pass (shown in the overview); the gate numbers are appended.
NOTES = {
    ("put_the_wine_bottle_on_the_wine_rack", "place"): "the scripted straight-down place does not set the bottle on the rack",
    ("place_and_organize_the_beverage_onto_the_shelf", "place"): "the scripted straight-down place does not suit a shelf; many held starts leave the stage while idle",
    ("place_and_organize_the_canned_foods_onto_the_counter", "place"): "the scripted straight-down place does not suit the counter; many held starts leave the stage while idle",
    ("put_the_wine_bottle_on_the_wine_rack", "reach"): "the scripted reach knocks the standing bottles over",
    ("sort_the_tools_into_containers", "pick"): "the scripted pinch cannot lift the flat wrenches and pliers",
}
NO_STARTS = "start states cannot be built: the container is tall or entered from the side, so the gripper cannot hold the object over it"


def status(task, stage, row):
    if row is None:
        return "waiting", ""
    note = NOTES.get((task, stage), "")
    if "G4" not in row:  # no rollouts
        if row["G3"]:
            return "building", ""
        return "stopped", f"{NO_STARTS if stage in ('carry', 'place') else 'too few start states'} ({row['starts']} starts kept)"
    if row["G3"] and row["G4"] and row["G5"]:
        return "passed", ""
    kept, seeds = (int(x) for x in row["starts"].split("/"))
    if not row["G3"] and row["G4"] and row["G5"] and kept * 100 >= 80 * seeds:
        # only the 10 CPU s per start limit is exceeded (sampled poses for tall containers)
        return "accepted", (f"slowest start took {row['cpu'].split(' / ')[1]} CPU s to build (rule: 10); "
                            "exception accepted 2026-11-17")
    hacks = {k: v for k, v in row["summary"].items() if k != "solver"}
    few = [k for k, v in hacks.items() if v["planned"] < min(10, v["starts"] / 2)]
    if (row["G3"] and row["G4"] and few and row["gap"] >= 0.5 and not row["hack_plus"].keys() - {"drop_high"}
            and all(v["as_expected"] >= 0.8 * v["planned"] for v in hacks.values())):
        # the reward behaved in every run; only the rule "a hack must be plannable on 10 starts" is unmet
        return "accepted", "; ".join(f"{k.replace('_', ' ')} hack plannable on {hacks[k]['planned']} of {hacks[k]['starts']} starts "
                                     "(rule: 10), never +1" for k in few) + "; exception accepted 2026-11-17 (the other arm cannot reach the object)"
    failed = [g for g in ("G3", "G4", "G5") if not row[g]]
    nums = f"starts kept {row['starts']}, solver +1 {row['solver'].split(' / ')[0]}"
    return "stopped", (note + "; " if note else "") + f"fails {' '.join(failed)} ({nums})"


def main():
    val, sim = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]).resolve()
    src = sim / "outputs/subtask_reward_space_src"
    tool = sim / "experiments/subtask_reward_space"
    rollouts = Path(sys.argv[3]) if len(sys.argv) > 3 else sim.parent / "abc_rl_data/r2g_rollouts"
    tasks_path, cat_path = src / "tasks.json", tool / "catalog.json"
    tasks = [t for t in json.loads(tasks_path.read_text()) if not t["id"].startswith("r2g_")]
    cat = json.loads(cat_path.read_text())
    cat["tasks"] = [t for t in cat["tasks"] if t.get("group") != "r2g"]
    for task, title in TITLES.items():
        rows = {s: val.get(f"{task}__{s}") for s in STAGES}
        if not any(rows.values()):
            continue
        pid = f"r2g_{task}"
        vids = src / pid
        shown = [s for s in STAGES if rows[s] and "G4" in rows[s] and (rollouts / task / s).is_dir()]
        has_page = bool(shown) and (vids / "videos.json").exists()
        if has_page:
            subprocess.run([sys.executable, str(tool / "build_task_page.py"), "--title", f"{title} (Real2Gym)",
                            "--task", pid, "--env-sep", "__",
                            "--intro", "Real2Gym scene (a real-to-sim scan of a real task) on the ABC robot. One episode has "
                                       "one target object and the arm its instruction names; the place goal is the "
                                       f"`{pid}__obj_<class>` env's own success check.",
                            "--stage-table", str(HERE.parent / "stage_table.md"), "--scripted", str(rollouts / task),
                            "--videos", str(vids), "--stages", ",".join(shown), "--out", str(vids / "page")],
                           check=True, cwd=tool)
            ok = [s for s in shown if status(task, s, rows[s])[0] == "passed"]
            tasks.append({"id": pid, "title": f"{title} (Real2Gym)",
                          "description": f"{pid}: {' / '.join(shown)}. {len(ok)} of {len(shown)} shown stages pass.",
                          "tags": ["real2gym", "r2g", task, "subtask"], "page": f"outputs/subtask_reward_space_src/{pid}/page"})
        entry = {"task": pid, "title": title, "group": "r2g", "stages": []}
        if has_page:
            entry["page"] = pid
        for s in STAGES:
            st, note = status(task, s, rows[s])
            e = {"name": s, "status": st}
            if note:
                e["note"] = note
            if has_page and s not in shown:
                e["page"] = None  # nothing to open for this stage
            entry["stages"].append(e)
        cat["tasks"].append(entry)
    tasks_path.write_text(json.dumps(tasks, indent=1))
    cat_path.write_text(json.dumps(cat, indent=2))
    print("r2g entries:", [t["task"] for t in cat["tasks"] if t.get("group") == "r2g"])


if __name__ == "__main__":
    main()
