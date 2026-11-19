"""Real2Gym task catalog (OneWorld-AI/ABC_simulators releases rebuilt on abc_sim's table scene): per task the prompt,
the goals the evaluator scores (task_eval/real2gym_tasks.py), the randomization (randomization/tasks/real2gym_tasks.py)
and the eval step budget. No imports: task_specs.py and task_registry.py read it at import time.

Registered as env task / sim task ``r2g_<task>`` (the plain names collide with ABC's own dish-rack aliases).

Goals were read off each release's example episode (the objects' end state: what each touches, its tilt, joints) and
are checked against it by ``tools/real2gym_check.py replay`` (the episode's end state passes, its start does not).
Prompts are the task names. Step budgets: 1.5x the example episode (30 Hz frames), at least 450, a multiple of 15.

Goal kinds (each listed object is one item of progress; ``count`` k: any k of them):
  in       inside one of the containers' boxes, resting on the container or on another object inside it, off the
           table; ``tilt`` (min, max) degrees of the body z axis from world z
  off      not touching the containers, centre of mass outside their boxes, resting on the table or on objects
           that are off; ``tilt`` as above
  upright  body z tilt within ``tilt`` (default 0 .. max_deg), resting on something (``on_table``: on the table)
  joint    the joint between lo and hi (one item; ``objects`` = the body checked for release)
  erased   at least ``frac`` of the ink cells (geoms ink_*) erased by the task runtime (one item, fractional)
           (``supports``: more bodies an object may rest on, e.g. the liners in a tin)
Constraints (required for success, not progress): joints_max, in_box (at least count objects inside a box).
The 8 tasks uploaded later (place_the_bread .. pack_the_cookies) ship Real2Gym's own criterion.py: their goals
follow its end-state conditions (not its grasp / tool-use history, which is specific to Real2Gym's grippers).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PREFIX = "r2g_"
PROMPTS_FILE = Path(__file__).resolve().parent / "real2gym_prompts.json"


def _in(objects, container, *, above=0.06, margin=0.0, tilt=None, floor_slack=0.01, count=None, supports=()):
    conts = [container] if isinstance(container, str) else list(container)
    return {"kind": "in", "objects": list(objects), "containers": conts, "above": above, "margin": margin,
            "tilt": tilt, "floor_slack": floor_slack, "count": count, "supports": list(supports)}


def _off(objects, container, *, tilt=None, clear=0.0, count=None, touch_ok=None, either_face=False):
    """touch_ok: an object outside the container's box and resting on the table or a fixed prop counts even while it
    leans on the container's outside (bags, racks); default True except for sockets / ports, where touching = plugged."""
    conts = [container] if isinstance(container, str) else list(container)
    if touch_ok is None:
        touch_ok = not any("socket" in c or "port" in c for c in conts)
    return {"kind": "off", "objects": list(objects), "containers": conts, "tilt": tilt, "clear": clear, "count": count,
            "touch_ok": bool(touch_ok), "either_face": bool(either_face)}


def _upright(objects, max_deg=15.0, *, tilt=None, on_table=False):
    return {"kind": "upright", "objects": list(objects), "tilt": tilt or (0.0, max_deg), "on_table": on_table,
            "count": None}


def _joint(joint, lo, hi, holder):
    return {"kind": "joint", "joint": joint, "lo": lo, "hi": hi, "objects": [holder], "count": None}


def _erased(frac, eraser):
    return {"kind": "erased", "frac": frac, "objects": [eraser], "count": None}


def _joints_max(joints, max_q):
    return {"kind": "joints_max", "joints": list(joints), "max": max_q}


def _stay_in(objects, container, count, above=0.08):
    """Constraint: at least ``count`` of the objects' centres of mass inside the container's box."""
    return {"kind": "in_box", "objects": list(objects), "container": container, "count": count, "above": above}


MUGS = ("mug01_gray_pattern", "mug02_cyan", "mug03_yellow", "mug04_slate", "mug05_green", "mug06_brown",
        "mug07_pale_blue", "mug08_ivory", "mug09_gray")
UPRIGHT = (0.0, 20.0)
FLAT = (0.0, 30.0)

TASKS: dict[str, dict[str, Any]] = {
    "put_the_screwdriver_in_the_bin": {
        "goals": [_in(["screwdriver"], "bin")], "budget": 555,
        # the bin's 9 mm floor boxes lie on the table: a dropped shaft or tip (5 / 1.8 mm) that sinks past their
        # mid-plane is pushed out through the bottom, into the table (wedged, then launched)
        "floor_slab": {"bin": "bin_floor*"}},
    "take_the_wine_bottle_off_the_wine_rack": {
        "goals": [_off(["square_clear", "wine_a", "wine_b", "slim_clear", "round_clear", "blue_square"], "rack", tilt=UPRIGHT)],
        "budget": 3150},
    "place_and_organize_the_fake_fruits_in_the_fruit_bowl": {
        "goals": [_in(["yellow_pear", "orange_red_fruit"], "bowl")], "budget": 450},
    "stack_the_hats": {
        "goals": [_in(["beige_hat", "black_hat", "purple_hat", "red_hat", "white_hat"], "stand", above=0.30)], "budget": 3600},
    "turn_the_mug_right_side_up": {
        "goals": [_upright(MUGS)], "budget": 10860,
        "groups": [MUGS]},  # nine mugs packed 2-6 mm apart: the grid moves as one
    "organize_the_condiment_bottles": {
        "goals": [_in(["jar01_red_right", "jar02_red_left", "bottle03_grinder", "jar04_silver", "bottle05_black"],
                      "organizer_tray", above=0.15, tilt=UPRIGHT)], "budget": 795},
    "serve_the_lunch_box": {
        "goals": [_in(["drumstick01", "bacon01", "wing01", "bacon02", "prawn01", "drumstick02", "sausage01", "shrimp01"],
                      "lunch_box", above=0.015),
                  _in(["lid"], "lunch_box", above=0.04, margin=0.015, tilt=(0.0, 20.0))], "budget": 3765},
    "erase_the_whiteboard": {
        "goals": [_erased(0.9, "eraser")], "fixed": ["whiteboard"], "budget": 4260},
    "load_the_bowls_into_the_dish_rack": {
        "goals": [_in(["bowl_0", "bowl_1", "bowl_2", "bowl_3", "bowl_4"], "rack", above=0.05)], "budget": 11265},
    "load_the_plates_into_the_dish_rack": {
        "goals": [_in(["tan", "red", "blue", "white", "green", "beige"], "dish_rack", above=0.05, tilt=(55.0, 125.0))],
        "budget": 4185},
    "unload_the_bowls_from_the_dish_rack": {
        "goals": [_off(["floral", "orange_a", "orange_b", "teal", "lime", "black", "yellow", "beige"], "rack", tilt=FLAT)],
        "budget": 7185},
    "organize_the_makeup": {
        "goals": [_in(["sponge_puff", "green_puff", "compact_case", "eyeshadow_palette", "lipstick_case", "lip_gloss"],
                      "organizer_tray", above=0.10, tilt=FLAT)], "budget": 1995},
    "place_and_organize_the_beverage_onto_the_shelf": {
        "goals": [_in(["redbull_1", "redbull_2", "pocari_1", "pocari_2", "chang_1", "chang_2"], "shelf", above=0.20,
                      tilt=UPRIGHT)], "budget": 1500},
    "place_and_organize_the_canned_foods_onto_the_counter": {
        "goals": [_in(["progresso_1", "progresso_2", "congee_1", "congee_2", "short_can_1", "short_can_2"], "counter_board",
                      above=0.15, tilt=UPRIGHT)], "budget": 3060},
    "place_the_plates_into_the_plastic_bin_on_the_countertop": {
        "goals": [_in(["orange_1", "orange_2", "black_1", "black_2", "lime", "panda", "floral"], "plastic_bin", above=0.10,
                      tilt=FLAT)], "budget": 1305},
    "place_the_mixed_dishes_into_the_plastic_bin": {
        "goals": [_in(["object_plate_right", "object_plate_left", "object_tray", "object_cup_back", "object_cup_front"],
                      "bin_main", above=0.08)], "table_x_min": 0.255, "budget": 1215},
    "prepare_toiletry_sets_of_a_toothbrush_and_toothpaste_on_the_tray": {
        "goals": [_in(["yellow_mug", "purple_mug", "holder_block", "yellow_toothpaste", "pink_toothpaste"], "clear_tray",
                      above=0.25),
                  _in(["toothbrush_blue", "toothbrush_gray", "toothbrush_white"], "clear_tray", above=0.25, count=2)],
        "budget": 5580},
    "take_the_beverage_out_of_the_canvas_bag": {
        "goals": [_off(["fanta_1", "fanta_2", "coke_1", "fanta_3", "coke_2"], "canvas_bag", tilt=UPRIGHT)], "budget": 2070},
    "place_the_personal_care_products_into_the_canvas_bag": {
        "goals": [_in(["fairy_yellow_bottle", "fairy_red_bottle", "fairy_green_bottle", "mallory_pump_bottle_1",
                       "white_lotion_bottle", "mallory_pump_bottle_2"], "canvas_bag", above=0.10)], "budget": 1920},
    "unload_the_cups_from_the_dish_rack": {
        "goals": [_off(["clear_1", "green", "clear_2", "clear_3", "clear_4"], "dish_rack", tilt=UPRIGHT)], "budget": 6120},
    "unload_the_plates_from_the_dish_rack": {
        "goals": [_off(["floral", "orange", "lime", "teal"], "dish_rack", tilt=FLAT, either_face=True)], "budget": 2760},
    "sort_the_hair_cutting_tools": {
        "goals": [_in(["hair_clipper", "charging_cable", "oil_bottle", "battery", "guide_comb", "cleaning_brush"],
                      "clipper_box", above=0.02),
                  _joint("box_lid_hinge", -0.20, 0.20, "clipper_box")], "budget": 3930},
    "sort_the_tools_into_containers": {
        "goals": [_in(["pliers_yellow", "screwdriver_1", "screwdriver_2", "pliers_green", "wrench_large", "wrench_small"],
                      "drawer_cabinet", above=0.03)],
        "constraints": [_joints_max(("drawer_top_joint", "drawer_middle_joint", "drawer_bottom_joint"), 0.02)],
        "budget": 2760},
    "pull_the_plug_from_the_switch_port": {
        "goals": [_off(["plug_black", "plug_gray", "plug_yellow_1", "plug_yellow_2", "plug_red", "plug_blue", "plug_yellow_3"],
                       "switch_port", clear=0.005)], "budget": 2805},
    "pull_the_plug_off_the_socket": {
        "goals": [_off(["plug_white_1", "plug_white_2", "plug_cyan", "plug_blue_flat", "plug_white_flat", "plug_pink_flat"],
                       ["socket_blue", "socket_beige"], clear=0.005)], "budget": 3420,
        # the plug tenons fill the socket slots top to bottom: under abc's contact settings both slot walls clamp them
        # (~13 kN each), so the plugs cannot be pulled out; 90 % tenon height leaves clearance (85 %: plugs droop in the
        # slots; 92 %: one still jams; 95 %: clamped)
        "geom_scale": {"*_tenon_col": [1.0, 1.0, 0.90]}},
    "organize_the_hangers": {
        "goals": [_in(["pink_hanger_1", "pink_hanger_2", "pink_hanger_3", "pink_hanger_4", "blue_hanger_1", "blue_hanger_2",
                       "blue_hanger_3", "blue_hanger_4"], "hanger_rack", above=0.02)], "budget": 4170},
    "sort_the_stationery_into_containers": {
        "goals": [_in(["crayon_tube", "spray_bottle", "marker_yellow_1", "marker_yellow_2", "marker_yellow_3", "marker_yellow_4",
                       "pen_green", "pen_orange"], ["black_holder", "white_holder"], above=0.05)], "budget": 1530},
    "take_the_personal_care_products_out_of_the_canvas_bag": {
        "goals": [_off(["cream_pump_bottle", "fairy_yellow_bottle", "fairy_green_bottle", "sage_green_bottle"], "canvas_bag",
                       tilt=UPRIGHT)], "budget": 1590},
}
BREADS = ("bread_roll", "bread_slice", "bread_target", "croissant_static")
LINERS = tuple(f"liner_{i}" for i in range(5))
TASKS.update({
    # Real2Gym's criterion: its target bread (bread_target) on the serving plate, the other breads in the source tray,
    # the tongs back on the table. The prompt names no bread: any one bread on the plate, the other three in the tray.
    "place_the_bread": {
        "goals": [_in(BREADS, "serving_plate", above=0.06, count=1)],
        "constraints": [_stay_in(BREADS, "source_tray", count=3)], "budget": 675,
        # 70 cm deep layout on the 59 cm abc table: the place mat (under plate and cutlery) is left out, the plate
        # starts on the station base between the arms (level with the table)
        "drop": ["orange_mat"], "table_x_min": 0.245},
    # criterion: all six containers inverted (up axis . z <= -0.97), on the table, released, still
    "turn_the_large_container_upside_down": {
        "goals": [_upright(["pale_green", "blue", "turquoise", "white", "pink", "olive"], tilt=(166.0, 180.0), on_table=True)],
        "budget": 2535, "drop": ["green_cloth"], "table_x_min": 0.235},  # cloth would sit on the right arm base
    # criterion: each bottle's centre inside the rack's cradle box, lying along the rack, supported, released
    "put_the_wine_bottle_on_the_wine_rack": {
        "goals": [_in(["wine_a", "wine_b", "square_a", "slim", "square_b", "round_a", "round_b"], "rack", above=0.12,
                      tilt=(65.0, 115.0))], "budget": 3045},
    # criterion: four shoes inside the cabinet, upright (within ~16 deg), supported, released; door closed (|q| <= 0.12)
    "place_and_organize_the_shoes_in_the_shoe_cabinet": {
        "goals": [_in(["red01", "red02", "black01", "black02"], "cabinet", above=0.0, tilt=(0.0, 16.0)),
                  _joint("cabinet_door_hinge", -0.12, 0.12, "cabinet")], "budget": 6045},
    # criterion: all 8 pieces inside the tin (on the liners), liners in the tin, lid closed upside down on the tin
    "pack_the_chocolate": {
        "goals": [_in([f"chocolate_{i}" for i in range(8)], "tin", above=0.005, supports=LINERS),
                  _in(["lid"], "tin", above=0.05, margin=0.02, tilt=(165.0, 180.0))], "budget": 12720},
    # criterion: 5 cups in the rack's slots, inverted, released, still
    "load_the_cups_into_the_dish_rack": {
        "goals": [_in(["clear_1", "green", "clear_2", "clear_3", "clear_4"], "dish_rack", above=0.05, tilt=(150.0, 180.0))],
        "budget": 8310},
    # criterion: 7 glasses upright (15 deg) fully inside the tray, released; tray upright and still
    "place_the_glasses_into_the_tray": {
        "goals": [_in([f"glass_{i}" for i in range(1, 8)], "tray", above=0.08, tilt=(0.0, 15.0))], "budget": 4875},
    # criterion: cookies 0-4 in the tin's five liners, lid closed upside down. 15 look-alike cookies: any five.
    "pack_the_cookies": {
        "goals": [_in([f"cookie_{i}" for i in range(15)], "tin", above=0.005, supports=LINERS, count=5),
                  _in(["lid"], "tin", above=0.05, margin=0.02, tilt=(165.0, 180.0))], "budget": 10695},
})

# Longer settle at reset where objects were still moving 1 s after the default 0.5 s settle (seed check, 100 seeds):
# the whiteboard scene's headphones in their bin, the bread scene's croissant, the bowl rack's cable coils, the socket
# plugs with their reduced tenons.
for _t in ("erase_the_whiteboard", "place_the_bread", "unload_the_bowls_from_the_dish_rack", "pull_the_plug_off_the_socket"):
    TASKS[_t]["settle_steps"] = 1250

# Thin-walled crockery (glass walls 1.8 mm, bowl walls 3-4 mm, cups): stiff finger pads in these tasks, or the pads sink
# through the walls and the pad-tip spheres clamp them (randomization/tasks/real2gym_tasks.py, PAD_*).
for _t in ("load_the_bowls_into_the_dish_rack", "unload_the_bowls_from_the_dish_rack", "place_the_glasses_into_the_tray",
           "load_the_cups_into_the_dish_rack", "unload_the_cups_from_the_dish_rack"):
    TASKS[_t]["thin_shell"] = True

# floor_slab {container: geom name pattern}: the container's floor geoms (flat boxes / upright cylinders resting on
# the table) are made FLOOR_SLAB_M thicker downward, below the table top, and no longer touch the table (the container
# stands on its walls): anything that sinks into the floor is pushed back up, never through (physics(): floor_slab).
_DEFAULTS = {"xy_jitter": 0.03, "yaw_jitter": 0.20, "layout_jitter": 0.03, "settle_steps": 250, "fixed": [], "drop": [],
             "groups": [], "floor_slab": {}, "thin_shell": False}


def task_config(task: str) -> dict[str, Any]:
    cfg = {**_DEFAULTS, **TASKS[task]}
    cfg.setdefault("prompt", task.replace("_", " "))
    cfg.setdefault("constraints", [])
    return cfg


# ---------------------------------------------------------------------------------------------- physics
# <option> of each release's gym_scene.xml: (timestep s, friction cone, impratio, noslip iterations). The import
# dropped them (abc_sim: 0.002 s, pyramidal cone, impratio 1); see physics() for what the envs use.
RELEASE_PHYSICS: dict[str, tuple[float, str, float, int]] = {
    "erase_the_whiteboard": (0.002, "elliptic", 1, 0),
    "load_the_bowls_into_the_dish_rack": (0.002, "elliptic", 10, 0),
    "load_the_cups_into_the_dish_rack": (0.002, "elliptic", 1, 0),
    "load_the_plates_into_the_dish_rack": (0.002, "elliptic", 10, 0),
    "organize_the_condiment_bottles": (0.002, "elliptic", 30, 0),
    "organize_the_hangers": (0.002, "elliptic", 30, 0),
    "organize_the_makeup": (0.002, "elliptic", 30, 0),
    "pack_the_chocolate": (0.002, "elliptic", 1, 10),
    "pack_the_cookies": (0.001, "elliptic", 10, 10),
    "place_and_organize_the_beverage_onto_the_shelf": (0.002, "elliptic", 30, 0),
    "place_and_organize_the_canned_foods_onto_the_counter": (0.002, "elliptic", 30, 0),
    "place_and_organize_the_fake_fruits_in_the_fruit_bowl": (0.001, "elliptic", 30, 0),
    "place_and_organize_the_shoes_in_the_shoe_cabinet": (0.002, "elliptic", 30, 5),
    "place_the_bread": (0.001, "elliptic", 30, 0),
    "place_the_glasses_into_the_tray": (0.001, "elliptic", 1, 0),
    "place_the_mixed_dishes_into_the_plastic_bin": (0.0002, "pyramidal", 1, 0),
    "place_the_personal_care_products_into_the_canvas_bag": (0.002, "elliptic", 30, 0),
    "place_the_plates_into_the_plastic_bin_on_the_countertop": (0.002, "elliptic", 10, 0),
    "prepare_toiletry_sets_of_a_toothbrush_and_toothpaste_on_the_tray": (0.002, "elliptic", 50, 5),
    "pull_the_plug_from_the_switch_port": (0.002, "elliptic", 30, 0),
    "pull_the_plug_off_the_socket": (0.002, "pyramidal", 1, 0),
    "put_the_screwdriver_in_the_bin": (0.001, "elliptic", 1, 0),
    "put_the_wine_bottle_on_the_wine_rack": (0.001, "elliptic", 10, 0),
    "serve_the_lunch_box": (0.001, "elliptic", 30, 0),
    "sort_the_hair_cutting_tools": (0.002, "elliptic", 30, 0),
    "sort_the_stationery_into_containers": (0.002, "elliptic", 30, 0),
    "sort_the_tools_into_containers": (0.002, "elliptic", 30, 0),
    "stack_the_hats": (0.001, "elliptic", 30, 0),
    "take_the_beverage_out_of_the_canvas_bag": (0.002, "elliptic", 30, 0),
    "take_the_personal_care_products_out_of_the_canvas_bag": (0.002, "elliptic", 30, 0),
    "take_the_wine_bottle_off_the_wine_rack": (0.001, "elliptic", 30, 0),
    "turn_the_large_container_upside_down": (0.001, "elliptic", 1, 0),
    "turn_the_mug_right_side_up": (0.002, "elliptic", 30, 0),
    "unload_the_bowls_from_the_dish_rack": (0.001, "elliptic", 1, 0),
    "unload_the_cups_from_the_dish_rack": (0.002, "elliptic", 1, 0),
    "unload_the_plates_from_the_dish_rack": (0.002, "elliptic", 10, 0),
}
CONTROL_DT = 0.034  # abc_sim's control step (0.002 s x 17): kept whatever the physics step
# Defaults; env overrides R2G_DT, R2G_CONE, R2G_IMPRATIO, R2G_NOSLIP, R2G_SOLREF_MAX, R2G_FLOOR_SLAB, R2G_SIM_GUARD
# ("release" = the release's own value). Measured on recorded rollouts re-simulated bit for bit, drop tests and the
# 36 release demos (tools/real2gym_blowup_scan.py, real2gym_blowup_resim.py, real2gym_drop_test.py, real2gym_physics_replay.py):
#   solref_max 0.006  imported geoms left at MuJoCo's default contact softness (0.02 s) get 0.006 s: at 0.02 s a tool
#                     dropped from 10 cm sinks 10 mm, through a 9 mm container floor; at 0.006 s 2 mm.
#   floor_slab 0.05   depth (m) of the downward floor slabs of the containers listed per task (TASKS: floor_slab).
#   sim_guard 1       abc_sim/sim_guard.py: removes bogus box-box / cylinder contacts, detects launches (the latch).
#   cone pyramidal    abc_sim's. A wedged part (opposing contacts) is pushed apart with about 3 / mu^2 x the force of
#                     the elliptic cone: 13 kN vs 29 N on a 93 g screwdriver through the mu 0.08 floor of its bin; the
#                     releases (34 of 36 elliptic) use mu down to 0.01. But abc_sim's gripper was tuned on the
#                     pyramidal cone: with the elliptic one 2 - 3 of the 4 demos that replay today fail (screwdriver,
#                     condiments), so it stays, and the slab, the stiffness and the latch cover the wedges.
#   dt 0.002, impratio 1, noslip 0: abc_sim's. The releases' own (11 tasks at 1 ms) replay no more demos and do
#                     not stop a blow-up (a wedge is as violent at 0.5 ms).
PHYSICS_DEFAULTS = {"dt": "0.002", "cone": "pyramidal", "impratio": "1", "noslip": "0", "solref_max": "0.006", "sim_guard": "1",
                    "floor_slab": "0.05"}
# Per-task exceptions to PHYSICS_DEFAULTS (an R2G_* environment override still wins over both).
#   toiletry: no solref cap. abc_assets_v3 rebuilt the toiletry collision (CoACD hulls at MuJoCo's default softness,
#   finger-only grasp helpers removed) and was validated at that softness; with the 0.006 cap the yellow toothpaste
#   pushes the holder block over after every reset (+4-5 deg in 1 s, still tipping after 3 s: 0 / 20 task seeds at
#   rest); without it the block settles within 0.25 s (1.25 deg, then still), as on the code before the cap.
TASK_PHYSICS: dict[str, dict[str, str]] = {
    "prepare_toiletry_sets_of_a_toothbrush_and_toothpaste_on_the_tray": {"solref_max": "0"},
}


def physics(task: str) -> dict[str, Any]:
    """Physics options of a Real2Gym task (base task name or any variant's env name)."""
    import os
    base = split_name(task)[0]
    rel = RELEASE_PHYSICS.get(base, (0.002, "elliptic", 1.0, 0))
    own = TASK_PHYSICS.get(base, {})
    opt = {k: os.environ.get("R2G_" + k.upper(), own.get(k, v)) for k, v in PHYSICS_DEFAULTS.items()}
    dt = max(min(rel[0], 0.002), 0.001) if opt["dt"] == "release" else float(opt["dt"])
    return {"dt": dt, "decimation": max(1, int(round(CONTROL_DT / dt))),
            "cone": rel[1] if opt["cone"] == "release" else opt["cone"],
            "impratio": rel[2] if opt["impratio"] == "release" else float(opt["impratio"]),
            "noslip": rel[3] if opt["noslip"] == "release" else int(opt["noslip"]),
            "solref_max": float(opt["solref_max"]), "sim_guard": opt["sim_guard"] not in ("0", "", "off"),
            "floor_slab": float(opt["floor_slab"])}


# ---------------------------------------------------------------------------------------------- variants
# Stage prompts (real2gym_prompts.json, tools/real2gym_prompts.py): per goal object the source episode's training-label
# sentence that brings a gripper to it (grasp_text) and the one that places it (place_text), in the order the example
# completes them; lid / door sentences for joint goals. Variants of a task <t> (env task = sim task names):
#   r2g_<t>__staged      the full task; the prompt follows the current object's stage (grasp until it is held and lifted,
#                        then place; then joints): the label sentences
#   r2g_<t>__obj_<c>     one object of class <c>: grasp it with the designated arm and bring it to its goal
#   r2g_<t>__pick_<c>    one object of class <c>: grasp it with the designated arm, lift it >= 3 cm and hold it 0.5 s
# Subtask envs (obj / pick), spec_version 2: instruction = enforced spec (real2gym_spec.py).
#   A class <c> is a set of look-alike goal objects (the phrase table's group / phrase), named by the slug of its group
#   (green_capped_glass_bottle); at every reset one valid member is drawn with the reset seed and the live prompt
#   describes it by features and position (qualifier among its look-alikes, side of the table or the container it is in),
#   never by an internal name; the check enforces every clause (arm, target, lift and hold / goal).
#   "prompt" here is the layout-free fallback (no arm, no position), "title" a name for result pages.
#   Legacy aliases r2g_<t>__obj_<object name> / __pick_<object name> (the env names before spec_version 2) fix the
#   instance; they are registered (make_env accepts them: old recordings replay) and listed with include_legacy=True.
#   Their "label_prompt" is the label sentence the env used before (it describes the source demo's stage: often both
#   arms, another object, demo-layout position words; not enforced); a class env lists them as "label_prompts".
#   A class whose members the phrase table all rules out for a kind of env (needs both grippers, pick_ok / obj_ok
#   false) has no env of that kind (listed by ruled_out_classes() with the reason). Legacy aliases of such objects
#   stay registered for replays ("offered" False, class_env None): their spec is not valid and never succeeds.
VARIANT_SEP = "__"
SPEC_VERSION = 2
_CACHE: dict[str, Any] = {}


def load_prompts() -> dict[str, Any]:
    if "prompts" not in _CACHE:
        try:
            _CACHE["prompts"] = json.loads(PROMPTS_FILE.read_text())
        except (OSError, ValueError):
            _CACHE["prompts"] = {}
    return _CACHE["prompts"]


def object_stages(task: str) -> list[dict[str, Any]]:
    """Goal objects with both stage sentences, in the example's completion order (objects met at the start left out)."""
    p = load_prompts().get(task, {})
    return [o for o in p.get("objects", []) if o.get("grasp_text") and o.get("place_text") and not o.get("met_at_start")]


def _build_variants() -> dict[str, dict[str, Any]]:
    from .real2gym_spec import classes, env_title, fallback_prompt, static_validity  # text only (no numpy / mujoco)
    out: dict[str, dict[str, Any]] = {}
    legacy: dict[str, dict[str, Any]] = {}
    _CACHE["ruled_out"] = {}
    prompts = load_prompts()
    for t, cfg in TASKS.items():
        if t not in prompts:
            continue
        objs = object_stages(t)
        if objs:
            out[f"{PREFIX}{t}{VARIANT_SEP}staged"] = {"task": t, "variant": "staged", "object": None,
                                                     "prompt": objs[0]["grasp_text"], "budget": cfg["budget"],
                                                     "title": "Full task with stage prompts: " + t.replace("_", " ")}
        frames = [0] + [o["goal_frame"] for o in objs]
        budget = {o["name"]: int(min(max(450, -(-int(2.0 * max(o["goal_frame"] - frames[i], 60)) // 15) * 15), 1800))
                  for i, o in enumerate(objs)}
        by = {o["name"]: o for o in objs}
        for c in classes(t, [o["name"] for o in objs]):
            mem = c["members"]
            if c["slug"] in by and mem != [c["slug"]]:  # the slug is another object's old env name: keep that alias
                c["slug"] += "_any"
            for kind in ("obj", "pick"):
                why = {m: static_validity(t, kind, m) for m in mem}
                name = f"{PREFIX}{t}{VARIANT_SEP}{kind}_{c['slug']}"
                offered = any(ok for ok, _ in why.values())
                entry = {
                    "task": t, "variant": kind, "object": None, "group": c["slug"], "members": list(mem),
                    "phrase": c["phrase"], "phrase_source": c["source"],
                    "prompt": fallback_prompt(t, kind, mem), "title": env_title(t, kind, mem),
                    "budget": max(budget[m] for m in mem) if kind == "obj" else 450, "spec_version": SPEC_VERSION,
                    "offered": any(ok for ok, _ in why.values()),
                    "not_offered_reason": "" if any(ok for ok, _ in why.values()) else "; ".join(f"{m}: {w}" for m, (_, w) in why.items()),
                    "label_prompts": {m: by[m]["grasp_text"] for m in mem},  # the old label sentence per member
                    "legacy": False}
                if offered:
                    out[name] = entry
                else:
                    _CACHE.setdefault("ruled_out", {})[name] = entry
                for m in mem:
                    legacy[f"{PREFIX}{t}{VARIANT_SEP}{kind}_{m}"] = {
                        "task": t, "variant": kind, "object": m, "group": c["slug"], "members": [m],
                        "phrase": c["phrase"], "phrase_source": c["source"],
                        "prompt": fallback_prompt(t, kind, [m]), "title": env_title(t, kind, [m]),
                        "label_prompt": by[m]["grasp_text"], "label_place_prompt": by[m]["place_text"],
                        "budget": budget[m] if kind == "obj" else 450, "spec_version": SPEC_VERSION,
                        "offered": why[m][0], "not_offered_reason": why[m][1], "legacy": True,
                        "class_env": name if offered else None}
    # a legacy name that is also a class env name (a one-word object name equal to its phrase slug): the class env
    for name, v in legacy.items():
        if name not in out:
            out[name] = v
    return out


def variants(include_legacy: bool = False) -> dict[str, dict[str, Any]]:
    """env task name -> {task, variant, object, prompt, budget, title}; subtask envs also group (class slug), members,
    phrase, spec_version, offered, not_offered_reason, legacy. ``include_legacy``: also the per-instance aliases
    (object = the fixed instance, class_env, label_prompt, label_place_prompt)."""
    if "variants" not in _CACHE:
        _CACHE["variants"] = _build_variants()
    return {n: dict(v) for n, v in _CACHE["variants"].items() if include_legacy or not v.get("legacy")}


def offered_variants() -> dict[str, dict[str, Any]]:
    """variants() (a class env exists only where the phrase table leaves a member valid)."""
    return {n: v for n, v in variants().items() if v.get("offered", True)}


def ruled_out_classes() -> dict[str, dict[str, Any]]:
    """The class envs that do not exist because the phrase table rules out every member (not_offered_reason)."""
    variants()
    return {n: dict(v) for n, v in _CACHE.get("ruled_out", {}).items()}


def variant_info(env_task: str) -> dict[str, Any] | None:
    """The variants() entry of an env task name (class env or legacy alias); None for a plain task."""
    name = env_task if env_task.startswith(PREFIX) else PREFIX + env_task
    v = variants(include_legacy=True).get(name)
    return dict(v) if v else None


def split_name(env_task: str) -> tuple[str, str | None, str | None]:
    """r2g_<t>[__staged | __obj_<k> | __pick_<k>] -> (t, variant, k); k is a class slug or, for a legacy alias, an
    object name (variant_info(env_task) tells which: "object" is None for a class env)."""
    name = env_task[len(PREFIX):] if env_task.startswith(PREFIX) else env_task
    if VARIANT_SEP not in name:
        return name, None, None
    t, v = name.split(VARIANT_SEP, 1)
    if v == "staged":
        return t, "staged", None
    kind, obj = v.split("_", 1)
    return t, kind, obj


# staged mid-task starts r2g_<t>__<pick | obj>_<o>__midtask (abc_sim/real2gym_midtask.py): listed with the variants
from .real2gym_midtask import with_midtask as _with_midtask  # noqa: E402

variants = _with_midtask(variants)

# tool envs r2g_<t>__tool_pick / __tool_<item> (abc_sim/real2gym_tool_envs.py): listed with the variants
from .real2gym_tool_envs import with_tool_envs as _with_tool_envs  # noqa: E402

variants = _with_tool_envs(variants)
