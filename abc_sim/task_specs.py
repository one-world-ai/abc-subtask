"""Task registry for policy deployment and delivered replay."""

from __future__ import annotations

from dataclasses import dataclass
import re

# Every task in this catalogue is assembled from the one bimanual scene family, so the
# scene name exported alongside an episode is a constant rather than a per-spec field.
# The internal sim datasets record it, so exports have to carry it too.
DEFAULT_SCENE = "hybrid"


@dataclass(frozen=True)
class SimTaskSpec:
    """Canonical definition of a deployable sim task."""

    name: str
    env_task: str
    prompt: str
    max_chunks: int = 10
    randomize: bool = True
    description: str = ""
    aliases: tuple[str, ...] = ()
    evaluator_name: str | None = None
    evaluator_options: tuple[tuple[str, object], ...] = ()

    def evaluator_kwargs(self) -> dict[str, object]:
        """Return evaluator options as a plain kwargs dict."""

        return dict(self.evaluator_options)


def _normalize_key(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


_TASK_SPECS: tuple[SimTaskSpec, ...] = (
    SimTaskSpec(
        name="throw_plastic_bottles_in_bin",
        env_task="bottles",
        prompt="throw plastic bottles in bin",
        description="Throw plastic bottles into the bin.",
        evaluator_name="bottles_in_bin",
        aliases=(
            "bottles",
            "sim_throw_plastic_bottles_in_bin",
        ),
    ),
    SimTaskSpec(
        name="put_plastic_bottles_in_bin",
        env_task="put_bottles",
        prompt="put the plastic bottles in the bin",
        description="Put plastic bottles into the bin.",
        evaluator_name="put_bottles_in_bin",
        aliases=(
            "put_bottles",
            "water_bottles",
            "put plastic bottles in bin",
            "put the plastic bottles in the bin",
            "sim_put the plastic bottles in the bin",
            "sim_put_the_plastic_bottles_in_the_bin",
            "sim_put_the_plastic_bottles_in_bin",
        ),
    ),
    SimTaskSpec(
        name="conveyor_pick",
        env_task="conveyor_pick",
        prompt="pick objects from the conveyor belt and put them in the bin",
        description="Pick objects from a moving conveyor belt before they reach the end and place them into the bin.",
        evaluator_name="conveyor_pick_objects_in_bin",
        aliases=(
            "conveyor_pick",
            "sim_conveyor_pick",
            "pick objects from the conveyor",
            "pick objects off a belt before they fall off the end",
        ),
    ),
    SimTaskSpec(
        name="count_into_opaque_box",
        env_task="count_into_opaque_box",
        prompt="put the prompted number of objects in the opaque box",
        description="Follow a sampled counting directive into the opaque box; the scene randomizer writes the live prompt each episode.",
        evaluator_name="count_into_opaque_box",
        aliases=("sim_count_into_opaque_box",),
    ),
    SimTaskSpec(
        name="count_one_into_opaque_box",
        env_task="count_into_opaque_box",
        evaluator_name="count_into_opaque_box",
        prompt="put exactly one object in the opaque box",
        description="Put exactly one countable object into the opaque box.",
        aliases=(
            "sim_count_one_into_opaque_box",
            "put exactly 1 object in the opaque box",
            "put exactly one item in the opaque box",
            "put 1 item in opaque box",
        ),
    ),
    SimTaskSpec(
        name="count_two_into_opaque_box",
        env_task="count_into_opaque_box",
        evaluator_name="count_into_opaque_box",
        prompt="put exactly two objects in the opaque box",
        description="Put exactly two countable objects into the opaque box.",
        aliases=(
            "sim_count_two_into_opaque_box",
            "put exactly 2 objects in the opaque box",
            "put exactly two items in the opaque box",
            "put 2 items in opaque box",
        ),
    ),
    SimTaskSpec(
        name="count_three_into_opaque_box",
        env_task="count_into_opaque_box",
        evaluator_name="count_into_opaque_box",
        prompt="put exactly three objects in the opaque box",
        description="Put exactly three countable objects into the opaque box.",
        aliases=(
            "sim_count_three_into_opaque_box",
            "put exactly 3 objects in the opaque box",
            "put exactly three items in the opaque box",
            "put 3 items in opaque box",
        ),
    ),
    SimTaskSpec(
        name="count_four_into_opaque_box",
        env_task="count_into_opaque_box",
        evaluator_name="count_into_opaque_box",
        prompt="put exactly four objects in the opaque box",
        description="Put exactly four countable objects into the opaque box.",
        aliases=(
            "sim_count_four_into_opaque_box",
            "put exactly 4 objects in the opaque box",
            "put exactly four items in the opaque box",
            "put 4 items in opaque box",
        ),
    ),
    SimTaskSpec(
        name="put_relative",
        env_task="put_relative",
        prompt="put one object in a relative position to another object",
        description="Place one sampled object in a prompted relative direction around another sampled object.",
        evaluator_name="put_relative",
        aliases=(
            "put_relative",
            "sim_put_relative",
            "relative_put",
            "put object relative to object",
            "put one object relative to another",
        ),
    ),
    SimTaskSpec(name="grab_specific_object_from_clutter__reach", env_task="grab_clutter",
                prompt="clutter subtask: approach the prompted target",
                description="Clutter reach subtask: start with env.reset_to_stage.",
                evaluator_name="clutter_subtask", evaluator_options=(("stage", "reach"),)),
    SimTaskSpec(name="grab_specific_object_from_clutter__pick", env_task="grab_clutter",
                prompt="clutter subtask: grasp and lift the prompted target",
                description="Clutter pick subtask: start with env.reset_to_stage.",
                evaluator_name="clutter_subtask", evaluator_options=(("stage", "pick"),)),
    SimTaskSpec(name="grab_specific_object_from_clutter__handoff", env_task="grab_clutter",
                prompt="clutter subtask: hand the prompted target to the box-side arm",
                description="Clutter handoff subtask (opposite-side scenes): start with env.reset_to_stage.",
                evaluator_name="clutter_subtask", evaluator_options=(("stage", "handoff"),)),
    SimTaskSpec(name="grab_specific_object_from_clutter__carry", env_task="grab_clutter",
                prompt="clutter subtask: carry the prompted target over the box",
                description="Clutter carry subtask: start with env.reset_to_stage.",
                evaluator_name="clutter_subtask", evaluator_options=(("stage", "carry"),)),
    SimTaskSpec(name="grab_specific_object_from_clutter__place", env_task="grab_clutter",
                prompt="clutter subtask: lower the prompted target into the box and release",
                description="Clutter place subtask: start with env.reset_to_stage.",
                evaluator_name="clutter_subtask", evaluator_options=(("stage", "place"),)),
    SimTaskSpec(
        name="grab_specific_object_from_clutter",
        env_task="grab_clutter",
        prompt="grab the target object from the cluttered table and place it in the box",
        description="Grab the prompted target object from a cluttered tabletop full of distractors and place it in the target box.",
        evaluator_name="grab_clutter_target_in_box",
        aliases=(
            "grab_clutter",
            "grab object from clutter",
            "grab specific object from clutter",
            "grab specific object from cluttered table",
            "pick specific object from clutter",
            "sim_grab_specific_object_from_clutter",
            "sim_grab_specific_object_from_cluttered_table",
        ),
    ),
    SimTaskSpec(
        name="ball_tray_balancing",
        env_task="ball_tray_balancing",
        prompt="grab the tray handles and keep the ball balanced on the tray",
        description="Grab a tabletop tray by its handles and keep a ball balanced on the open tray for low-latency policy inference testing.",
        evaluator_name="ball_tray_balancing",
        aliases=(
            "ball tray balancing",
            "ball-on-tray balancing",
            "ball on tray balancing",
            "balance ball on tray",
            "keep ball on tray",
            "sim_ball_tray_balancing",
            "sim_ball_on_tray_balancing",
        ),
    ),
    SimTaskSpec(
        name="inhand_transfer_item_to_other_side",
        env_task="inhand_transfer",
        prompt="inhand transfer the item to other side",
        description="Pick the sampled kitchen tool with the near arm, pass it to the other hand, and set it down on the far side of the table.",
        evaluator_name="inhand_transfer_other_side",
        aliases=(
            "inhand_transfer",
            "inhand transfer",
            "sim_inhand_transfer",
            "sim_inhand_transfer_the_item_to_other_side",
            "transfer the item to the other side",
        ),
    ),
    SimTaskSpec(
        name="multi_drawer_search",
        env_task="multi_drawer_search",
        prompt="find the target object hidden in one of the drawers and place it in the bin",
        description="Remember objects seen across several drawers, then find and place a prompted sequence of them into the bin.",
        evaluator_name="multi_drawer_target_in_bin",
        aliases=(
            "drawer search",
            "multi drawer search",
            "multi-drawer search",
            "find object in drawer",
            "find block in drawer",
            "find target in drawer",
            "find target hidden in drawer",
            "sim_multi_drawer_search",
            "sim_find_target_hidden_in_drawer",
        ),
    ),
    SimTaskSpec(
        name="put_markers_in_top_drawer",
        env_task="drawer",
        prompt="put the markers in the top drawer",
        aliases=("sim_put_the_markers_in_the_top_drawer",),
    ),
    SimTaskSpec(
        name="put_markers_in_middle_drawer",
        env_task="drawer",
        prompt="put the markers in the middle drawer",
        aliases=("sim_put_the_markers_in_the_middle_drawer",),
    ),
    SimTaskSpec(
        name="put_markers_in_bottom_drawer",
        env_task="drawer",
        prompt="put the markers in the bottom drawer",
        aliases=("sim_put_the_markers_in_the_bottom_drawer",),
    ),
    SimTaskSpec(
        name="pouring",
        env_task="pour",
        prompt="pour from one container into another",
        evaluator_name="pour_beads_in_container",
        aliases=("sim_pouring", "sim_pouring_beads", "pour"),
    ),
    SimTaskSpec(
        name="hang_mug_on_mug_rack",
        env_task="mug_tree",
        prompt="hang mug on mug rack",
        evaluator_name="mug_on_rack",
        aliases=(
            "sim_hang_mug_on_mug_rack",
            "sim_hang_the_mug_on_the_mug_rack",
            "mug_tree",
        ),
    ),
    SimTaskSpec(
        name="turn_mug_right_side_up",
        env_task="mug_flip",
        prompt="turn mug right side up",
        evaluator_name="mug_flip_upright",
        aliases=(
            "sim_turn_mug_right_side_up",
            "sim_turn_the_mug_right_side_up",
            "mug_flip",
        ),
    ),
    SimTaskSpec(
        name="turn_mug_right_side_up_staged",
        env_task="mug_flip",
        # Not "turn mug right side up": prompts are lookup keys, and that one normalizes to the stock task's name.
        prompt="turn the mug right side up",
        description="turn_mug_right_side_up with a staged RL reward (same scene, prompt and success).",
        evaluator_name="mug_flip_upright_staged",
    ),
    SimTaskSpec(
        name="turn_mug_right_side_up__reach",
        env_task="mug_flip",
        prompt="move the closest gripper to the mug",
        description="Mug-flip subtask 'reach' for single-stage RL: start with env.reset_to_stage (mug_flip_subtask.py).",
        evaluator_name="mug_flip_subtask",
        evaluator_options=(("stage", "reach"),),
    ),
    SimTaskSpec(
        name="turn_mug_right_side_up__pick",
        env_task="mug_flip",
        prompt="grasp the mug and lift it",
        description="Mug-flip subtask 'pick' for single-stage RL: start with env.reset_to_stage (mug_flip_subtask.py).",
        evaluator_name="mug_flip_subtask",
        evaluator_options=(("stage", "pick"),),
    ),
    SimTaskSpec(
        name="turn_mug_right_side_up__rotate",
        env_task="mug_flip",
        prompt="rotate the mug upright in the gripper",
        description="Mug-flip subtask 'rotate' for single-stage RL: start with env.reset_to_stage (mug_flip_subtask.py).",
        evaluator_name="mug_flip_subtask",
        evaluator_options=(("stage", "rotate"),),
    ),
    SimTaskSpec(
        name="turn_mug_right_side_up__place",
        env_task="mug_flip",
        prompt="put the mug down upright and let go",
        description="Mug-flip subtask 'place' for single-stage RL: start with env.reset_to_stage (mug_flip_subtask.py).",
        evaluator_name="mug_flip_subtask",
        evaluator_options=(("stage", "place"),),
    ),
    SimTaskSpec(
        name="put_relative__reach",
        env_task="put_relative",
        prompt="put relative subtask: move a gripper to the object to move",
        description="put_relative subtask 'reach' for single-stage RL: start with env.reset_to_stage (put_relative_subtask.py).",
        evaluator_name="put_relative_subtask",
        evaluator_options=(("stage", "reach"),),
    ),
    SimTaskSpec(
        name="put_relative__pick",
        env_task="put_relative",
        prompt="put relative subtask: grasp the object to move and lift it",
        description="put_relative subtask 'pick' for single-stage RL: start with env.reset_to_stage (put_relative_subtask.py).",
        evaluator_name="put_relative_subtask",
        evaluator_options=(("stage", "pick"),),
    ),
    SimTaskSpec(
        name="put_relative__handoff",
        env_task="put_relative",
        prompt="put relative subtask: hand the object to the other arm",
        description="put_relative subtask 'handoff' (cross-table episodes) for single-stage RL: start with env.reset_to_stage (put_relative_subtask.py).",
        evaluator_name="put_relative_subtask",
        evaluator_options=(("stage", "handoff"),),
    ),
    SimTaskSpec(
        name="put_relative__carry",
        env_task="put_relative",
        prompt="put relative subtask: carry the object to its goal",
        description="put_relative subtask 'carry' for single-stage RL: start with env.reset_to_stage (put_relative_subtask.py).",
        evaluator_name="put_relative_subtask",
        evaluator_options=(("stage", "carry"),),
    ),
    SimTaskSpec(
        name="put_relative__place",
        env_task="put_relative",
        prompt="put relative subtask: put the object down at its goal",
        description="put_relative subtask 'place' for single-stage RL: start with env.reset_to_stage (put_relative_subtask.py).",
        evaluator_name="put_relative_subtask",
        evaluator_options=(("stage", "place"),),
    ),
    SimTaskSpec(
        name="sweep_away_paper_scraps_from_table",
        env_task="sweep",
        prompt="sweep away paper scraps from the table",
        evaluator_name="sweep_away",
        aliases=("sim_sweep_away_paper_scraps_from_the_table", "sweep"),
    ),
    SimTaskSpec(
        name="sort_nuts_and_bolts__reach",
        env_task="nuts_bolts_sorting",
        prompt="nuts bolts subtask: move a gripper to the closest unsorted part",
        description="Nuts/bolts reach subtask; start with env.reset_to_stage.",
        evaluator_name="nuts_bolts_subtask",
        evaluator_options=(("stage", "reach"),),
    ),
    SimTaskSpec(
        name="sort_nuts_and_bolts__pick",
        env_task="nuts_bolts_sorting",
        prompt="nuts bolts subtask: grasp the target part and lift it",
        description="Nuts/bolts pick subtask; start with env.reset_to_stage.",
        evaluator_name="nuts_bolts_subtask",
        evaluator_options=(("stage", "pick"),),
    ),
    SimTaskSpec(
        name="sort_nuts_and_bolts__handoff",
        env_task="nuts_bolts_sorting",
        prompt="nuts bolts subtask: hand the target part to the gripper on its bin's side",
        description="Nuts/bolts handoff subtask (cross-side parts only); start with env.reset_to_stage.",
        evaluator_name="nuts_bolts_subtask",
        evaluator_options=(("stage", "handoff"),),
    ),
    SimTaskSpec(
        name="sort_nuts_and_bolts__carry",
        env_task="nuts_bolts_sorting",
        prompt="nuts bolts subtask: carry the target part over its assigned bin",
        description="Nuts/bolts carry subtask; start with env.reset_to_stage.",
        evaluator_name="nuts_bolts_subtask",
        evaluator_options=(("stage", "carry"),),
    ),
    SimTaskSpec(
        name="sort_nuts_and_bolts__drop",
        env_task="nuts_bolts_sorting",
        prompt="nuts bolts subtask: release the target part into its assigned bin",
        description="Nuts/bolts drop subtask; start with env.reset_to_stage.",
        evaluator_name="nuts_bolts_subtask",
        evaluator_options=(("stage", "drop"),),
    ),
    SimTaskSpec(
        name="sort_nuts_and_bolts",
        env_task="nuts_bolts_sorting",
        prompt="sort the nuts and bolts into their matching bins",
        description="Sort loose nuts into the teal bin and loose bolts into the orange bin.",
        evaluator_name="nuts_bolts_sorting",
        aliases=(
            "nuts_bolts_sorting",
            "nuts and bolts sorting",
            "sort nuts and bolts",
            "sim_sort_nuts_and_bolts",
            "sim_sort_the_nuts_and_bolts",
        ),
    ),
    SimTaskSpec(
        name="sort_lego_blocks",
        env_task="lego_blocks_sorting",
        prompt="sort the red, yellow, and blue lego blocks into their matching bins",
        description="Sort red, yellow, and blue LEGO-style blocks into matching color bins.",
        evaluator_name="lego_blocks_sorting",
        aliases=(
            "lego_blocks_sorting",
            "lego sorting",
            "lego block sorting",
            "sort lego blocks",
            "sort red yellow blue lego blocks",
            "sim_sort_lego_blocks",
            "sim_sort_the_lego_blocks",
        ),
    ),
    SimTaskSpec(
        name="sort_lego_blocks__reach",
        env_task="lego_blocks_sorting",
        prompt="lego sorting subtask: move a gripper to the target block",
        description="LEGO subtask reach: start with env.reset_to_stage.",
        evaluator_name="lego_subtask",
        evaluator_options=(("stage", "reach"),),
    ),
    SimTaskSpec(
        name="sort_lego_blocks__pick",
        env_task="lego_blocks_sorting",
        prompt="lego sorting subtask: grasp the target block and lift it",
        description="LEGO subtask pick: start with env.reset_to_stage.",
        evaluator_name="lego_subtask",
        evaluator_options=(("stage", "pick"),),
    ),
    SimTaskSpec(
        name="sort_lego_blocks__handoff",
        env_task="lego_blocks_sorting",
        prompt="lego sorting subtask: hand the target block to the arm on its bin's side",
        description="LEGO subtask handoff (cross-side blocks only): start with env.reset_to_stage.",
        evaluator_name="lego_subtask",
        evaluator_options=(("stage", "handoff"),),
    ),
    SimTaskSpec(
        name="sort_lego_blocks__carry",
        env_task="lego_blocks_sorting",
        prompt="lego sorting subtask: carry the target block over its matching bin",
        description="LEGO subtask carry: start with env.reset_to_stage.",
        evaluator_name="lego_subtask",
        evaluator_options=(("stage", "carry"),),
    ),
    SimTaskSpec(
        name="sort_lego_blocks__drop",
        env_task="lego_blocks_sorting",
        prompt="lego sorting subtask: release the target block into its matching bin",
        description="LEGO subtask drop: start with env.reset_to_stage.",
        evaluator_name="lego_subtask",
        evaluator_options=(("stage", "drop"),),
    ),
    SimTaskSpec(
        name="load_plates_into_dish_rack",
        env_task="dishrack",
        prompt="load plates into tabletop dish rack",
        evaluator_name="dishrack_plates_in_rack",
        evaluator_options=(
            ("margin_m", 0.05),
            ("height_min_m", -0.06),
            ("height_max_m", 0.45),
            ("upright_normal_z_max", 0.55),
            ("horizontal_normal_z_min", 0.75),
        ),
        aliases=(
            "sim_load_plates_into_tabletop_dish_rack",
            "sim_load_the_plates_into_the_dish_rack",
            "load the plates into the dish rack",
            "dishrack",
        ),
    ),
    SimTaskSpec(
        name="set_up_chess_pieces_on_board",
        env_task="chess",
        prompt="set up chess pieces on the board",
        evaluator_name="chess_board_setup",
        aliases=("sim_set_up_chess_pieces_on_the_board", "chess"),
    ),
    SimTaskSpec(
        name="build_wood_block_tower",
        env_task="building_blocks",
        prompt="build a wood block tower",
        aliases=("sim_build_wood_block_tower", "building_blocks"),
    ),
    SimTaskSpec(
        name="spell_cat",
        env_task="blocks",
        prompt="spell cat",
        evaluator_name="spell_word",
        evaluator_options=(("word", "cat"),),
        aliases=("sim_spell_cat",),
    ),
    SimTaskSpec(
        name="spell_dog",
        env_task="blocks",
        prompt="spell dog",
        evaluator_name="spell_word",
        evaluator_options=(("word", "dog"),),
        aliases=("sim_spell_dog",),
    ),
    SimTaskSpec(
        name="spell_fish",
        env_task="blocks",
        prompt="spell fish",
        evaluator_name="spell_word",
        evaluator_options=(("word", "fish"),),
        aliases=("sim_spell_fish",),
    ),
    SimTaskSpec(
        name="spell_bair",
        env_task="blocks",
        prompt="spell bair",
        evaluator_name="spell_word",
        evaluator_options=(("word", "bair"),),
        aliases=("sim_spell_bair",),
    ),
    SimTaskSpec(
        name="spell_xdof",
        env_task="blocks",
        prompt="spell xdof",
        evaluator_name="spell_word",
        evaluator_options=(("word", "xdof"),),
        aliases=("sim_spell_xdof",),
    ),
    SimTaskSpec(
        name="spell_abc",
        env_task="blocks",
        prompt="spell abc",
        evaluator_name="spell_word",
        evaluator_options=(("word", "abc"),),
        aliases=("sim_spell_abc",),
    ),
    SimTaskSpec(
        name="spell_yam",
        env_task="blocks",
        prompt="spell yam",
        evaluator_name="spell_word",
        evaluator_options=(("word", "yam"),),
        aliases=("sim_spell_yam",),
    ),
    SimTaskSpec(
        name="spell_agi",
        env_task="blocks",
        prompt="spell agi",
        evaluator_name="spell_word",
        evaluator_options=(("word", "agi"),),
        aliases=("sim_spell_agi",),
    ),
    # OpenWAM-added tasks (randomization/tasks/openwam_tasks.py, task_eval/openwam_tasks.py). The randomizer sets
    # the per-world prompt; the spec prompt is the generic form.
    SimTaskSpec(
        name="open_drawer",
        env_task="drawer_open",
        prompt="open the drawer",
        description="Open the named drawer (top, middle or bottom) of the three-drawer cabinet; leave the others closed.",
        evaluator_name="drawer_open_close",
        evaluator_options=(("mode", "open"), ("open_m", 0.12), ("other_max_m", 0.05)),
        aliases=("sim_open_drawer", "open the top drawer", "open the middle drawer", "open the bottom drawer"),
    ),
    SimTaskSpec(
        name="close_drawer",
        env_task="drawer_close",
        prompt="close the drawer",
        description="Close the named drawer, which starts 14-19 cm open; leave the others closed.",
        evaluator_name="drawer_open_close",
        evaluator_options=(("mode", "close"), ("closed_m", 0.015), ("other_max_m", 0.05)),
        aliases=("sim_close_drawer", "close the top drawer", "close the middle drawer", "close the bottom drawer"),
    ),
    SimTaskSpec(
        name="put_fruit_on_plate",
        env_task="fruit_on_plate",
        prompt="put the fruit on the plate",
        description="Put the named fruit (apple, lemon, orange, pear or tomato) on the plate; the other fruits stay off it.",
        evaluator_name="fruit_on_plate",
        aliases=("sim_put_fruit_on_plate", "fruit_on_plate"),
    ),
    SimTaskSpec(
        name="stack_plates",
        env_task="plates_stack",
        prompt="stack the plates",
        description="Stack 2-3 plates spread on the table into one nested stack.",
        evaluator_name="plates_stack",
        evaluator_options=(("mode", "stack"),),
        aliases=("sim_stack_plates", "stack_the_plates"),
    ),
    SimTaskSpec(
        name="unstack_plates",
        env_task="plates_unstack",
        prompt="unstack the plates",
        description="Take a stack of 2-3 plates apart, laying each plate on the table on its own.",
        evaluator_name="plates_stack",
        evaluator_options=(("mode", "unstack"),),
        aliases=("sim_unstack_plates", "unstack_the_plates"),
    ),
    SimTaskSpec(
        name="wipe_whiteboard",
        env_task="whiteboard",
        prompt="wipe the whiteboard clean",
        description="Pick up the duster and wipe the marker strokes off the whiteboard lying on the table.",
        evaluator_name="wipe_whiteboard",
        evaluator_options=(("erased_frac", 0.9),),
        aliases=("sim_wipe_whiteboard", "wipe the whiteboard"),
    ),
    SimTaskSpec(
        name="hang_headphones",
        env_task="headphones",
        prompt="hang the headphones on the stand",
        description="Pick up the headphones lying on the table and hang them on the headset stand.",
        evaluator_name="hang_headphones",
        aliases=("sim_hang_headphones", "hang the headphones"),
    ),
)


_TASK_SPECS += tuple(
    SimTaskSpec(
        name=f"put_plastic_bottles_in_bin__{stage}",
        env_task="put_bottles",
        prompt=prompt,
        description=f"Bottle subtask '{stage}': start with env.reset_to_stage (bottles_subtask.py).",
        evaluator_name="bottles_subtask",
        evaluator_options=(("stage", stage),),
    )
    for stage, prompt in (
        ("reach", "bottle subtask: move a gripper to the target bottle"),
        ("pick", "bottle subtask: grasp the target bottle and lift it"),
        ("carry", "bottle subtask: carry the target bottle over the bin opening"),
        ("drop", "bottle subtask: release the target bottle into the bin"),
    )
)


# Sweep hand-pick chain H (approved 2026-11-14); see subtask_rewards/sweep/.
_TASK_SPECS += tuple(
    SimTaskSpec(
        name=f"sweep_away_paper_scraps_from_table__{stage}",
        env_task="sweep",
        prompt=prompt,
        description=f"Sweep subtask '{stage}' (hand-pick chain): start with env.reset_to_stage (sweep_subtask.py).",
        evaluator_name="sweep_subtask",
        evaluator_options=(("stage", stage),),
    )
    for stage, prompt in (
        ("reach", "sweep subtask: move a gripper to the target paper scrap"),
        ("pick", "sweep subtask: grasp the target paper scrap and lift it"),
        ("carry", "sweep subtask: carry the target paper scrap over the bin opening"),
        ("drop", "sweep subtask: release the target paper scrap into the bin"),
    )
)

# inhand_transfer_item_to_other_side subtasks (subtask_rewards/inhand; approved stage table 2026-11-14).
_TASK_SPECS += tuple(
    SimTaskSpec(
        name=f"inhand_transfer_item_to_other_side__{stage}",
        env_task="inhand_transfer",
        prompt=prompt,
        description=f"Inhand transfer subtask '{stage}': start with env.reset_to_stage (inhand_subtask.py).",
        evaluator_name="inhand_subtask",
        evaluator_options=(("stage", stage),),
    )
    for stage, prompt in (
        ("reach", "inhand transfer subtask: reach the item with the near arm"),
        ("pick", "inhand transfer subtask: grasp and lift the item with the near arm"),
        ("present", "inhand transfer subtask: bring the item to the middle"),
        ("handoff", "inhand transfer subtask: hand the item to the other arm"),
        ("carry", "inhand transfer subtask: carry the item to the other side"),
        ("place", "inhand transfer subtask: set the item down on the other side"),
    )
)

# Mug-rack subtasks (subtask_rewards/mug_rack): one block per task so worktrees merge cleanly.
_TASK_SPECS += tuple(
    SimTaskSpec(
        name=f"hang_mug_on_mug_rack__{stage}",
        env_task="mug_tree",
        prompt=prompt,
        description=f"Mug-rack subtask '{stage}': start with env.reset_to_stage (mug_rack_subtask.py).",
        evaluator_name="mug_rack_subtask",
        evaluator_options=(("stage", stage),),
    )
    for stage, prompt in (
        ("reach", "mug rack subtask: move a gripper to the target mug"),
        ("pick", "mug rack subtask: grasp the target mug away from its handle and lift it"),
        ("align", "mug rack subtask: line the target mug's handle up with a free peg"),
        ("insert", "mug rack subtask: slide the target mug's handle onto the peg"),
        ("release", "mug rack subtask: let go of the target mug on the rack"),
    )
)

# open_drawer / close_drawer subtasks (subtask_rewards/drawer; task_eval/drawer_subtask.py). The live prompt names the
# target drawer (drawer_stages.OPEN_LIVE / CLOSE_LIVE); the spec prompt is the generic form.
_TASK_SPECS += tuple(
    SimTaskSpec(
        name=f"{task}__{stage}",
        env_task=env_task,
        prompt=prompt,
        description=f"{task} subtask '{stage}' for single-stage RL: start with env.reset_to_stage (drawer_subtask.py).",
        evaluator_name=f"{task}_subtask",
        evaluator_options=(("stage", stage),),
    )
    for task, env_task, prompts in (
        ("open_drawer", "drawer_open", (
            ("reach", "open drawer subtask: move a gripper to the drawer handle"),
            ("grasp", "open drawer subtask: grasp the drawer handle"),
            ("pull", "open drawer subtask: pull the drawer open"),
            ("release", "open drawer subtask: let go of the drawer"))),
        ("close_drawer", "drawer_close", (
            ("reach", "close drawer subtask: move a gripper to the front of the drawer"),
            ("push", "close drawer subtask: push the drawer closed"),
            ("release", "close drawer subtask: move the gripper away from the drawer"))),
    )
    for stage, prompt in prompts
)

# Dish-rack subtasks (subtask_rewards/dishrack): reach/pick STOPPED at G1, carry/insert/release only.
_TASK_SPECS += tuple(
    SimTaskSpec(
        name=f"load_plates_into_dish_rack__{stage}",
        env_task="dishrack",
        prompt=prompt,
        description=f"Dish-rack subtask '{stage}': start with env.reset_to_stage (dishrack_subtask.py).",
        evaluator_name="dishrack_subtask",
        evaluator_options=(("stage", stage),),
    )
    for stage, prompt in (
        ("carry", "dish rack subtask: carry the plate upright above the dish rack"),
        ("insert", "dish rack subtask: lower the plate into the rack slot"),
        ("release", "dish rack subtask: release the plate in the rack"),
    )
)

_LOOKUP: dict[str, SimTaskSpec] = {}
for _spec in _TASK_SPECS:
    for _alias in {_spec.name, _spec.prompt, *_spec.aliases}:
        _LOOKUP[_normalize_key(_alias)] = _spec


def list_task_specs() -> tuple[SimTaskSpec, ...]:
    """Return all known sim task specs."""

    return _TASK_SPECS


def get_task_spec(name: str) -> SimTaskSpec:
    """Resolve a task spec from a canonical name, alias, or prompt text."""

    key = _normalize_key(name)
    try:
        return _LOOKUP[key]
    except KeyError as exc:
        available = ", ".join(spec.name for spec in _TASK_SPECS)
        raise KeyError(f"Unknown sim task '{name}'. Available: {available}") from exc


def maybe_get_task_spec(name: str | None) -> SimTaskSpec | None:
    """Resolve a task spec if present, otherwise return ``None``."""

    if not name:
        return None
    return _LOOKUP.get(_normalize_key(name))
