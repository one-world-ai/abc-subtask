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
        name="sweep_away_paper_scraps_from_table",
        env_task="sweep",
        prompt="sweep away paper scraps from the table",
        evaluator_name="sweep_away",
        aliases=("sim_sweep_away_paper_scraps_from_the_table", "sweep"),
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


# Real2Gym tasks and their variants (imported from openwam-sim-eval abc_tasks/rl_suite): the lookup never lets them
# take an ABC task's or subtask's name, prompt or alias (_build_lookup).
# Real2Gym tasks (abc_sim/real2gym_catalog.py; randomization/tasks/real2gym_tasks.py, task_eval/real2gym_tasks.py)
from abc_sim.real2gym_catalog import PREFIX as _R2G_PREFIX, TASKS as _R2G_TASKS, task_config as _r2g_config  # noqa: E402
from abc_sim.real2gym_catalog import variants as _r2g_variants  # noqa: E402

_TASK_SPECS += tuple(
    SimTaskSpec(
        name=_R2G_PREFIX + t,
        env_task=_R2G_PREFIX + t,
        prompt=_r2g_config(t)["prompt"],
        description=f"Real2Gym (OneWorld-AI/ABC_simulators) task {t} on the abc table: " + _r2g_config(t)["prompt"] + ".",
        evaluator_name="real2gym",
        evaluator_options=(("task", t),),
        aliases=(f"sim_{_R2G_PREFIX}{t}",),
    )
    for t in _R2G_TASKS
)
_TASK_SPECS += tuple(
    SimTaskSpec(
        name=name,
        env_task=name,
        prompt=v["prompt"],
        description=f"Real2Gym {v['task']} ({v['variant']}" + (f", {v['object']}" if v["object"] else "") + "): "
        + ("stage prompts from the source episode's training labels." if v["variant"] == "staged" else
           "the instruction is rendered at reset from the layout and enforced by the check (real2gym_spec.py)."),
        evaluator_name="real2gym",
        evaluator_options=(("task", v["task"]), ("variant", v["variant"]), ("object", v["object"]),
                           ("members", tuple(v.get("members") or ())), ("group", v.get("group"))),
        aliases=(),
    )
    for name, v in _r2g_variants(include_legacy=True).items()
)


def _build_lookup(specs: tuple[SimTaskSpec, ...], stock: tuple[SimTaskSpec, ...]) -> dict[str, SimTaskSpec]:
    """name / prompt / alias -> spec. A name is unique and always resolves to its spec; stock tasks keep their prompts
    and aliases; a derived spec's prompt or alias is a key only where no name and no stock key has it."""
    names: dict[str, SimTaskSpec] = {}
    for spec in specs:
        key = _normalize_key(spec.name)
        if key in names:
            raise ValueError(f"sim task names {names[key].name!r} and {spec.name!r} share the lookup key {key!r}")
        names[key] = spec
    stock_ids = {id(spec) for spec in stock}
    lookup: dict[str, SimTaskSpec] = {}
    for spec in stock:
        for alias in {spec.prompt, *spec.aliases}:
            key = _normalize_key(alias)
            if key in names and names[key] is not spec:
                continue
            if key in lookup and lookup[key] is not spec:
                raise ValueError(f"task key {alias!r} is used by {lookup[key].name!r} and {spec.name!r}")
            lookup[key] = spec
    stock_keys = set(lookup)
    for spec in specs:
        if id(spec) in stock_ids:
            continue
        for alias in {spec.prompt, *spec.aliases}:
            key = _normalize_key(alias)
            if key not in names and key not in stock_keys:
                lookup.setdefault(key, spec)
    lookup.update(names)
    return lookup


# Single-stage subtask specs (<task>__<stage>) are not listed here: each task module declares its own and
# abc_sim/task_eval/subtasks.py discovers them on first lookup (no shared file to edit per task).
_ALL_SPECS: tuple[SimTaskSpec, ...] | None = None
_LOOKUP: dict[str, SimTaskSpec] = {}


def _catalogue() -> tuple[tuple[SimTaskSpec, ...], dict[str, SimTaskSpec]]:
    global _ALL_SPECS
    if _ALL_SPECS is None:
        from abc_sim.task_eval.subtasks import subtask_specs  # lazy: task modules import this module

        specs = _TASK_SPECS + subtask_specs()
        # strict (as before): every spec but the imported Real2Gym task / variant specs, whose prompts may repeat
        _LOOKUP.update(_build_lookup(specs, tuple(sp for sp in specs if sp.evaluator_name != "real2gym")))
        _ALL_SPECS = specs
    return _ALL_SPECS, _LOOKUP


def list_task_specs() -> tuple[SimTaskSpec, ...]:
    """Return all known sim task specs."""

    return _catalogue()[0]


def get_task_spec(name: str) -> SimTaskSpec:
    """Resolve a task spec from a canonical name, alias, or prompt text."""

    specs, lookup = _catalogue()
    key = _normalize_key(name)
    try:
        return lookup[key]
    except KeyError as exc:
        available = ", ".join(spec.name for spec in specs)
        raise KeyError(f"Unknown sim task '{name}'. Available: {available}") from exc


def maybe_get_task_spec(name: str | None) -> SimTaskSpec | None:
    """Resolve a task spec if present, otherwise return ``None``."""

    if not name:
        return None
    return _catalogue()[1].get(_normalize_key(name))
