"""Task evaluator registry."""

from __future__ import annotations

from typing import TypeAlias

import mujoco

from abc_sim.task_eval.base import TaskEvaluator
from abc_sim.task_eval.ball_tray import BallTrayBalancingEvaluator
from abc_sim.task_eval.bottles import BottlesInBinEvaluator, PutBottlesInBinEvaluator
from abc_sim.task_eval.bottles_subtask import BottleSubtaskEvaluator
from abc_sim.task_eval.chess import ChessBoardSetupEvaluator
from abc_sim.task_eval.conveyor import ConveyorPickObjectsInBinEvaluator
from abc_sim.task_eval.count_box import CountIntoOpaqueBoxEvaluator
from abc_sim.task_eval.dishrack import DishrackEvaluator
from abc_sim.task_eval.drawer_subtask import DrawerCloseSubtaskEvaluator, DrawerOpenSubtaskEvaluator
from abc_sim.task_eval.dishrack_subtask import DishrackSubtaskEvaluator
from abc_sim.task_eval.grab_clutter import GrabClutterTargetInBoxEvaluator
from abc_sim.task_eval.clutter_subtask import ClutterSubtaskEvaluator
from abc_sim.task_eval.inhand_subtask import InhandSubtaskEvaluator
from abc_sim.task_eval.inhand_transfer import InhandTransferOtherSideEvaluator
from abc_sim.task_eval.lego_blocks import LegoBlocksSortingEvaluator
from abc_sim.task_eval.lego_subtask import LegoSubtaskEvaluator
from abc_sim.task_eval.mug_tree import MugOnRackEvaluator
from abc_sim.task_eval.mug_rack_subtask import MugRackSubtaskEvaluator
from abc_sim.task_eval.multi_drawer import MultiDrawerSearchTargetInBinEvaluator
from abc_sim.task_eval.mugs import MugFlipUprightEvaluator
from abc_sim.task_eval.mug_flip_staged import MugFlipStagedEvaluator
from abc_sim.task_eval.mug_flip_subtask import MugFlipSubtaskEvaluator
from abc_sim.task_eval.nuts_bolts import NutsBoltsSortingEvaluator
from abc_sim.task_eval.nuts_bolts_subtask import NutsBoltsSubtaskEvaluator
from abc_sim.task_eval.openwam_tasks import DrawerOpenCloseEvaluator, FruitOnPlateEvaluator, PlateStackEvaluator, WipeWhiteboardEvaluator, HangHeadphonesEvaluator
from abc_sim.task_eval.pour import PourBeadsInContainerEvaluator
from abc_sim.task_eval.put_relative import PutRelativeEvaluator
from abc_sim.task_eval.put_relative_subtask import PutRelativeSubtaskEvaluator
from abc_sim.task_eval.spell import SpellWordEvaluator
from abc_sim.task_eval.sweep import SweepAwayEvaluator
from abc_sim.task_eval.sweep_subtask import SweepSubtaskEvaluator
from abc_sim.task_specs import SimTaskSpec, maybe_get_task_spec

TaskEvaluatorFactory: TypeAlias = type[TaskEvaluator]


_TASK_EVALUATORS: dict[str, TaskEvaluatorFactory] = {
    "bottles_in_bin": BottlesInBinEvaluator,
    "put_bottles_in_bin": PutBottlesInBinEvaluator,
    "bottles_subtask": BottleSubtaskEvaluator,
    "chess_board_setup": ChessBoardSetupEvaluator,
    "conveyor_pick_objects_in_bin": ConveyorPickObjectsInBinEvaluator,
    "count_into_opaque_box": CountIntoOpaqueBoxEvaluator,
    "dishrack_plates_in_rack": DishrackEvaluator,
    "dishrack_subtask": DishrackSubtaskEvaluator,
    "mug_flip_upright": MugFlipUprightEvaluator,
    "mug_flip_upright_staged": MugFlipStagedEvaluator,
    "mug_flip_subtask": MugFlipSubtaskEvaluator,
    "sweep_away": SweepAwayEvaluator,
    "sweep_subtask": SweepSubtaskEvaluator,
    "nuts_bolts_sorting": NutsBoltsSortingEvaluator,
    "nuts_bolts_subtask": NutsBoltsSubtaskEvaluator,
    "lego_blocks_sorting": LegoBlocksSortingEvaluator,
    "lego_subtask": LegoSubtaskEvaluator,
    "grab_clutter_target_in_box": GrabClutterTargetInBoxEvaluator,
    "clutter_subtask": ClutterSubtaskEvaluator,
    "inhand_transfer_other_side": InhandTransferOtherSideEvaluator,
    "inhand_subtask": InhandSubtaskEvaluator,
    "mug_on_rack": MugOnRackEvaluator,
    "mug_rack_subtask": MugRackSubtaskEvaluator,
    "multi_drawer_target_in_bin": MultiDrawerSearchTargetInBinEvaluator,
    "ball_tray_balancing": BallTrayBalancingEvaluator,
    "pour_beads_in_container": PourBeadsInContainerEvaluator,
    "put_relative": PutRelativeEvaluator,
    "put_relative_subtask": PutRelativeSubtaskEvaluator,
    "spell_word": SpellWordEvaluator,
    "drawer_open_close": DrawerOpenCloseEvaluator,
    "open_drawer_subtask": DrawerOpenSubtaskEvaluator,
    "close_drawer_subtask": DrawerCloseSubtaskEvaluator,
    "fruit_on_plate": FruitOnPlateEvaluator,
    "plates_stack": PlateStackEvaluator,
    "wipe_whiteboard": WipeWhiteboardEvaluator,
    "hang_headphones": HangHeadphonesEvaluator,
}


def make_task_evaluator(
    model: mujoco.MjModel,
    task: str | SimTaskSpec | None,
) -> TaskEvaluator | None:
    """Instantiate a task evaluator for a task name/spec if one is configured."""

    spec = task if isinstance(task, SimTaskSpec) else maybe_get_task_spec(task)
    if spec is None or spec.evaluator_name is None:
        return None

    try:
        evaluator_cls = _TASK_EVALUATORS[spec.evaluator_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown task evaluator {spec.evaluator_name!r} for task {spec.name!r}"
        ) from exc

    return evaluator_cls(model=model, spec=spec, **spec.evaluator_kwargs())
