"""Task randomizer registry."""

from __future__ import annotations

from .core import SceneRandomizer
from .tasks.ball_sorting import BallSortingRandomizer
from .tasks.ball_tray import BallTrayBalancingRandomizer
from .tasks.blocks import BlocksRandomizer
from .tasks.bottles import BottlesRandomizer, WaterBottleRandomizer
from .tasks.chess import ChessRandomizer
from .tasks.conveyor_pick import ConveyorPickObjectRandomizer
from .tasks.count_box import CountIntoOpaqueBoxRandomizer
from .tasks.dishrack import DishRackRandomizer
from .tasks.grab_clutter import GrabClutterRandomizer
from .tasks.drawer import DrawerRandomizer
from .tasks.mugs import MugFlipRandomizer, MugTreeRandomizer
from .tasks.marker import MarkerRandomizer
from .tasks.multi_drawer_search import MultiDrawerSearchRandomizer
from .tasks.pour import PourRandomizer
from .tasks.put_relative import PutRelativeRandomizer
from .tasks.openwam_tasks import DrawerOpenCloseRandomizer, FruitOnPlateRandomizer, PlateStackRandomizer, WipeWhiteboardRandomizer, HangHeadphonesRandomizer
from .tasks.sorting import LegoBlocksSortingRandomizer, NutsBoltsSortingRandomizer
from .tasks.sweep import SweepRandomizer

_WATER_BOTTLE_RANDOMIZER = WaterBottleRandomizer()

TASK_RANDOMIZERS: dict[str, SceneRandomizer] = {
    "bottles":      BottlesRandomizer(),
    "put_bottles":  _WATER_BOTTLE_RANDOMIZER,
    "water_bottles": _WATER_BOTTLE_RANDOMIZER,
    "marker":       MarkerRandomizer(),
    "pour":         PourRandomizer(),
    "drawer":       DrawerRandomizer(),
    "dishrack":     DishRackRandomizer(),
    "blocks":       BlocksRandomizer(),
    "mug_tree":     MugTreeRandomizer(),
    "mug_flip":     MugFlipRandomizer(),
    "ball_sorting": BallSortingRandomizer(),
    "chess":        ChessRandomizer(),
    "sweep":        SweepRandomizer(),
    "count_into_opaque_box": CountIntoOpaqueBoxRandomizer(),
    "put_relative": PutRelativeRandomizer(),
    "conveyor_pick": ConveyorPickObjectRandomizer(),
    "ball_tray_balancing": BallTrayBalancingRandomizer(),
    "grab_clutter":  GrabClutterRandomizer(),
    "multi_drawer_search": MultiDrawerSearchRandomizer(),
    "nuts_bolts_sorting": NutsBoltsSortingRandomizer(),
    "lego_blocks_sorting": LegoBlocksSortingRandomizer(),
    "drawer_open":   DrawerOpenCloseRandomizer("open"),
    "drawer_close":  DrawerOpenCloseRandomizer("close"),
    "fruit_on_plate": FruitOnPlateRandomizer(),
    "plates_stack":  PlateStackRandomizer("stack"),
    "plates_unstack": PlateStackRandomizer("unstack"),
    "whiteboard":    WipeWhiteboardRandomizer(),
    "headphones":    HangHeadphonesRandomizer(),
    # "empty": no free-jointed objects to randomize
}

# Real2Gym tasks (abc_sim/real2gym_catalog.py): one randomizer per task, env task r2g_<task>
from ..real2gym_catalog import PREFIX as _R2G_PREFIX, TASKS as _R2G_TASKS  # noqa: E402
from .tasks.real2gym_tasks import Real2GymRandomizer  # noqa: E402

TASK_RANDOMIZERS.update({_R2G_PREFIX + t: Real2GymRandomizer(t) for t in _R2G_TASKS})
from ..real2gym_catalog import variants as _r2g_variants  # noqa: E402

TASK_RANDOMIZERS.update({name: Real2GymRandomizer(v["task"], v["variant"], v["object"], v.get("members"), v.get("group"))
                         for name, v in _r2g_variants(include_legacy=True).items()})


__all__ = ['TASK_RANDOMIZERS']

# staged mid-task starts (abc_sim/real2gym_midtask.py): the predecessors placed at their goals after the reset settle
from .tasks.real2gym_midtask import midtask_randomizers as _r2g_midtask  # noqa: E402

TASK_RANDOMIZERS.update(_r2g_midtask())
