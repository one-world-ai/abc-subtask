"""Tool-use envs of the Real2Gym tasks whose demos move the items with tongs / a clamp (fix-tools).

  r2g_<task>__tool_pick         "Move the <arm> arm to the <tool> and grasp it, then lift it off the table."
                                check (task_eval/real2gym_tooluse.py): both fingers of the <arm> gripper touch the tool, its
                                centre of mass >= 3 cm above its reset height, it touches nothing but the robot; held 0.5 s
  r2g_<task>__tool_<item>       "Grasp the <tool> with the <arm> arm and use it to pick up the <item> and <goal clause>."
                                check: the item meets its catalog goal, at rest, touching neither the tool nor a finger, held
                                0.5 s, AND earlier in the episode the item was carried by the tool: both tool jaws on it, the
                                tool pinched by the <arm> gripper, no finger on it, >= 2 cm above its reset height and
                                touching nothing but the tool, for >= 0.3 s. An item pushed or carried by hand does not pay.
Only the tasks / items listed in TOOL_ENVS are registered: those where scripted probes showed the physics can do it
(rl_envs/reward_align/tools/). Phrases from the verified phrase tables (no internal names).

Only json at import time (real2gym_catalog wraps its variants() with ``with_tool_envs``).
"""
from __future__ import annotations

import functools
import os
from typing import Any

# task -> tool body, its phrase, the arm that holds it (the demo's), goal clause, item -> phrase; items: the ones the
# probes carried with the tool (empty: only the tool pick-up is offered)
TOOL_ENVS: dict[str, dict[str, Any]] = {
    "place_the_bread": {"tool": "food_tongs", "tool_phrase": "metal tongs", "it": "them", "arm": "right",
                        "goal_clause": "put it on the square white plate", "items": {}},
    "pack_the_cookies": {"tool": "tweezers", "tool_phrase": "metal tongs", "it": "them", "arm": "left",
                         "goal_clause": "put it in the square cookie tin", "items": {}},
    "serve_the_lunch_box": {"tool": "tongs", "tool_phrase": "clamp", "it": "it", "arm": "left",
                            "goal_clause": "put it into the lunch box", "items": {}},
}


def tool_pick_prompt(cfg: dict[str, Any]) -> str:
    it = cfg.get("it", "it")
    return f"Move the {cfg['arm']} arm to the {cfg['tool_phrase']} and grasp {it}, then lift {it} off the table."


def tool_members(task: str, item: str) -> list[str]:
    """The look-alikes a tool-use env accepts (the item itself if it has none)."""
    it = TOOL_ENVS.get(task, {}).get("items", {}).get(item)
    return list(it["members"]) if isinstance(it, dict) else [item]


def tool_use_prompt(cfg: dict[str, Any], item_phrase: str, any_of: bool = False) -> str:
    art = "a" if any_of else "the"
    return (f"Grasp the {cfg['tool_phrase']} with the {cfg['arm']} arm and use {cfg.get('it', 'it')} to pick up {art} {item_phrase} "
            f"and {cfg['goal_clause']}.")


# validation only: R2G_TOOL_EXTRA_ITEMS="task:item:phrase;..." registers more tool-use envs (to test the check on items
# the physics may not carry); never set in an eval
_EXTRA = [x.split(":", 2) for x in os.environ.get("R2G_TOOL_EXTRA_ITEMS", "").split(";") if x.count(":") == 2]


def tool_variants(prefix: str = "r2g_", sep: str = "__") -> dict[str, dict[str, Any]]:
    out = {}
    for task, cfg in TOOL_ENVS.items():
        cfg = dict(cfg, items={**cfg["items"], **{i: ph for t, i, ph in _EXTRA if t == task}})
        base = {"task": task, "object": None, "legacy": False, "offered": True, "not_offered_reason": "",
                "tool": dict(tool=cfg["tool"], arm=cfg["arm"], phrase=cfg["tool_phrase"])}
        out[f"{prefix}{task}{sep}tool_pick"] = dict(base, variant="tool_pick", prompt=tool_pick_prompt(cfg), budget=450,
                                                    title=f"{task}: pick up the {cfg['tool_phrase']}")
        for item, it in cfg["items"].items():
            phrase, members = (it, [item]) if isinstance(it, str) else (it["phrase"], list(it["members"]))
            out[f"{prefix}{task}{sep}tool_{item}"] = dict(base, variant="tool_use", object=item, members=members,
                                                          prompt=tool_use_prompt(cfg, phrase, len(members) > 1), budget=900,
                                                          title=f"{task}: {phrase} moved with the {cfg['tool_phrase']}")
    return out


def with_tool_envs(variants_fn):
    """Wrap real2gym_catalog.variants so its listing includes the tool envs (every registry reads it)."""
    @functools.wraps(variants_fn)
    def variants(*args, **kwargs):
        out = variants_fn(*args, **kwargs)
        out.update(tool_variants())
        return out
    return variants


def is_tool_variant(variant: str | None) -> bool:
    return variant in ("tool_pick", "tool_use")


class FixedPromptRuntime:
    """Task runtime that shows one fixed instruction for the whole episode (the tool envs)."""

    def __init__(self, prompt: str) -> None:
        self._prompt = prompt

    def bind(self, model, data) -> None:
        pass

    def after_reset(self, model, data, *, randomization=None) -> None:
        pass

    def before_step(self, model, data) -> None:
        pass

    def after_step(self, model, data) -> None:
        pass

    def observation_prompt(self) -> str:
        return self._prompt

    def debug_state(self) -> dict:
        return {"prompt": self._prompt}


__all__ = ["TOOL_ENVS", "FixedPromptRuntime", "is_tool_variant", "tool_members", "tool_pick_prompt", "tool_use_prompt", "tool_variants", "with_tool_envs"]
