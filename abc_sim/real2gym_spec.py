"""Instruction = enforced spec for the Real2Gym subtask envs (r2g_<task>__pick_<class>, r2g_<task>__obj_<class>).

An RL env is (start state, instruction, check). The instruction of a subtask env is rendered at every reset from the
settled layout, and the check (task_eval/real2gym_tasks.py) enforces exactly its clauses:

  spec = {object, arm, targets, qualifier, position, lift_clause, lift_free, instruction, valid, reason, ...}
  (``resolve_spec``; SPEC_VERSION 2)

Envs are per describable object CLASS (look-alikes: the phrase table's group, or the same phrase), named by the slug
of the group (green_capped_glass_bottle; the phrase agents write the group as a short slug of the phrase). The old per-instance env names (..__pick_chang_1) are legacy aliases
with the instance fixed. Objects are never referred to by an internal name or index: the instruction describes the
target as  [qualifier] <feature phrase> [position clause],  all true of the layout at reset.

  object     the target instance. Class env: drawn uniformly, by an RNG derived from the reset seed (and the class;
             not the layout's RNG), among the members that are valid (not ruled out by the phrase table, not buried
             under a look-alike; obj: goal not already met at reset), among those no other object rests on if there
             are such (an object lying on it would have to come off before a pick can succeed; interlocked parts, a
             plug and its socket, do not count).
  arm        the gripper that has to do it. By the target's lateral position at reset (centre of mass):
             y >= +ARM_Y_BAND_M -> left, y <= -ARM_Y_BAND_M -> right, in between -> the phrase table's demo_arm, and
             if that is unknown the nearer arm base. None where the goal is not pinch-and-place (phrase table:
             goals.<o>.pinch = false): then no arm is named or enforced.
  qualifier  the word that tells the target from ALL its look-alikes in the scene, only where it is unambiguous from
             above:
               top                     the look-alikes are stacked / nested (xy within STACK_XY_M, different height),
                                       every one is in the target's stack and the target is the highest
               front / rear            (1 look-alike) the target is the unique extreme along depth x
               left / right            ... along y
               frontmost / rearmost / leftmost / rightmost   (2+ look-alikes)
             "extreme" needs both: the centres >= TWIN_SEP_M apart along that axis AND the whole extents (world
             boxes of all the object's geoms) separated by >= EXT_GAP_M along it: centres alone are ambiguous for
             long objects (two pens side by side, one 6 cm further back, still overlap in depth). Of the directions
             that qualify the one with the LARGEST extent gap is used (depth first on a tie). front = nearest the
             robot = smaller x = lower in the top camera image; left = the robot's left = +y = image left.
  position   where the target is: {kind, clause, side | landmark}
               landmark   the target starts on / in a container or fixture L of the scene (L carries it: a contact
                          that pushes it up, directly or through the objects under it in a stack; or it touches L with
                          its centre inside L's box) that has a clause in the phrase table's landmarks map ("in the
                          canvas bag"). Landmarks of the table that are not bodies of the scene are ignored; a
                          container without a clause does not name the position (the table side does).
               table      otherwise by y: >= +SIDE_Y_M "on the left side of the table", <= -SIDE_Y_M "on the right
                          side of the table", else "in the middle of the table"
             With look-alikes and no qualifier the clause narrows the accepted targets to the look-alikes it does
             not clearly rule out (same landmark; table side: y not more than SIDE_MARGIN_M outside the region: a
             viewer could not tell an object 1 cm past the boundary from one 1 cm inside it).
  targets    the objects the check accepts: the target, and, when no qualifier fits, every look-alike the instruction
             is equally true of (all of them without a position clause, else those in the same place).
  lift_clause, lift_free, lift_out_of   pick envs: "lift it off the <what it rests on at reset>". The check always
             enforces "lifted off" (lift_free): during the hold the target touches nothing but the designated gripper:
             not the table, no fixed scene geometry, not the other arm, no other object (unless the designated
             gripper pinches that one too). The height of the centre of mass alone would pass a plate tilted up on
             one edge, or a bowl lifted together with the bowl under it. The clause names what the target rests on
             at reset, first match of: the phrase table's clause about a support ("lift it out of the green bowl") if
             that support carries the target; the landmark of the position clause if it carries the target directly
             or holds it ("lift it out of the blue basket", "lift it off the wire dish rack"); the fixed scene (the
             table's clause about the table, else "lift it off the table"); the free objects under it ("lift it off
             the black dress shoes"); the landmark through a stack; "lift it up" (counted by the spec table tool).
             "out of <container>" is enforced as well (lift_out_of): the target's lowest point above the container's
             rim (the highest point of its collision geometry, mesh vertices; not a bag's visual handles). A plug plugged into a socket: "lift it out of the <socket>", enforced by
             touching nothing.
  valid      False (reason says why) when no true and enforceable sentence exists: the target (legacy env) or every
             member (class env) is buried under a look-alike, ruled out by the phrase table (pick_ok / obj_ok false,
             needs both grippers) or is already at its goal (obj). The check then never reports success.
  instruction  pick: "Move the <arm> arm to the [<qualifier>] <phrase> [<position>] and grasp it, then <lift clause>."
               obj:  "Move the <arm> arm to the [<qualifier>] <phrase> [<position>] and grasp it, then <goal clause>."
               no grasp-location words, no references to other stages.

Phrase table: abc_sim/real2gym_phrases.json, written by tools/real2gym_phrases_merge.py from the phrase agents' files
through ``normalize_table`` (one schema whatever the source file used). Per task:
  objects.<name>.{phrase (noun phrase, no position words; what the sim shows), group, title_noun, demo_arm, pick_ok,
                  obj_ok, pick_reason, obj_reason, label_mapping_ok, lift_clause, starts_on, label_phrase}
  goals.<name or *>.{phrase (imperative clause stating what the goal enforces), enforced[, pinch]}
  lift_clause (the task's default), landmarks.<scene object>.{phrase, clause ("in the canvas bag")}
Objects / tasks missing from it fall back to phrases built from the object names (phrase_source "name": such an
instruction contains an internal name and is flagged by tools/real2gym_spec_table.py) and the catalog's goal kinds.

Only json / re / zlib at import time (real2gym_catalog.variants() uses the text part when task_specs is imported).
"""

from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Any

SPEC_VERSION = 2
PHRASES_FILE = Path(__file__).resolve().parent / "real2gym_phrases.json"

ARM_BASE_XY = {"left": (0.253, 0.31), "right": (0.253, -0.31)}
ARM_Y_BAND_M = 0.04  # |y| below this: the object sits between the arms, the side does not decide the arm
TWIN_SEP_M = 0.05  # a qualifier needs the centres this far apart along its axis ...
EXT_GAP_M = 0.01  # ... and the whole extents separated by this much
STACK_XY_M = 0.04  # look-alikes with centres this close in xy and at different heights are stacked / nested
STACK_DZ_M = 0.005
SIDE_Y_M = 0.08  # table sides: left of +SIDE_Y_M, right of -SIDE_Y_M, the middle between
SIDE_MARGIN_M = 0.02  # look-alikes up to this far outside the named region are accepted as well
SIDE_TEXT = {"left": "on the left side of the table", "right": "on the right side of the table",
             "middle": "in the middle of the table"}
FIXED = "<fixed>"  # the fixed scene (table and everything welded to the world that is not a library object)
SUPPORT_NZ = 0.15  # a contact carries an object when its normal points this much upwards (nested baskets rest on
#                     slanted walls: every normal is far from vertical)
BOTH_KEYS = ("both_grippers", "needs_both_grippers", "needs_both", "two_arm", "two_arms", "bimanual", "two_grippers")
PINCH_KINDS = ("in", "off", "upright")  # catalog goal kinds reached by grasping the object and putting it somewhere
_PREPOSITIONS = {"in", "on", "inside", "at", "against", "under", "beside", "next", "near", "behind", "atop", "within",
                 "plugged", "hanging", "leaning", "resting", "stacked", "over", "above", "between", "standing", "lying"}
_IN_NOUNS = {"bag", "basket", "bin", "box", "bowl", "tray", "drawer", "cabinet", "container", "case", "crate", "tin",
             "cup", "holder", "organizer", "pouch", "tub", "pot", "jar", "sink", "tote", "caddy", "bucket"}
_PLUG_NOUNS = {"socket", "port", "outlet", "switch", "strip"}  # landmarks things are plugged into
_TABLE_WORDS = r"\b(table|tabletop|countertop|counter|desk|worktop|floor)\b"
_OFF_WORDS = r"\b(off|out of|from)\b"


# ---------------------------------------------------------------------------------------------- phrase table
_PHRASES: dict[str, Any] | None = None


def name_phrase(name: str) -> str:
    """Fallback noun phrase from an object name: chang_1 -> "chang", object_plate_right -> "plate right"."""
    s = re.sub(r"^object_", "", name)
    s = re.sub(r"_(\d+|[a-d])$", "", s)
    s = re.sub(r"(?<=[a-z])\d+(?=_|$)", "", s)
    return s.replace("_", " ").strip() or name


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _clean_phrase(text: str) -> str:
    s = str(text).strip().rstrip(".").strip()
    return re.sub(r"^(the|a|an)\s+", "", s, flags=re.I)


def _clean_clause(text: Any) -> str | None:
    if not isinstance(text, str):
        return None
    s = text.strip().rstrip(".").strip()
    if not s:
        return None
    return (s[:1].lower() + s[1:]) if not s[:2].isupper() else s


def _landmark_entry(e: Any) -> dict[str, str] | None:
    """A landmarks value (noun phrase, clause, or {phrase, clause, prep}) -> {phrase, clause}."""
    prep = None
    if isinstance(e, dict):
        clause, prep = e.get("clause"), e.get("prep")
        noun = e.get("phrase") or e.get("text")
        if isinstance(clause, str) and clause.strip():
            got = _landmark_entry(clause)
            if got and isinstance(noun, str) and noun.strip():
                got["phrase"] = _clean_phrase(noun)
            return got
        e = noun
    if not isinstance(e, str) or not e.strip():
        return None
    s = e.strip().rstrip(".").strip()
    words = s.split()
    if words[0].lower() in _PREPOSITIONS:
        i = 2 if len(words) > 1 and words[1].lower() in ("to", "into", "of") else 1
        return {"phrase": _clean_phrase(" ".join(words[i:])), "clause": s}
    noun = _clean_phrase(s)
    if prep is None:
        words = re.findall(r"[a-z]+", noun.lower())
        prep = "plugged into" if any(w in _PLUG_NOUNS for w in words) else "in" if any(w in _IN_NOUNS for w in words) else "on"
    return {"phrase": noun, "clause": f"{prep} the {noun}"}


def normalize_task(raw: dict[str, Any], prefer: str = "sim") -> dict[str, Any]:
    """One task entry of a phrase file -> the schema of the module docstring. The phrase agents' files differ
    (lift_clause / pick_phrase / support; sim_phrase and sim_group where the sim asset does not look like the label
    says; group ""): ``prefer`` "sim" describes what the sim shows, "label" keeps the label's words. Idempotent."""
    if raw.get("_norm"):
        return raw

    def lift(e, default):
        for k in ("lift_clause", "pick_phrase"):
            if k in e:
                return _clean_clause(e[k])
        if "support" in e:
            s = e["support"]
            return f"lift it off the {_clean_phrase(s)}" if isinstance(s, str) and s.strip() else None
        return default

    task_lift = lift(raw, "lift it off the table")
    out: dict[str, Any] = {"_norm": 1, "lift_clause": task_lift, "objects": {}, "goals": {}, "landmarks": {}}
    for name, e in (raw.get("objects") or {}).items():
        if not isinstance(e, dict) or not e.get("phrase"):
            continue
        label = _clean_phrase(e["phrase"])
        phrase = _clean_phrase(e["sim_phrase"]) if prefer == "sim" and e.get("sim_phrase") else label
        group = (e.get("sim_group") if prefer == "sim" else None) or e.get("group") or slugify(phrase)
        arm = str(e.get("demo_arm") or "").strip().lower()
        o = {"phrase": phrase, "group": str(group), "title_noun": _clean_phrase(e.get("title_noun") or phrase),
             "demo_arm": arm if arm in ("left", "right") else None,
             "pick_ok": e.get("pick_ok", True) is not False, "obj_ok": e.get("obj_ok", True) is not False,
             "pick_reason": str(e.get("pick_reason") or ""), "obj_reason": str(e.get("obj_reason") or ""),
             "label_mapping_ok": e.get("label_mapping_ok"), "lift_clause": lift(e, task_lift),
             "label_phrase": _clean_phrase(e.get("label_phrase") or label)}
        if any(bool(e.get(k)) for k in BOTH_KEYS):
            o["both_grippers"] = True
        if isinstance(e.get("starts_on"), str) and e["starts_on"].strip():
            o["starts_on"] = e["starts_on"].strip()
        for k in ("starts_on_object", "covered_by_object", "fine_phrase"):  # carried for the record, not read
            if isinstance(e.get(k), str) and e[k].strip():
                o[k] = e[k].strip()
        if isinstance(e.get("confusable_with"), list) and e["confusable_with"]:
            o["confusable_with"] = [str(k) for k in e["confusable_with"]]  # other classes a viewer may mix it up with
        out["objects"][name] = o
    for name, e in (raw.get("goals") or {}).items():
        if isinstance(e, str):
            e = {"phrase": e}
        if not isinstance(e, dict) or not _clean_clause(e.get("phrase")):
            continue
        g = {"phrase": _clean_clause(e["phrase"]), "enforced": e.get("enforced")}
        if e.get("pinch", e.get("pinch_and_place")) is False:
            g["pinch"] = False
        out["goals"][name] = g
    lm = raw.get("landmarks")
    for name, e in (lm.items() if isinstance(lm, dict) else []):
        entry = _landmark_entry(e)
        if entry:
            out["landmarks"][name] = entry
    return out


def normalize_table(raw: dict[str, Any], prefer: str = "sim") -> dict[str, Any]:
    return {t: (v if (t.startswith("_") or not isinstance(v, dict)) else normalize_task(v, prefer)) for t, v in raw.items()}


def load_phrases(path: str | Path | None = None, *, reload: bool = False) -> dict[str, Any]:
    """The phrase table {task: {objects, goals, lift_clause, landmarks}} (normalised); {} if the file is missing."""
    global _PHRASES
    if path is not None:
        try:
            return normalize_table(json.loads(Path(path).read_text()))
        except (OSError, ValueError):
            return {}
    if _PHRASES is None or reload:
        try:
            _PHRASES = normalize_table(json.loads(PHRASES_FILE.read_text()))
        except (OSError, ValueError):
            _PHRASES = {}
    return _PHRASES


_EMPTY = {"_norm": 1, "lift_clause": "lift it off the table", "objects": {}, "goals": {}, "landmarks": {}}


def _task(task: str, phrases: dict[str, Any] | None) -> dict[str, Any]:
    t = (phrases if phrases is not None else load_phrases()).get(task)
    if not isinstance(t, dict):
        return _EMPTY
    return t if t.get("_norm") else normalize_task(t)


def object_info(task: str, name: str, phrases: dict[str, Any] | None = None) -> dict[str, Any]:
    """Phrase-table entry of one object: phrase, group (its class), slug (of the group: the env name), phrase_slug,
    title_noun, demo_arm, pick_ok, obj_ok, both, source."""
    e = _task(task, phrases)["objects"].get(name)
    if not e:
        p = name_phrase(name)
        return {"phrase": p, "group": "name:" + p, "slug": slugify(p), "phrase_slug": slugify(p), "title_noun": p,
                "demo_arm": None, "pick_ok": True, "obj_ok": True, "both": False, "label_ok": None, "source": "name",
                "why": {}, "starts_on": None, "lift_clause": "lift it off the table", "confusable_with": []}
    why = {k: e[k] for k in ("pick_reason", "obj_reason") if e.get(k)}
    return {"phrase": e["phrase"], "group": e["group"], "slug": slugify(e["group"]) or slugify(e["phrase"]),
            "phrase_slug": slugify(e["phrase"]), "title_noun": e["title_noun"],
            "demo_arm": e["demo_arm"], "pick_ok": e["pick_ok"], "obj_ok": e["obj_ok"], "both": bool(e.get("both_grippers")),
            "label_ok": e.get("label_mapping_ok"), "source": "table", "why": why, "starts_on": e.get("starts_on"),
            "lift_clause": e.get("lift_clause"), "confusable_with": list(e.get("confusable_with") or [])}


def look_alike(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Two objects (object_info) the feature phrase does not tell apart: the same group or the same phrase."""
    return a["group"] == b["group"] or a["phrase_slug"] == b["phrase_slug"]


def classes(task: str, names: list[str], phrases: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The objects ``names`` (in order) sorted into classes of look-alikes: [{slug, phrase, members, source}]."""
    infos = {n: object_info(task, n, phrases) for n in names}
    out: list[dict[str, Any]] = []
    for n in names:
        home = [c for c in out if any(look_alike(infos[n], infos[m]) for m in c["members"])]
        if not home:
            out.append({"slug": infos[n]["slug"], "phrase": infos[n]["phrase"], "members": [n],
                        "source": infos[n]["source"]})
            continue
        home[0]["members"].append(n)
        for c in home[1:]:  # n links two classes: merge
            home[0]["members"] += c["members"]
            out.remove(c)
    seen: dict[str, int] = {}
    for c in out:  # distinct classes with one slug cannot happen (same slug = look-alikes); kept as a guard
        k = seen.get(c["slug"], 0)
        seen[c["slug"]] = k + 1
        if k:
            c["slug"] += "_" + "abcdefghij"[min(k, 9)]
    return out


def landmark_phrases(task: str, phrases: dict[str, Any] | None = None) -> dict[str, str]:
    """Container / fixture name -> position clause ("in the canvas bag") from the phrase table's landmarks map."""
    return {n: e["clause"] for n, e in _task(task, phrases)["landmarks"].items()}


def landmark_candidates(task: str, phrases: dict[str, Any] | None = None) -> list[str]:
    """Scene objects that can hold others: the phrase table's landmarks, plus the catalog's containers and fixtures."""
    from .real2gym_catalog import TASKS
    out = list(_task(task, phrases)["landmarks"])
    cfg = TASKS[task]
    for g in cfg["goals"]:
        out += [c for c in g.get("containers", []) if c not in out]
    for c in cfg.get("constraints", []):
        if c.get("container") and c["container"] not in out:
            out.append(c["container"])
    out += [c for c in cfg.get("fixed", []) if c not in out]
    return out


def lift_kind(clause: str | None) -> str:
    """What a lift clause claims: "up" (height only), "table" (off the table), "support" (off something else)."""
    if not clause or not re.search(_OFF_WORDS, clause):
        return "up"
    return "table" if re.search(_TABLE_WORDS, clause) else "support"


def _catalog_goal(task: str, name: str) -> dict[str, Any] | None:
    from .real2gym_catalog import TASKS
    for g in TASKS[task]["goals"]:
        if name in g.get("objects", []) and g["kind"] in PINCH_KINDS:
            return g
    return None


def _tilt_word(tilt) -> str:
    if not tilt:
        return ""
    lo, hi = float(tilt[0]), float(tilt[1])
    if lo <= 0.0 and hi <= 20.0:
        return "upright"
    if lo >= 150.0:
        return "upside down"
    if lo >= 45.0 and hi <= 135.0:
        return "on its side"
    return ""


def _fallback_goal_clause(task: str, g: dict[str, Any], phrases) -> str:
    """A clause from the catalog goal alone (used until the phrase table has one)."""
    lm = _task(task, phrases)["landmarks"]
    conts = [lm[c]["phrase"] if c in lm else object_info(task, c, phrases)["phrase"] for c in g.get("containers", [])]
    cont = " or the ".join(conts)
    word = _tilt_word(g.get("tilt"))
    if g["kind"] == "in":
        return f"place it {word + ' ' if word else ''}in the {cont}"
    if g["kind"] == "off":
        return f"take it off the {cont}" + (f" and set it down {word}" if word else " and set it down")
    return f"set it down {word or 'upright'}" + (" on the table" if g.get("on_table") else "")


def goal_info(task: str, name: str, phrases: dict[str, Any] | None = None) -> dict[str, Any]:
    """Goal clause of an obj env: {clause, pinch, enforced, source}. pinch False: no arm named or enforced."""
    goals = _task(task, phrases)["goals"]
    e = goals.get(name) or goals.get("*")
    g = _catalog_goal(task, name)
    if e:
        return {"clause": e["phrase"], "pinch": e.get("pinch", True) is not False, "enforced": e.get("enforced"),
                "source": "table"}
    return {"clause": _fallback_goal_clause(task, g, phrases) if g else "bring it to its goal", "pinch": True,
            "enforced": None, "source": "catalog"}


# ---------------------------------------------------------------------------------------------- text
def _what(qualifier: str | None, phrase: str, position: str | None, article: str = "the") -> str:
    return f"{article} {qualifier + ' ' if qualifier else ''}{phrase}{' ' + position if position else ''}"


def render_pick(arm: str | None, qualifier: str | None, phrase: str, lift_clause: str | None,
                position: str | None = None, article: str = "the") -> str:
    what = _what(qualifier, phrase, position, article)
    lift = lift_clause or "lift it up"
    if arm is None:
        return f"Grasp {what} and {lift}."
    return f"Move the {arm} arm to {what} and grasp it, then {lift}."


def render_obj(arm: str | None, qualifier: str | None, phrase: str, clause: str, pinch: bool = True,
               position: str | None = None, article: str = "the") -> str:
    what = _what(qualifier, phrase, position, article)
    if not pinch or arm is None:  # neutral imperative: "{obj}" in the clause is the object, else "it" refers to it
        if "{obj}" in clause:
            s = clause.replace("{obj}", what)
            return s[:1].upper() + s[1:] + "."
        if pinch:
            return f"Grasp {what} and {clause}."
        return f"Take {what} and {clause}." if re.search(r"\bit\b", clause) else clause[:1].upper() + clause[1:] + "."
    return f"Move the {arm} arm to {what} and grasp it, then {clause}."


def _article(phrase: str, members: int) -> str:
    return "the" if members <= 1 else ("an" if phrase[:1].lower() in "aeiou" else "a")


def fallback_prompt(task: str, variant: str, members: list[str], phrases: dict[str, Any] | None = None) -> str:
    """Layout-free instruction of a subtask env (no arm, qualifier or position): the registry prompt before a reset."""
    info = object_info(task, members[0], phrases)
    art = _article(info["phrase"], len(members))
    if variant == "pick":
        return f"Grasp {art} {info['phrase']} and lift it."
    g = goal_info(task, members[0], phrases)
    return render_obj(None, None, info["phrase"], g["clause"], g["pinch"], article=art)


def env_title(task: str, variant: str, members: list[str], phrases: dict[str, Any] | None = None) -> str:
    """Human-readable name of a subtask env for result pages: "Pick up a green-capped glass bottle"."""
    info = object_info(task, members[0], phrases)
    noun = info["title_noun"]
    what = f"{'an' if noun[:1].lower() in 'aeiou' else 'a'} {noun}"
    if variant == "pick":
        return f"Pick up {what}"
    clause = goal_info(task, members[0], phrases)["clause"]
    if "{obj}" in clause:
        s = clause.replace("{obj}", what)
    elif re.search(r"\bit\b", clause):
        s = f"Grasp {what} and {clause}"
    else:
        s = f"{what}: {clause}"
    return s[:1].upper() + s[1:]


def static_validity(task: str, variant: str, name: str, phrases: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Layout-independent part of spec.valid: (offered, reason)."""
    info = object_info(task, name, phrases)
    if variant == "pick":
        if info["both"]:
            return False, "needs both grippers (phrase table)"
        if not info["pick_ok"]:
            w = info["why"].get("pick_reason")
            return False, "pick_ok false in the phrase table" + (f" [{w}]" if w else "")
    if variant == "obj":
        if not info["obj_ok"]:
            w = info["why"].get("obj_reason")
            return False, "obj_ok false in the phrase table" + (f" [{w}]" if w else "")
        if _catalog_goal(task, name) is None:
            return False, "no pinch-and-place goal for this object in the catalog"
    return True, ""


# ---------------------------------------------------------------------------------------------- geometry
def choose_arm(y: float, x: float, demo_arm: str | None) -> tuple[str, str]:
    """(arm, rule): by side, else the demo's arm, else the nearer arm base."""
    if y >= ARM_Y_BAND_M:
        return "left", "side"
    if y <= -ARM_Y_BAND_M:
        return "right", "side"
    if demo_arm in ("left", "right"):
        return demo_arm, "demo"
    d = {a: (x - bx) ** 2 + (y - by) ** 2 for a, (bx, by) in ARM_BASE_XY.items()}
    return ("left" if d["left"] <= d["right"] else "right"), "nearer"


def distinguish(target: str, twins: list[str], pos: dict[str, Any], box: dict[str, Any] | None = None) -> dict[str, Any]:
    """How the instruction tells ``target`` from its look-alikes ``twins`` (pos: name -> centre xyz; box: name ->
    (lo xyz, hi xyz) of its whole extent; without boxes the centres alone decide).
    -> {qualifier, margin_m (centres), gap_m (extents), buried_by, stack}; qualifier None = not told apart."""
    out: dict[str, Any] = {"qualifier": None, "margin_m": None, "gap_m": None, "buried_by": None, "stack": []}
    if not twins:
        return out
    x, y, z = (float(v) for v in pos[target][:3])
    stack = [n for n in twins if ((float(pos[n][0]) - x) ** 2 + (float(pos[n][1]) - y) ** 2) ** 0.5 < STACK_XY_M
             and abs(float(pos[n][2]) - z) > STACK_DZ_M]
    out["stack"] = stack
    if stack:
        above = [n for n in stack if float(pos[n][2]) > z]
        if above:
            out["buried_by"] = max(above, key=lambda n: float(pos[n][2]))
            return out
        if len(stack) == len(twins):
            out["qualifier"] = "top"
            out["margin_m"] = round(min(z - float(pos[n][2]) for n in stack), 4)
        return out  # top of its stack, but look-alikes elsewhere: not told apart
    many = len(twins) >= 2
    cand = []
    for word, ax, sign in (("front", 0, -1), ("rear", 0, 1), ("left", 1, 1), ("right", 1, -1)):
        margin = min(sign * (float(pos[target][ax]) - float(pos[n][ax])) for n in twins)
        if box is None:
            gap = margin
        elif sign > 0:  # the target beyond every look-alike: its low side past their high sides
            gap = min(float(box[target][0][ax]) - float(box[n][1][ax]) for n in twins)
        else:
            gap = min(float(box[n][0][ax]) - float(box[target][1][ax]) for n in twins)
        if margin >= TWIN_SEP_M and (box is None or gap >= EXT_GAP_M):
            cand.append((gap, margin, word))
    if cand:
        gap, margin, word = max(cand, key=lambda c: c[0])  # the first of equal gaps: depth before sideways
        out["qualifier"] = {"front": "frontmost", "rear": "rearmost", "left": "leftmost", "right": "rightmost"}[word] if many else word
        out["margin_m"], out["gap_m"] = round(margin, 4), round(gap, 4)
    return out


def table_side(y: float) -> str:
    """left (y >= +SIDE_Y_M) / right (y <= -SIDE_Y_M) / middle of the table."""
    if y >= SIDE_Y_M:
        return "left"
    if y <= -SIDE_Y_M:
        return "right"
    return "middle"


def side_membership(side: str, y: float) -> bool | None:
    """Is an object at y on ``side``? True clearly, False clearly not, None: a viewer could not tell."""
    lo, hi = SIDE_Y_M - SIDE_MARGIN_M, SIDE_Y_M + SIDE_MARGIN_M
    v = y if side == "left" else -y if side == "right" else None
    if v is not None:
        return True if v >= hi else (False if v < lo else None)
    return True if abs(y) <= lo else (False if abs(y) > hi else None)


def scene_geometry(model, data, names: list[str]) -> dict[str, Any]:
    """What the layout in ``data`` is, per library object in ``names`` that is in the model:
      touch[n]     the others it touches (names, FIXED for the fixed scene)
      supports[n]  those of them that carry it (a contact whose normal pushes it upwards)
      box[n]       (lo, hi) of its whole extent in the world: the corners of every geom's box, visual ones included
      body[n]      its body id (-1: not in the model)
    """
    import mujoco
    import numpy as np

    parent, weld = np.asarray(model.body_parentid), np.asarray(model.body_weldid)
    bid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names}
    owner_of_body: dict[int, str | None] = {b: n for n, b in bid.items() if b >= 0}
    for b in range(1, model.nbody):  # bodies come parents first
        if b not in owner_of_body:
            p = int(parent[b])
            owner_of_body[b] = owner_of_body.get(p) if p else None
    gb = np.asarray(model.geom_bodyid)
    owner: list[str | None] = []
    for g in range(model.ngeom):
        b = int(gb[g])
        o = owner_of_body.get(b)
        owner.append(o if o is not None else (FIXED if int(weld[b]) == 0 else None))
    touch: dict[str, set] = {n: set() for n, b in bid.items() if b >= 0}
    supports: dict[str, set] = {n: set() for n in touch}
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist > 1e-3:
            continue
        o1, o2 = owner[int(c.geom1)], owner[int(c.geom2)]
        if o1 == o2:
            continue
        nz = float(c.frame[2])  # the normal points from geom1 to geom2
        if o1 in touch and o2 is not None:
            touch[o1].add(o2)
            if -nz >= SUPPORT_NZ:
                supports[o1].add(o2)
        if o2 in touch and o1 is not None:
            touch[o2].add(o1)
            if nz >= SUPPORT_NZ:
                supports[o2].add(o1)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    lo = {n: np.full(3, np.inf) for n in touch}
    hi = {n: np.full(3, -np.inf) for n in touch}
    aabb = np.asarray(model.geom_aabb).reshape(-1, 6)
    for g in range(model.ngeom):
        o = owner[g]
        if o not in lo:
            continue
        pts = (corners * aabb[g, 3:] + aabb[g, :3]) @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        lo[o], hi[o] = np.minimum(lo[o], pts.min(0)), np.maximum(hi[o], pts.max(0))
    return {"touch": touch, "supports": supports, "box": {n: (lo[n], hi[n]) for n in lo}, "body": bid}


def _inside_xy(data, cont_body: int, extent, p) -> bool:
    """Is point p above / inside the container's footprint box (its body frame; any height above its floor)?"""
    loc = data.xmat[cont_body].reshape(3, 3).T @ (p - data.xpos[cont_body])
    return bool(extent[0] <= loc[0] <= extent[1] and extent[2] <= loc[1] <= extent[3] and loc[2] >= extent[4] - 0.01)


def choose_instance(valid: list[str], seed: int | None, task: str, slug: str) -> str:
    """The class env's target for this reset: uniform over ``valid``, from the reset seed and the class."""
    if len(valid) == 1:
        return valid[0]
    import numpy as np

    key = zlib.crc32(f"{task}/{slug}".encode())
    rng = np.random.default_rng() if seed is None else np.random.default_rng([int(seed) % (2 ** 32), key])
    return valid[int(rng.integers(len(valid)))]


# ---------------------------------------------------------------------------------------------- spec
def resolve_spec(model, data, task: str, variant: str, obj: str | None = None, *, members: list[str] | None = None,
                 group: str | None = None, seed: int | None = None, names: list[str] | None = None,
                 phrases: dict[str, Any] | None = None, goal_met: dict[str, bool] | None = None,
                 extents: dict[str, Any] | None = None) -> dict[str, Any]:
    """The spec of a subtask env for the layout in ``data`` (the settled reset state). ``obj``: the instance is fixed
    (legacy env); else it is drawn among ``members`` (class env ``group``) with ``seed``. ``goal_met``: per goal
    object whether its goal holds in this state (obj envs: such a member is not a target)."""
    import numpy as np

    if names is None or extents is None:
        from .randomization.tasks.real2gym_tasks import load_library
        lib = load_library(task)["objects"]
        names = names or [o["name"] for o in lib]
        extents = extents or {o["name"]: o["extent"] for o in lib}
    geo = scene_geometry(model, data, names)
    bid, touch, supports, box = geo["body"], geo["touch"], geo["supports"], geo["box"]
    present = [n for n in names if bid[n] >= 0]
    pos = {n: np.asarray(data.xipos[bid[n]], float).copy() for n in present}
    infos = {n: object_info(task, n, phrases) for n in present}
    members = [obj] if obj is not None else list(members or [])
    for m in members:
        infos.setdefault(m, object_info(task, m, phrases))
    slug = group or infos[members[0]]["slug"]
    spec: dict[str, Any] = {"spec_version": SPEC_VERSION, "task": task, "variant": variant, "group": slug,
                            "members": list(members), "legacy": obj is not None, "object": members[0], "arm": None,
                            "targets": [members[0]], "qualifier": None, "position": None, "instruction": "",
                            "valid": True, "reason": ""}

    def twins_of(n):
        return [k for k in present if k != n and look_alike(infos[n], infos[k])]

    def cover_of(n):  # objects resting on n (n carries them: a shoe on a shoe, brushes in a holder, a nested bowl),
        # not counting one that carries n as well (interlocked: a plug in its socket pushes the socket up and back)
        return sorted(k for k in present if k != n and n in supports.get(k, ()) and k not in supports.get(n, ()))

    why: dict[str, list[str]] = {}
    for m in members:
        r = []
        ok, w = static_validity(task, variant, m, phrases)
        if not ok:
            r.append(w)
        if m not in pos:
            r.append("not in the scene")
        else:
            if distinguish(m, twins_of(m), pos)["buried_by"]:
                r.append("buried under a look-alike")
            if variant == "obj" and goal_met and goal_met.get(m):
                r.append("goal already met at reset")
        why[m] = r
    valid = [m for m in members if not why[m]]
    spec["candidates"] = list(valid)
    if valid:  # members another object rests on (a sneaker lying across two shoes) only if every valid one has one
        free = [m for m in valid if not cover_of(m)]
        target = choose_instance(free or valid, seed, task, slug)
    else:
        target = next((m for m in members if m in pos), members[0])
        spec["valid"] = False
        spec["reason"] = "; ".join(why[target]) if obj is not None else \
            "no valid member: " + "; ".join(f"{m}: {', '.join(why[m])}" for m in members)
    info = infos[target]
    spec.update(object=target, targets=[target], phrase=info["phrase"], phrase_source=info["source"])
    if target not in pos:
        spec["instruction"] = fallback_prompt(task, variant, members, phrases)
        return spec
    twins = twins_of(target)
    d = distinguish(target, twins, pos, box)
    x, y, z = (float(v) for v in pos[target])
    # where it is: on / in a container or fixture with a clause in the phrase table, else the side of the table.
    # Candidates are bodies of this scene only (a landmark of the table that is not in the scene is ignored).
    cands = [c for c in landmark_candidates(task, phrases) if c in pos and c != target and c not in twins]
    named = {c: t for c, t in landmark_phrases(task, phrases).items() if c in cands}

    def carriers(n):  # everything that carries n, nearest first: direct supports, then what carries those (a stack)
        order, seen, frontier = [], {n}, [n]
        while frontier:
            nxt = []
            for k0 in frontier:
                for k in sorted(supports.get(k0, ()), key=str):
                    if k not in seen:
                        seen.add(k)
                        order.append(k)
                        nxt.append(k)
            frontier = nxt
        return order

    def landmark_of(n, with_clause=True):
        """The nearest landmark that carries n, else one it touches with its centre inside the landmark's box."""
        ok = named if with_clause else cands
        for c in carriers(n):
            if c in ok:
                return c
        for c in cands:
            if c in ok and c != n and c in touch[n] and _inside_xy(data, bid[c], extents[c], pos[n]):
                return c
        return None

    lm = landmark_of(target)
    position: dict[str, Any] = {"kind": "landmark" if lm else "table", "clause": None}
    if lm:
        position["landmark"] = lm
        position["clause"] = named[lm]
    else:
        position["side"] = table_side(y)
        position["clause"] = SIDE_TEXT[position["side"]]
        unnamed = landmark_of(target, with_clause=False)
        if unnamed:  # in / on a container the phrase table has no clause for: the table side names the place
            position["unnamed_landmark"] = unnamed
    if info.get("starts_on"):  # what the phrase table expects; the layout decides
        position["starts_on_table"] = info["starts_on"]
    same_place = list(twins)  # look-alikes the position clause does not clearly rule out
    if twins:
        if lm:
            same_place = [n for n in twins if landmark_of(n) == lm]
        else:  # the side of the table: every look-alike not clearly on another side (in a container or not)
            same_place = [n for n in twins if side_membership(position["side"], float(pos[n][1])) is not False]
    spec["position"] = position
    spec["qualifier"] = d["qualifier"]
    spec["targets"] = [target] + ([] if d["qualifier"] else sorted(same_place))
    spec["twins"] = sorted(twins)
    if d["qualifier"]:
        spec["margin_m"], spec["gap_m"] = d["margin_m"], d["gap_m"]
    if [k for k in info["confusable_with"] if k in pos]:  # other classes a viewer may mix it up with (not accepted)
        spec["confusable_with"] = [k for k in info["confusable_with"] if k in pos]
    spec["xyz"] = [round(x, 4), round(y, 4), round(z, 4)]
    cover = cover_of(target)
    if cover:
        spec["covered_by"] = cover
    arm, rule = choose_arm(y, x, info["demo_arm"])
    spec["arm"], spec["arm_rule"] = arm, rule
    if variant == "pick":
        sup = supports[target]
        clause, out_of, rule = _lift_clause(task, phrases, info["lift_clause"], sup, lm, named, touch[target],
                                            carriers(target), cands)
        if rule != "table clause" and info["lift_clause"]:
            spec["lift_clause_table"] = info["lift_clause"]  # the phrase table's clause, not used at this reset
        spec["lift_clause"], spec["lift_rule"] = clause, rule
        spec["supports"] = sorted(sup)
        spec["lift_free"] = True  # always: the target ends up touching nothing but the designated gripper
        spec["lift_out_of"] = out_of  # "out of <container>": its lowest point above the container's highest point
        spec["instruction"] = render_pick(arm, d["qualifier"], info["phrase"], clause, position["clause"])
    else:
        g = goal_info(task, target, phrases)
        if not g["pinch"]:
            spec["arm"], spec["arm_rule"] = None, "none"
        spec["goal_clause"], spec["goal_source"] = g["clause"], g["source"]
        spec["instruction"] = render_obj(spec["arm"], d["qualifier"], info["phrase"], g["clause"], g["pinch"],
                                         position["clause"])
    return spec


_INTO = ("in ", "inside ", "within ", "into ", "plugged ")


def _lift_clause(task, phrases, table_clause, sup, lm, named, touching, carriers, cands):
    """The pick env's lift clause for what the target rests on at reset -> (clause, out_of containers, rule).
    Module docstring (lift_clause): first match of table clause about a support / landmark / fixed scene / free
    objects under it / landmark through a stack / "lift it up"."""
    lms = _task(task, phrases)["landmarks"]
    direct = set(sup)

    def noun(k):  # what the instruction calls a support
        return lms[k]["phrase"] if k in lms else object_info(task, k, phrases)["phrase"]

    def from_landmark(k):
        if named[k].lower().startswith(_INTO):
            return f"lift it out of the {lms[k]['phrase']}", ([k] if not named[k].lower().startswith("plugged") else [])
        return f"lift it off the {lms[k]['phrase']}", []

    kind = lift_kind(table_clause)
    if kind == "support":  # "lift it out of the green bowl": true if the support it names carries the target
        low = table_clause.lower()
        hit = [k for k in sorted(direct - {FIXED}, key=str) if noun(k).lower() in low]
        if hit:
            return table_clause, (hit if re.search(r"\bout of\b", low) else []), "table clause"
    if lm and (lm in direct or lm in touching):
        c, out = from_landmark(lm)
        return c, out, "landmark"
    if FIXED in direct:
        return (table_clause, [], "table clause") if kind == "table" else ("lift it off the table", [], "fixed scene")
    free = [k for k in sorted(direct - {FIXED}, key=str)]
    if free:
        nouns = []
        for k in free:
            if noun(k) not in nouns:
                nouns.append(noun(k))
        return "lift it off the " + " and the ".join(nouns), [], "objects under it"
    if lm:
        c, out = from_landmark(lm)
        return c, out, "landmark through a stack"
    return "lift it up", [], "no support"


def spec_metrics(spec: dict[str, Any] | None) -> dict[str, Any]:
    """The spec as flat evaluator metrics (strings / scalars: TaskEvalResult keeps strings as they are)."""
    if not spec:
        return {"spec_version": SPEC_VERSION, "spec_valid": False, "spec_object": "", "spec_arm": "", "spec_targets": "",
                "spec_qualifier": "", "spec_position": "", "spec_instruction": "", "spec_reason": "no spec"}
    return {"spec_version": int(spec.get("spec_version", SPEC_VERSION)), "spec_valid": bool(spec.get("valid")),
            "spec_object": spec.get("object") or "", "spec_arm": spec.get("arm") or "",
            "spec_targets": ",".join(spec.get("targets", [])), "spec_qualifier": spec.get("qualifier") or "",
            "spec_position": (spec.get("position") or {}).get("clause") or "",
            "spec_instruction": spec.get("instruction", ""), "spec_reason": spec.get("reason", "")}


__all__ = ["SPEC_VERSION", "PHRASES_FILE", "FIXED", "load_phrases", "normalize_table", "normalize_task", "object_info",
           "look_alike", "classes", "goal_info", "landmark_phrases", "landmark_candidates", "lift_kind", "render_pick",
           "render_obj", "fallback_prompt", "env_title", "static_validity", "choose_arm", "distinguish", "table_side",
           "side_membership", "scene_geometry", "choose_instance", "resolve_spec", "spec_metrics", "slugify"]
