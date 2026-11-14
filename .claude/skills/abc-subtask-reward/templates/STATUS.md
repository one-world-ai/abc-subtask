# <task> — subtask rewards ledger

short name: `<short>` · template(s): <A/B/…> · seeds: SEED_BASE = <n> · baseline: <exp12 x/10 or none>
status: <QUEUED | IN PROGRESS (phase N) | DONE | PARTIAL (stages …) | STOPPED>

## Phases
| phase | gate | result | numbers | evidence | date |
|---|---|---|---|---|---|
| 0 scene + scorer | G0 | | resets 0/20, probes …/… | | |
| 1 stage table | G1 | | approved by … / auto | design.md#stages | |
| 2 code + tests | G2 | | tests …/… | tests/test_<short>_subtasks.py | |
| 3 starts | G3 | | per stage valid/100, s/start, slip % | outputs/…/starts | |
| 4 scripted | G4/G5 | | per stage solver +1 rate, solver − best hack | outputs/…/scripted/summary.json | |
| 5 report | – | | | design.md | |

## Stages
| stage | status | why (if stopped) |
|---|---|---|

## Decisions (made with or for the user)
-

## Repair log
| gate | round | what failed | what changed |
|---|---|---|---|
