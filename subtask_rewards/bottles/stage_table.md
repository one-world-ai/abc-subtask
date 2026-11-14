# bottles — stage table (APPROVED 2026-11-13)

User confirmed bottles next, then LEGO, and selected dropping from above the opening as in this draft. Release
outside the opening is rejected; success requires settling inside. Lowering before release is not required.

One episode = one **target bottle** (the not-yet-binned bottle nearest a gripper at the start). Template A.
Bin frame = the tub's own frame (follows bin pose); "over the opening" = CoM radius ≤ r_top − 2 cm, CoM above rim.

| stage | prompt (live) | in it when (= start) | goal (= next start) | failures (−1) | progress (span) |
|---|---|---|---|---|---|
| reach | move a gripper to the {nth} bottle | none of the below | grasp site ≤ 5 cm from its CoM, or a finger on it | pushed > 5 cm while not held; bin moved/tipped | −dist (0.30 m) |
| pick | grasp the bottle and lift it | near, or grasped but lifted < 3 cm | both fingers of one gripper, lifted ≥ 3 cm | pushed; bin moved/tipped | 0.25·fingers + 0.5·lift frac (1.0) |
| carry | carry the bottle over the bin | grasped and lifted, not over the opening | over the opening while held | dropped outside the bin (falls back to reach); bin moved/tipped; knocked a binned bottle out | −3-D gap to the opening box (0.40 m) |
| drop | drop the bottle in the bin | over the opening, held/touched | stock in-bin for the target **and** no finger on it **and** at rest | lands outside (back to reach); bin moved/tipped; knocked a binned bottle out | −radius past the opening + 0.05 if released (0.10 m) |

Thresholds: bin moved = > 5 cm xy or tilted > 20°; push 5 cm (bottles roll: 3 cm was for put_relative's objects);
hold 5 actions; hysteresis near 5/7 cm, opening 2 cm.

Starts: reach = reset; pick = open gripper around the resting bottle (top-down, as put_relative);
carry = held 5–15 cm above its spot (top-down hold works on the table: put_relative 12/12);
drop = held over the opening — **chained from the carry solver's end states** (synthesized leaning holds: 4/20).

Scripted checks: reach / move away / idle; grasp-lift / close only / push / idle; carry / wrong way / open early /
idle; open over the bin / open beside the bin / hold still / idle.
