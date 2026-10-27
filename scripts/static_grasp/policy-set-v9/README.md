# Policy set v9, ready to tag (not published)

What `pollen-robotics/microduck-policies` v9 would be: v8 plus `static_ground_pick.onnx` (the static low grasp,
run 4 of `Mjlab-StaticGrasp-Hover-FromPick-MicroDuck`, = notes `candidate-r4-final.onnx`), which becomes the walking
ground pick, with its trained beak as the entry's `mouth`. `alpha_ground_pick.onnx` stays in the set as a plain file.
Every other entry is v8's, unchanged.

Publishing: upload `manifest.json` and `static_ground_pick.onnx` to the set's main and tag `v9`, before the daemon
whose pin says v9 is released. A board running an older daemon that moves to v9 (`robotctl policy update`) keeps the
old pick (its default names `alpha_ground_pick.onnx`) with the v8 timing, which v9 repeats.
