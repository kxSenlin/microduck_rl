# Policy set v9, ready to tag (not published)

What `pollen-robotics/microduck-policies` v9 would be: v8 plus `static_ground_pick.onnx` (the static low grasp, run 4
of `Mjlab-StaticGrasp-Hover-FromPick-MicroDuck`, sha256 40f3b357...) as the walking ground pick. `alpha_ground_pick.onnx`
stays in the set as a plain file; every other entry is v8's. The beak is played by the daemon (microduck#481).

Try it on a dev board without publishing anything (beak included):

    scp static_ground_pick.onnx <duck>:
    ssh <duck> 'sudo robotctl update apply --ref static-grasp-default daemon && sudo robotctl policy load ground_pick ~/static_ground_pick.onnx'

Back: `sudo robotctl policy reset ground_pick && sudo robotctl update apply --ref main daemon`.
