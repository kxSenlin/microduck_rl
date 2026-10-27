"""Bake the safe variant's action clip into an exported nudge ONNX: actions -> Max(Min(actions, hi), lo), per joint, the
same bounds training used (mdp.action_clamp_bounds on the training model). robotd then cannot be asked for a target past
a joint range + margin, whatever the network does.
  PYTHONPATH=src .venv-mjlab/bin/python scripts/precision/pm_bake_clamp.py IN.onnx OUT.onnx [--margin 0.05]"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pm_env import Sim  # noqa: E402

from mjlab_microduck.tasks import mdp as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("inp")
ap.add_argument("out")
ap.add_argument("--margin", type=float, default=0.05)
a = ap.parse_args()
sim = Sim(n=1)
t = sim.env.action_manager.get_term("joint_pos")
lo, hi = M.action_clamp_bounds(sim.robot, t._target_ids, t._target_names, a.margin)
lo, hi = lo.numpy().astype(np.float32), hi.numpy().astype(np.float32)
m = onnx.load(a.inp)
g = m.graph
out_name = g.output[0].name
raw = out_name + "_unclamped"
for n in g.node:
    n.output[:] = [raw if o == out_name else o for o in n.output]
g.initializer.extend([numpy_helper.from_array(hi, "clamp_hi"), numpy_helper.from_array(lo, "clamp_lo")])
g.node.extend([helper.make_node("Min", [raw, "clamp_hi"], [raw + "_min"]),
               helper.make_node("Max", [raw + "_min", "clamp_lo"], [out_name])])
onnx.checker.check_model(m)
onnx.save(m, a.out)
print("bounds (rad, raw action):")
for n_, l_, h_ in zip(t._target_names, lo[0], hi[0]):
    print(f"  {n_:16s} {l_:+.2f} .. {h_:+.2f}")
print("wrote", a.out)
