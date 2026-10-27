"""A nudge ONNX in its own training env (Mjlab-Nudge[-Runtime]-Flat-Backlash, DR + noise), one episode per env:
final stance error grouped by the requested axis. Ground truth for the training metrics (no walking policy before or
after, unlike pm_skill_eval.py).
  PYTHONPATH=src .venv-mjlab/bin/python scripts/precision/pm_trainenv_eval.py NUDGE.onnx [--envs 48] [--task ...]"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pm_env import Policy  # noqa: E402

import mjlab_microduck.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("onnx")
ap.add_argument("--envs", type=int, default=48)
ap.add_argument("--task", default="Mjlab-Nudge-Flat-Backlash-MicroDuck")
ap.add_argument("--seed", type=int, default=5)
a = ap.parse_args()
cfg = load_env_cfg(a.task)
cfg.scene.num_envs = a.envs
cfg.seed = a.seed
env = ManagerBasedRlEnv(cfg, device="cpu")
pi = Policy(a.onnx)
obs, _ = env.reset()
term = env.command_manager.get_term("twist")
T = int(round(cfg.episode_length_s / env.step_dt))
target = None
for k in range(T - 1):
    obs, *_ = env.step(pi(obs["actor"]))
    if target is None and not term.start_pending.any():
        target = term.target.clone()
err = (term.target - term.moved()).numpy()
tgt = target.numpy()
moved = term.moved().numpy()
up = (env.scene["robot"].data.projected_gravity_b[:, 2] < -0.8).numpy()
axis = np.where(np.abs(tgt).sum(1) == 0, -1, np.abs(tgt).argmax(1))
for ax, name in ((0, "x"), (1, "y"), (2, "yaw"), (-1, "stand")):
    m = axis == ax
    if not m.any():
        continue
    e = err[m]
    print(f"{name:5s} n={m.sum():2d}  err fwd {1000*e[:,0].mean():+5.1f}±{1000*e[:,0].std():4.1f} mm  "
          f"left {1000*e[:,1].mean():+5.1f}±{1000*e[:,1].std():4.1f} mm  yaw {np.degrees(e[:,2]).mean():+5.1f}±{np.degrees(e[:,2]).std():4.1f}°  "
          f"|target| {1000*np.abs(tgt[m][:, :2]).sum(1).mean():.0f} mm / {np.degrees(np.abs(tgt[m][:, 2])).mean():.1f}°  upright {up[m].sum()}/{m.sum()}")
