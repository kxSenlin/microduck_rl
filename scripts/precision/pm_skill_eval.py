"""Evaluate a nudge policy the way robotd runs it, in velstand's own training env (all DR, obs noise, backlash).

Each trial: the walking policy (velstand v8) stands at zero command, the nudge skill runs (its constant command for
`--go` s, zeros for `--unwind` s, fed raw as robotd feeds skills), then the walking policy again at zero command for
`--after` s. The stance displacement from the skill start to the end is the nudge. Optionally records the qpos of the
first `--record` envs (npz) for a replay render.

  PYTHONPATH=src .venv-mjlab/bin/python scripts/precision/pm_skill_eval.py NUDGE.onnx --axis x --units 0.75 \
      --envs 16 --reps 3 --out OUT.json [--record 4 --npz OUT.npz] [--scale 1.0]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pm_env import Policy, Sim, rel  # noqa: E402

AX = {"x": 0, "y": 1, "w": 2}
UNIT = {"x": 20.0, "y": 20.0, "w": 10.0}       # mm, mm, degrees per command unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--axis", choices=list(AX), required=True)
    ap.add_argument("--units", type=float, required=True, help="signed command units (x, y: 1 = 2 cm; w: 1 = 10°)")
    ap.add_argument("--go", type=float, default=0.5)
    ap.add_argument("--unwind", type=float, default=1.3)
    ap.add_argument("--before", type=float, default=1.0)
    ap.add_argument("--after", type=float, default=1.5)
    ap.add_argument("--scale", type=float, default=None, help="action scale while the skill runs (robot default 0.9)")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--chain", action="store_true", help="reps follow each other without re-centring (repeated nudges)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--record", type=int, default=0)
    ap.add_argument("--npz", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sim = Sim(n=a.envs, seed=a.seed)
    nudge = Policy(a.onnx)
    walk_scale = sim.action_scale
    cmd = [0.0, 0.0, 0.0]
    cmd[AX[a.axis]] = a.units
    for _ in range(100):
        sim.step([0.0, 0.0, 0.0])
    rows, frames, marks = [], [], []
    rec = list(range(min(a.record, a.envs)))
    t0 = time.time()
    s_first = sim.stance()
    for r in range(a.reps):
        for _ in range(int(a.before / sim.dt)):
            sim.step([0.0, 0.0, 0.0])
            if rec: frames.append(sim.qpos(rec))
        s0 = sim.stance()
        marks.append(len(frames))
        sim.action_scale = a.scale if a.scale is not None else walk_scale
        for k in range(int(round((a.go + a.unwind) / sim.dt))):
            sim.step(cmd if k * sim.dt < a.go - 1e-6 else [0.0, 0.0, 0.0], policy=nudge, smooth=False)
            if rec: frames.append(sim.qpos(rec))
        sim.action_scale = walk_scale
        s_skill = sim.stance()
        for _ in range(int(a.after / sim.dt)):
            sim.step([0.0, 0.0, 0.0])
            if rec: frames.append(sim.qpos(rec))
        d = rel(sim.stance(), s0)
        dh = rel(sim.stance(), s_skill)
        rows.append({"fwd_mm": (d[:, 0] * 1000).tolist(), "left_mm": (d[:, 1] * 1000).tolist(),
                     "yaw_deg": (d[:, 2] * 180 / math.pi).tolist(),
                     "handback_mm": (torch.norm(dh[:, :2], dim=1) * 1000).tolist(),
                     "handback_deg": (dh[:, 2].abs() * 180 / math.pi).tolist(), "upright": sim.upright().tolist()})
    f = np.array(sum((r["fwd_mm"] for r in rows), []))
    l = np.array(sum((r["left_mm"] for r in rows), []))
    y = np.array(sum((r["yaw_deg"] for r in rows), []))
    hb = np.array(sum((r["handback_mm"] for r in rows), []))
    up = sum(sum(r["upright"]) for r in rows)
    tgt = a.units * UNIT[a.axis]
    print(f"nudge {a.axis} {a.units:+.2f} units (target {tgt:+.1f} {'deg' if a.axis == 'w' else 'mm'}): n={len(f)}  "
          f"fwd {f.mean():+.1f}±{f.std():.1f} mm  left {l.mean():+.1f}±{l.std():.1f} mm  yaw {y.mean():+.1f}±{y.std():.1f}°  "
          f"handback drift {hb.mean():.1f} mm  upright {up}/{len(f)}  [{time.time() - t0:.0f}s]", flush=True)
    out = {"args": vars(a), "rows": rows, "target": tgt}
    Path(a.out).write_text(json.dumps(out, default=float))
    if rec and a.npz:
        np.savez_compressed(a.npz, qpos=np.stack(frames), marks=np.array(marks), dt=sim.dt)


if __name__ == "__main__":
    main()
