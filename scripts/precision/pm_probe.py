"""Diagnose small moves with the shipped walking policy: command pulses from a settled stand, many envs at once.

One trial = settle at zero command, then a pulse (twist for T s, the way a stick tap reaches the policy), then zero
command until the duck has settled. Measured from the stance frame (feet midpoint + trunk heading) before and after:
the displacement forward / left / yaw in the start frame, the number of foot lift-offs (steps), the slip of planted
feet, how long the duck keeps moving after the command returns to zero.

  PYTHONPATH=src .venv-mjlab/bin/python scripts/precision/pm_probe.py --envs 16 --reps 3 --out OUT.json \
      --pulses "x:0.05:0.5,x:0.1:0.5,y:0.1:0.5,w:0.3:0.5"   [--train-mode] [--no-dr] [--no-noise] [--friction 0.7]
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
from pm_env import Sim, rel  # noqa: E402

AX = {"x": 0, "y": 1, "w": 2}
LIFT_MM = 5.0        # a foot more than this above its standing height = a step
PLANT_MM = 1.5       # a foot within this of its standing height = planted (carries load)


def parse_pulses(s):
    out = []
    for p in s.split(","):
        ax, v, T = p.split(":")
        out.append((ax, float(v), float(T)))
    return out


def run_trial(sim, ax, v, T, settle_before, settle_after, foot_z0):
    n = sim.n
    for _ in range(int(settle_before / sim.dt)):
        sim.step([0.0, 0.0, 0.0])
    s0 = sim.stance()
    tw = [0.0, 0.0, 0.0]
    tw[AX[ax]] = v
    lifted = torch.zeros(n, 2, dtype=torch.bool)
    steps = torch.zeros(n)
    slip = torch.zeros(n)
    last_move = torch.zeros(n)
    prev = sim.stance()
    prev_feet = sim.feet_xy()
    track = []
    ticks_on = int(round(T / sim.dt))
    total = ticks_on + int(settle_after / sim.dt)
    for k in range(total):
        sim.step(tw if k < ticks_on else [0.0, 0.0, 0.0])
        feet = sim.feet_xy()
        _, fz = sim.foot_speed()
        h = (fz - foot_z0) * 1000.0
        now_lift = h > LIFT_MM
        steps += (now_lift & ~lifted).sum(dim=1).float()
        lifted = now_lift
        planted = h < PLANT_MM
        slip += (torch.norm(feet - prev_feet, dim=-1) * planted.float()).sum(dim=1)
        prev_feet = feet
        st = sim.stance()
        moved = torch.norm(st[:, :2] - prev[:, :2], dim=1) > 2e-4      # > 0.2 mm per tick = 1 cm/s
        last_move = torch.where(moved, torch.full_like(last_move, (k + 1) * sim.dt), last_move)
        prev = st
        if k % 5 == 0:
            track.append(rel(st, s0)[:, :].tolist())
    d = rel(sim.stance(), s0)
    up = sim.upright()
    return {
        "fwd_mm": (d[:, 0] * 1000).tolist(), "left_mm": (d[:, 1] * 1000).tolist(), "yaw_deg": (d[:, 2] * 180 / math.pi).tolist(),
        "steps": steps.tolist(), "slip_mm": (slip * 1000).tolist(), "moving_until_s": last_move.tolist(),
        "pulse_s": T, "upright": up.tolist(), "track": track,
    }


def summary(rows, ax):
    f = np.array(sum((r["fwd_mm"] for r in rows), []))
    l = np.array(sum((r["left_mm"] for r in rows), []))
    y = np.array(sum((r["yaw_deg"] for r in rows), []))
    st = np.array(sum((r["steps"] for r in rows), []))
    sl = np.array(sum((r["slip_mm"] for r in rows), []))
    mv = np.array(sum((r["moving_until_s"] for r in rows), []))
    T = rows[0]["pulse_s"]
    return {"n": int(len(f)), "fwd_mean": f.mean(), "fwd_sd": f.std(), "left_mean": l.mean(), "left_sd": l.std(),
            "yaw_mean": y.mean(), "yaw_sd": y.std(), "steps_mean": st.mean(), "steps_sd": st.std(),
            "slip_mean": sl.mean(), "settle_after_stop_s": float(np.median(mv) - T), "zero_steps_frac": float((st == 0).mean()),
            "fwd_min": f.min(), "fwd_max": f.max(), "left_min": l.min(), "left_max": l.max(), "yaw_min": y.min(), "yaw_max": y.max()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--pulses", required=True)
    ap.add_argument("--settle-before", type=float, default=1.0)
    ap.add_argument("--settle-after", type=float, default=2.5)
    ap.add_argument("--train-mode", action="store_true", help="no robotd layer (no EMA, no low-pass, scale 1.0)")
    ap.add_argument("--no-dr", action="store_true")
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--friction", type=float, default=None)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kw = {}
    if a.policy:
        kw["policy"] = a.policy
    sim = Sim(n=a.envs, robot=not a.train_mode, dr=not a.no_dr, noise=not a.no_noise, friction=a.friction, seed=a.seed, **kw)
    for _ in range(100):
        sim.step([0.0, 0.0, 0.0])
    _, fz = sim.foot_speed()
    foot_z0 = fz.clone()
    results = {"args": vars(a), "pulses": []}
    t0 = time.time()
    for ax, v, T in parse_pulses(a.pulses):
        rows = []
        for r in range(a.reps):
            rows.append(run_trial(sim, ax, v, T, a.settle_before, a.settle_after, foot_z0))
        s = summary(rows, ax)
        results["pulses"].append({"axis": ax, "v": v, "T": T, "summary": s, "rows": rows})
        print(f"{ax} v={v:+.2f} T={T:.2f}s  n={s['n']}  fwd {s['fwd_mean']:+6.1f}±{s['fwd_sd']:4.1f} mm  left {s['left_mean']:+6.1f}±{s['left_sd']:4.1f} mm  "
              f"yaw {s['yaw_mean']:+5.1f}±{s['yaw_sd']:4.1f}°  steps {s['steps_mean']:.1f}±{s['steps_sd']:.1f} (none {100*s['zero_steps_frac']:.0f}%)  "
              f"slip {s['slip_mean']:.1f} mm  moving {s['settle_after_stop_s']:.2f}s after stop  [{time.time()-t0:.0f}s]", flush=True)
        Path(a.out).write_text(json.dumps(results, default=float))
    print("upright at the end:", int(sim.upright().sum()), "/", sim.n)


if __name__ == "__main__":
    main()
