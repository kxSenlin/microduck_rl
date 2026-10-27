"""Nudge strategies on the shipped walking policy, measured like pm_probe: final stance displacement per nudge.

Strategies (one axis at a time, target `--target` in mm or degrees):
  pulse:  open loop, twist `v` on the axis for the time that should cover the target at the policy's measured speed;
  closed: closed loop on contact odometry: twist `v` on the axis until the odometry says the remaining distance is
          under `--lead` (the overshoot the policy adds after the stop), then zero, then wait for the duck to settle.
Odometry here = the robot's contact-odometry idea (no-slip anchor on the support foot, IMU heading): the trunk moves
with respect to the lower foot, re-anchored when the support changes. It reads the slip of the support foot as no
motion, as the robot's does. `--odom truth` uses the simulator's own pose instead (an upper bound).

  PYTHONPATH=src .venv-mjlab/bin/python scripts/precision/pm_nudge.py --strategy closed --axis x --target 20 \
      --v 0.15 --lead 5 --envs 16 --reps 3 --out OUT.json
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


class Odom:
    """Contact odometry, simplified: trunk xy relative to the support foot (the lower sole), heading from the trunk
    orientation (the IMU). Accumulates in the world frame; reset() zeroes it."""

    def __init__(self, sim):
        self.sim = sim
        self.reset()

    def _state(self):
        feet = self.sim.feet_xy()                                # (n, 2, 2)
        _, fz = self.sim.foot_speed()                            # (n, 2)
        sup = fz.argmin(dim=1)                                   # support foot = the lower one
        tr = self.sim.trunk()
        anchor = feet[torch.arange(self.sim.n), sup]
        return sup, tr[:, :2] - anchor, tr[:, 2]

    def reset(self):
        self.sup, self.vec, self.yaw0 = self._state()
        self.pos = torch.zeros(self.sim.n, 2)

    def update(self):
        sup, vec, yaw = self._state()
        same = sup == self.sup
        self.pos += torch.where(same[:, None], vec - self.vec, torch.zeros_like(vec))
        self.sup, self.vec, self.yaw = sup, vec, yaw

    def rel(self):
        """(n, 3) displacement in the start frame: forward, left, yaw change."""
        c, s = torch.cos(self.yaw0), torch.sin(self.yaw0)
        f = c * self.pos[:, 0] + s * self.pos[:, 1]
        l = -s * self.pos[:, 0] + c * self.pos[:, 1]
        dy = torch.remainder(self.yaw - self.yaw0 + math.pi, 2 * math.pi) - math.pi
        return torch.stack([f, l, dy], dim=1)


def nudge(sim, a, odom):
    """One nudge per env, in parallel. Returns final displacement in the start frame (stance), and stats."""
    i = AX[a.axis]
    unit = 1e-3 if a.axis != "w" else math.pi / 180
    target = a.target * unit
    lead = a.lead * unit
    sgn = 1.0 if target >= 0 else -1.0
    s0 = sim.stance()
    tr0 = sim.trunk()
    odom.reset()
    active = torch.ones(sim.n, dtype=torch.bool)
    t_stop = torch.zeros(sim.n)
    max_ticks = int(a.max_s / sim.dt)
    pulse_ticks = int(round(a.pulse_s / sim.dt)) if a.strategy == "pulse" else 0
    for k in range(max_ticks + int(a.settle / sim.dt)):
        if a.strategy == "pulse":
            on = torch.full((sim.n,), k < pulse_ticks)
        else:
            odom.update()
            if a.odom == "truth":
                done = sgn * (rel(sim.trunk(), tr0)[:, i]) >= abs(target) - lead
            else:
                done = sgn * odom.rel()[:, i] >= abs(target) - lead
            if k >= max_ticks:
                done = torch.ones_like(done)
            newly = active & done
            t_stop = torch.where(newly, torch.full_like(t_stop, k * sim.dt), t_stop)
            active &= ~done
            on = active
        tw = torch.zeros(sim.n, 3)
        tw[:, i] = torch.where(on, torch.full((sim.n,), sgn * a.v), torch.zeros(sim.n))
        sim.step(tw)
        if a.strategy == "pulse" and k > pulse_ticks + a.settle / sim.dt:
            break
    d = rel(sim.stance(), s0)
    return d, t_stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["pulse", "closed"], required=True)
    ap.add_argument("--axis", choices=list(AX), required=True)
    ap.add_argument("--target", type=float, required=True, help="mm (x, y) or degrees (w)")
    ap.add_argument("--v", type=float, required=True, help="twist magnitude on the axis (m/s or rad/s)")
    ap.add_argument("--pulse-s", type=float, default=0.5)
    ap.add_argument("--lead", type=float, default=0.0, help="stop this early (mm or degrees)")
    ap.add_argument("--odom", choices=["contact", "truth"], default="contact")
    ap.add_argument("--max-s", type=float, default=3.0)
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sim = Sim(n=a.envs, seed=a.seed)
    for _ in range(100):
        sim.step([0.0, 0.0, 0.0])
    odom = Odom(sim)
    rows = []
    t0 = time.time()
    for r in range(a.reps):
        for _ in range(50):
            sim.step([0.0, 0.0, 0.0])
        d, ts = nudge(sim, a, odom)
        rows.append({"fwd_mm": (d[:, 0] * 1000).tolist(), "left_mm": (d[:, 1] * 1000).tolist(),
                     "yaw_deg": (d[:, 2] * 180 / math.pi).tolist(), "t_stop": ts.tolist(), "upright": sim.upright().tolist()})
    f = np.array(sum((r["fwd_mm"] for r in rows), []))
    l = np.array(sum((r["left_mm"] for r in rows), []))
    y = np.array(sum((r["yaw_deg"] for r in rows), []))
    print(f"{a.strategy} {a.axis} target {a.target:+.0f} v={a.v} lead={a.lead} odom={a.odom}: n={len(f)} "
          f"fwd {f.mean():+.1f}±{f.std():.1f} mm  left {l.mean():+.1f}±{l.std():.1f} mm  yaw {y.mean():+.1f}±{y.std():.1f}°  "
          f"upright {sum(sum(r['upright']) for r in rows)}/{len(f)}  [{time.time() - t0:.0f}s]", flush=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, default=float))


if __name__ == "__main__":
    main()
