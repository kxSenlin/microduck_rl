"""Record scenario rollouts for replay renders (scenekit format: qpos per frame in the standalone Microduck MJCF
order — free joint, then the 14 servos; on the backlash model each servo column is servo + its gear-play hinge, the
angle of the link).

  tap:    the walking policy, a stick tap (robotd path: EMA, low-pass, 0.9 scale): --twist vx,vy,wz for --tap-s
  nudge:  the walking policy standing, then the nudge skill(s) from --onnx with --units ax:u[,ax:u...] in sequence
          (units after cde7159: x, y metres / 0.05, yaw radians / 0.25; robotd action scale 0.9 + low-pass)
  PYTHONPATH=src .venv-mjlab/bin/python scripts/precision/pm_record.py tap --twist 0.4,0,0 --tap-s 0.3 --envs 6 --out X.npz
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pm_env import Policy, Sim, rel  # noqa: E402

SERVOS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch",
          "head_yaw", "head_roll", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
NAMES = ["trunk_base_freejoint"] + SERVOS
AX = {"x": 0, "y": 1, "w": 2}


def layout(sim):
    m = sim.env.sim.mj_model
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + n)
    free = m.jnt_qposadr[jid("trunk_base_freejoint")]
    cols = []
    for s in SERVOS:
        a = [m.jnt_qposadr[jid(s)]]
        b = jid(f"passive_{s}_backlash")
        if b >= 0:
            a.append(m.jnt_qposadr[b])
        cols.append(a)
    return free, cols


def frame(sim, free, cols):
    q = sim.qpos()
    out = np.zeros((q.shape[0], 7 + len(cols)))
    out[:, :7] = q[:, free:free + 7]
    for i, a in enumerate(cols):
        out[:, 7 + i] = q[:, a].sum(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tap", "nudge"])
    ap.add_argument("--twist", default="0.4,0,0")
    ap.add_argument("--tap-s", type=float, default=0.3)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--units", default="x:0.75")
    ap.add_argument("--go", type=float, default=0.5)
    ap.add_argument("--unwind", type=float, default=1.3)
    ap.add_argument("--gap", type=float, default=0.6, help="walking policy at zero between two nudges")
    ap.add_argument("--skill-scale", type=float, default=None, help="action scale while a nudge runs (default: robotd 0.9)")
    ap.add_argument("--before", type=float, default=1.0)
    ap.add_argument("--after", type=float, default=2.0)
    ap.add_argument("--envs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sim = Sim(n=a.envs, seed=a.seed)
    free, cols = layout(sim)
    for _ in range(100):
        sim.step([0.0, 0.0, 0.0])
    frames, events = [], []
    tick = lambda tw, **kw: (sim.step(tw, **kw), frames.append(frame(sim, free, cols)))
    for _ in range(int(a.before / sim.dt)):
        tick([0.0, 0.0, 0.0])
    s0 = sim.stance()
    if a.mode == "tap":
        tw = [float(v) for v in a.twist.split(",")]
        events.append({"t": len(frames) * sim.dt, "what": f"tap {a.twist} for {a.tap_s}s"})
        for k in range(int(round(a.tap_s / sim.dt))):
            tick(tw)
    else:
        nudge = Policy(a.onnx)
        walk_scale = sim.action_scale
        for item in a.units.split(","):
            ax, u = item.split(":")
            cmd = [0.0, 0.0, 0.0]
            cmd[AX[ax]] = float(u)
            events.append({"t": len(frames) * sim.dt, "what": f"nudge {ax} {u}"})
            sim.action_scale = a.skill_scale if a.skill_scale is not None else walk_scale
            for k in range(int(round((a.go + a.unwind) / sim.dt))):
                tick(cmd if k * sim.dt < a.go - 1e-6 else [0.0, 0.0, 0.0], policy=nudge, smooth=False)
            sim.action_scale = walk_scale
            for _ in range(int(a.gap / sim.dt)):
                tick([0.0, 0.0, 0.0])
    for _ in range(int(a.after / sim.dt)):
        tick([0.0, 0.0, 0.0])
    d = rel(sim.stance(), s0).numpy()
    q = np.stack(frames, axis=1)                     # (envs, T, 21)
    np.savez_compressed(a.out, qpos=q, joint_names=np.array(NAMES), fps=1.0 / sim.dt, robot="microduck",
                        events=json.dumps(events), final=d)
    print("final displacement per env (fwd mm, left mm, yaw deg):")
    for i, r in enumerate(d):
        print(f"  {i}: {1000 * r[0]:+6.1f} {1000 * r[1]:+6.1f} {np.degrees(r[2]):+6.1f}")


if __name__ == "__main__":
    main()
