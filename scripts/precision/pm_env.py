"""Precision mode: the shipped walking policy (velstand v8) in its exact training env, on the Mac CPU.

The env is Mjlab-VelStand-Flat-Backlash-MicroDuck (the v8 recipe on a flat floor) with every domain randomisation
kept (foot friction 0.7-1.3, servo gains, masses, CoM, joint friction, armature, encoder bias, IMU mounting, action
and observation delays, gear backlash) and observation noise on, so the spread between envs is the spread a real
duck sees between tries. Removed: pushes, prone/crouch spawns, fall terminations, timeouts, command resampling.
The caller writes the twist every tick (`Sim.step(twist)`), like robotd; head and body commands stay zero.

Robot-side layer (robotd semantics, duck-control + robotd/src/main.rs, `robot=True`): the twist goes through the
command EMA `cmd += alpha (target - cmd)` (cmd_alpha 0.2 at 50 Hz) before the policy sees it; the action is low-passed
(legs 0.7, head 0.5: robotd's walk-mode defaults, NOT in training since 939c29e) and scaled by `action_scale` (robot 0.9,
training 1.0) before it reaches the servos, while the policy's last-action observation stays the raw output (robotd
keeps `last_action = action` before scaling and filtering).

Measures: the stance frame = midpoint of the two foot sites and the trunk heading; foot slip = horizontal speed of a
foot while it carries load.
"""
import math
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT / "src"))

import mjlab_microduck.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

TASK = "Mjlab-VelStand-Flat-Backlash-MicroDuck"
VELSTAND = "/Users/remi/microduck/notes/precision-mode/policies/velstand.onnx"
DROP_EVENTS = ("push_robot", "random_prone_init", "topple_push")
DROP_CURRICULA = ("fell_over_disable", "prone_init_prob", "topple_push_range", "standing_envs")
DROP_TERMS = ("time_out", "fell_over", "fallen_too_long", "out_of_terrain_bounds")


def yaw_of(q):
    """Heading (rad) of wxyz quaternions, shape (n, 4)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Policy:
    """An ONNX 61 -> 14 actor, batched one env at a time (the exports have batch 1)."""

    def __init__(self, path):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name

    def __call__(self, obs):
        o = obs.numpy().astype(np.float32)
        return torch.as_tensor(np.concatenate([self.sess.run(None, {self.iname: o[i:i + 1]})[0] for i in range(len(o))]))


class Sim:
    def __init__(self, n=16, policy=VELSTAND, task=TASK, robot=True, seed=0, noise=True, dr=True, friction=None,
                 cmd_alpha=None, action_scale=None, lowpass=None):
        cfg = load_env_cfg(task)
        cfg.scene.num_envs = n
        cfg.seed = seed
        cfg.episode_length_s = 1.0e5
        for k in DROP_EVENTS:
            cfg.events.pop(k, None)
        for k in DROP_CURRICULA:
            cfg.curriculum.pop(k, None)
        for k in DROP_TERMS:
            cfg.terminations.pop(k, None)
        if not dr:
            for k in [k for k in cfg.events if k not in ("reset_base", "reset_robot_joints", "expand_bam_friction_fields",
                                                          "reset_action_history")]:
                cfg.events.pop(k)
            for k in [k for k, c in cfg.curriculum.items() if c.params.get("event_name") not in (None, *cfg.events)]:
                cfg.curriculum.pop(k)
        if friction is not None and "foot_friction" in cfg.events:
            cfg.events["foot_friction"].params["ranges"] = (friction, friction)
        tw = cfg.commands["twist"]
        tw.resampling_time_range = (1.0e6, 1.0e6)
        tw.rel_standing_envs = 0.0
        tw.rel_heading_envs = 0.0
        tw.rel_turn_in_place_envs = 0.0
        if hasattr(tw, "init_velocity_prob"):
            tw.init_velocity_prob = 0.0
        if hasattr(tw, "rel_forward_envs"):
            tw.rel_forward_envs = 0.0
        for name in ("head_pose", "body_pose"):
            c = cfg.commands[name]
            c.resampling_time_range = (1.0e6, 1.0e6)
            if hasattr(c, "zero_command_prob"):
                c.zero_command_prob = 1.0
        if not noise:
            cfg.observations["actor"].enable_corruption = False
        # standing start on the flat floor, heading +x, no initial velocity
        rb = cfg.events["reset_base"].params
        rb["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": rb["pose_range"].get("z", (0.12, 0.13)),
                            "yaw": (0.0, 0.0)}
        rb["velocity_range"] = {}
        self.cfg = cfg
        self.env = ManagerBasedRlEnv(cfg, device="cpu")
        self.env.common_step_counter = 10 ** 9            # remaining curricula at their final stage
        self.n = n
        self.pi = Policy(policy) if isinstance(policy, str) else policy
        self.cmd_alpha = cmd_alpha if cmd_alpha is not None else (0.2 if robot else 1.0)
        self.action_scale = action_scale if action_scale is not None else (0.9 if robot else 1.0)
        lp = lowpass if lowpass is not None else ((0.7, 0.5) if robot else (1.0, 1.0))
        self.lp_alpha = torch.full((14,), lp[0])
        self.lp_alpha[5:9] = lp[1]                         # neck_pitch, head_pitch, head_yaw, head_roll
        self.robot = self.env.scene["robot"]
        self.twist_term = self.env.command_manager.get_term("twist")
        self.head_term = self.env.command_manager.get_term("head_pose")
        self.body_term = self.env.command_manager.get_term("body_pose")
        m = self.robot
        self.foot_sites = [m.site_names.index(s) for s in ("left_foot", "right_foot")]
        self.dt = self.env.step_dt
        self.reset()

    def reset(self):
        self.obs, _ = self.env.reset()
        self.ema = torch.zeros(self.n, 3)
        self.filt = torch.zeros(self.n, 14)
        self.t = 0
        self._zero_pose_cmds()
        self.obs = self.env.observation_manager.compute()
        self.obs = self.obs["actor"] if isinstance(self.obs, dict) else self.obs

    def _zero_pose_cmds(self):
        for term in (self.head_term, self.body_term):
            for attr in ("pose_command", "command_b", "_command"):
                if hasattr(term, attr) and isinstance(getattr(term, attr), torch.Tensor):
                    getattr(term, attr).zero_()

    def step(self, twist, policy=None, smooth=True):
        """One 50 Hz tick. twist: (3,) or (n, 3) target [vx, vy, wz]. smooth=True passes it through robotd's command
        EMA (the gait's path); a skill's command is fed raw (robotd builds a fresh command block for skills).
        policy: a different network for this tick (robotd switching to a skill and back)."""
        target = torch.as_tensor(twist, dtype=torch.float32).expand(self.n, 3)
        if smooth:
            self.ema += self.cmd_alpha * (target - self.ema)
        else:
            self.ema = target.clone()
        self.twist_term.vel_command_b[:] = self.ema
        self._zero_pose_cmds()
        self.obs[:, 48:51] = self.ema                      # the command the policy sees on this tick
        act = (policy or self.pi)(self.obs)
        self.filt = self.lp_alpha * act + (1 - self.lp_alpha) * self.filt
        obs, *_ = self.env.step(self.filt * self.action_scale)
        self.obs = obs["actor"].clone()
        self.obs[:, 34:48] = act                          # robotd observes the raw action
        self.t += 1

    # ---- measurements ----
    def feet_xy(self):
        p = self.robot.data.site_pos_w[:, self.foot_sites, :2] - self.env.scene.env_origins[:, None, :2]
        return p                                    # (n, 2, 2)

    def stance(self):
        """(n, 3): x, y of the feet midpoint, trunk heading."""
        f = self.feet_xy()
        mid = f.mean(dim=1)
        yaw = yaw_of(self.robot.data.root_link_quat_w)
        return torch.cat([mid, yaw[:, None]], dim=1)

    def trunk(self):
        p = self.robot.data.root_link_pos_w[:, :2] - self.env.scene.env_origins[:, :2]
        return torch.cat([p, yaw_of(self.robot.data.root_link_quat_w)[:, None]], dim=1)

    def foot_speed(self):
        """(n, 2) horizontal speed of each foot site, m/s, and (n, 2) foot height above the floor."""
        v = self.robot.data.site_lin_vel_w[:, self.foot_sites, :2] if hasattr(self.robot.data, "site_lin_vel_w") else None
        z = self.robot.data.site_pos_w[:, self.foot_sites, 2]
        return v, z

    def qpos(self, ids=None):
        """Full qpos of the given envs (numpy, (k, nq)), world frame minus the env origin — for replay renders."""
        q = self.env.sim.data.qpos
        q = (q if isinstance(q, torch.Tensor) else torch.as_tensor(np.asarray(q))).clone()
        q[:, :2] -= self.env.scene.env_origins[:, :2]
        return (q if ids is None else q[ids]).numpy()

    def upright(self):
        g = self.robot.data.projected_gravity_b
        return g[:, 2] < -0.8

    def joint_speed(self):
        return self.robot.data.joint_vel.abs().max(dim=1).values


def rel(s1, s0):
    """Displacement of stance s1 from s0 expressed in s0's frame: (n, 3) forward, left, yaw change."""
    d = s1[:, :2] - s0[:, :2]
    c, s = torch.cos(s0[:, 2]), torch.sin(s0[:, 2])
    fwd = c * d[:, 0] + s * d[:, 1]
    left = -s * d[:, 0] + c * d[:, 1]
    dyaw = torch.remainder(s1[:, 2] - s0[:, 2] + math.pi, 2 * math.pi) - math.pi
    return torch.stack([fwd, left, dyaw], dim=1)
