"""Precision nudge: one small, exact move along ONE axis, then stand still. Design and lessons:
docs/superpowers/specs/2026-10-27-precision-nudge-design.md.

Why (2026-10-27, Rémi): placing the duck before a grasp is "very random" with the walking policy. Measured on
velstand v8 in its own training env (scripts/precision/pm_probe.py, 32 tries per row, all DR + obs noise): a stick
push under ~0.2 m/s or 0.3 rad/s for half a second does NOTHING (dead zone: uniform command sampling almost never
trains tiny pure commands); a full tap (0.4 m/s, 0.3 s) gives 59 ± 21 mm forward with ±12° of random yaw and ±9 mm
sideways; closed loop on contact odometry asking 20 mm gives 55 ± 23 mm (it always takes >= 2 steps and keeps walking
~0.6 s after the stop).

Contract = robotd's one-shot skill (SkillDef): twist = the requested move in units (x, y: 1.0 = 2 cm; yaw:
1.0 = 10°) for `go_s` (the skill's `duration`), then zeros (`unwind` = [0, 0, 0] for `unwind_s`), then the walking
policy at zero command. Head and body slots stay zero. Deployment needs no daemon code: six (or twelve) skill
entries with constant commands, e.g. nudge_fwd = [0.75, 0, 0], duration 0.5, unwind_s 1.3.

Episode: PRE_S of zero command (the duck stands as the walking policy left it), the go window, then SETTLE_S of zeros.
The move is measured on the stance frame (feet midpoint, trunk heading) from the go onset.

Built on the velocity recipe (DR, obs noise, delays, backlash variant) with velstand's all-collisions model; the
actor starts from velstand v8 (NudgeFromVelstandRunner), the critic is fresh and sees the remaining error.
"""
import math
from copy import deepcopy
from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg
from mjlab_microduck.tasks.microduck_velstand_env_cfg import _collapse_curricula_to_final

PRE_S = 0.3          # standing before the go window (the start state the walking policy hands over)
GO_S = 0.5           # robotd skill `duration`: the twist carries the request this long
SETTLE_S = 1.3       # robotd skill `unwind_s`: zeros, finish the step and stand still
EPISODE_S = PRE_S + GO_S + SETTLE_S

AXIS_PROBS = (0.35, 0.30, 0.35)       # x, y, yaw
MAGNITUDE_RANGE = (0.25, 1.5)         # units: 0.5-3 cm, 2.5-15°
ZERO_PROB = 0.1                       # stand still the whole episode

# Rewards (per step, 50 Hz; times are from the go onset).
PROGRESS_WEIGHT = 4.0                 # potential: sums to 4 x (initial error - final error), error unit = 1 cm / 5°
HOLD_AFTER_S = 0.9                    # the move should be over by then (the go window is 0.5 s)
HOLD_WEIGHT, HOLD_STD = 1.5, 1.0      # broad: gradient from ~2 cm / 10° away
HOLD_FINE_WEIGHT, HOLD_FINE_STD = 1.5, 0.3   # sharp: 3 mm / 1.5°
STILL_AFTER_S = 1.2
STILL_WEIGHT = -0.5
FEET_DOWN_WEIGHT = 0.5
SLIP_WEIGHT = -5.0                    # planted feet must not slide (always on, not only while "walking")
TERMINATION_WEIGHT = -10.0
STAND_Z = (0.116, 0.119)              # spawn just above the measured standing trunk height (0.1158)

DROP_REWARDS = ("track_linear_velocity", "track_angular_velocity", "air_time", "foot_clearance",
                "foot_swing_height", "foot_slip", "head_pose_bias", "body_pose_tracking")
DROP_CURRICULA = ("standing_envs", "head_pose_bias_weight", "body_pose_range", "head_pose_range")


def make_microduck_nudge_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    # velstand's model and contact budget (the actor comes from velstand)
    cfg.scene.entities = {"robot": MICRODUCK_ALLCOLLISIONS_ROBOT_CFG}
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 200)

    # inherited curricula at their final stage (the actor was trained under them); drop the command ones
    for k in DROP_CURRICULA:
        cfg.curriculum.pop(k, None)
    _collapse_curricula_to_final(cfg)

    cfg.episode_length_s = EPISODE_S
    cfg.events.pop("push_robot", None)
    rb = cfg.events["reset_base"].params
    rb["pose_range"]["z"] = STAND_Z
    rb["velocity_range"] = {}

    # ── Command: the nudge in the twist slots; head and body zero (robotd feeds skills zeros there) ──
    old = cfg.commands["twist"]
    keep = {f.name for f in fields(microduck_mdp.NudgeCommandCfg)}
    cfg.commands["twist"] = microduck_mdp.NudgeCommandCfg(
        **{**{k: v for k, v in vars(old).items() if k in keep}, "class_type": microduck_mdp.NudgeCommand,
           "rel_standing_envs": 0.0, "rel_heading_envs": 0.0, "resampling_time_range": (1.0e6, 1.0e6)},
        pre_s=PRE_S, go_s=GO_S, axis_probs=AXIS_PROBS, magnitude_range=MAGNITUDE_RANGE, zero_prob=ZERO_PROB,
    )
    head = cfg.commands["head_pose"]
    head.ranges = tuple((-0.005, 0.005) for _ in head.ranges)      # alive, ~zero
    body = cfg.commands["body_pose"]
    body.ranges = tuple((-0.001, 0.001) for _ in body.ranges)

    # ── Critic: the remaining error and the time (privileged) ──
    cfg.observations["critic"].terms["nudge_state"] = ObservationTermCfg(
        func=microduck_mdp.nudge_critic_obs, params={"command_name": "twist"},
    )

    # ── Rewards ──
    for k in DROP_REWARDS:
        cfg.rewards.pop(k, None)
    cfg.rewards["nudge_progress"] = RewardTermCfg(
        func=microduck_mdp.nudge_progress, weight=PROGRESS_WEIGHT, params={"command_name": "twist"})
    cfg.rewards["nudge_hold"] = RewardTermCfg(
        func=microduck_mdp.nudge_hold, weight=HOLD_WEIGHT,
        params={"command_name": "twist", "std": HOLD_STD, "after_s": HOLD_AFTER_S})
    cfg.rewards["nudge_hold_fine"] = RewardTermCfg(
        func=microduck_mdp.nudge_hold, weight=HOLD_FINE_WEIGHT,
        params={"command_name": "twist", "std": HOLD_FINE_STD, "after_s": HOLD_AFTER_S})
    cfg.rewards["nudge_still"] = RewardTermCfg(
        func=microduck_mdp.nudge_still, weight=STILL_WEIGHT, params={"command_name": "twist", "after_s": STILL_AFTER_S})
    cfg.rewards["nudge_feet_down"] = RewardTermCfg(
        func=microduck_mdp.nudge_feet_down, weight=FEET_DOWN_WEIGHT,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist", "after_s": STILL_AFTER_S})
    cfg.rewards["nudge_slip"] = RewardTermCfg(
        func=mdp.feet_slip, weight=SLIP_WEIGHT,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist", "command_threshold": -1.0,
                "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))})
    cfg.rewards["termination"] = RewardTermCfg(func=envs_mdp.is_terminated, weight=TERMINATION_WEIGHT)
    cfg.rewards["nudge_metrics"] = RewardTermCfg(
        func=microduck_mdp.nudge_metrics, weight=0.0, params={"command_name": "twist"})
    # weight-0 terms are not logged at all: give the logger a vanishing weight
    cfg.rewards["nudge_metrics"].weight = 1.0e-9
    return cfg


MicroduckNudgeRlCfg = deepcopy(MicroduckRlCfg)
MicroduckNudgeRlCfg.experiment_name = "nudge"
MicroduckNudgeRlCfg.run_name = "nudge"
MicroduckNudgeRlCfg.max_iterations = 1500
