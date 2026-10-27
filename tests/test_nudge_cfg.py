"""Precision nudge: the cfg must match robotd's one-shot skill contract and price the right things."""
import math

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_nudge_env_cfg as nudge
from mjlab_microduck.tasks.microduck_nudge_env_cfg import make_microduck_nudge_env_cfg


def test_command_is_the_skill_timeline():
    cfg = make_microduck_nudge_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.NudgeCommandCfg)
    # robotd skill: `duration` = go_s with the constant command, then `unwind` zeros for unwind_s
    assert cmd.go_s == nudge.GO_S
    assert math.isclose(cfg.episode_length_s, nudge.PRE_S + nudge.GO_S + nudge.SETTLE_S)
    # never resampled mid-episode, never a standing or heading env
    assert cmd.resampling_time_range[0] > cfg.episode_length_s
    assert cmd.rel_standing_envs == 0.0 and cmd.rel_heading_envs == 0.0
    # one pure axis per episode, small moves, some stand-still episodes
    assert len(cmd.axis_probs) == 3 and math.isclose(sum(cmd.axis_probs), 1.0)
    lo, hi = cmd.dist_range_m
    assert 0.0 < lo < hi <= 0.04                           # centimetres, not a walk
    lo, hi = cmd.yaw_range_rad
    assert 0.0 < lo < hi <= math.radians(20.0)
    assert 0.0 < cmd.zero_prob < 0.5


def test_requests_stay_inside_the_walking_policys_speed_range():
    # the actor starts from velstand (trained on |vx| <= 0.4 m/s, |wz| <= 1.0 rad/s): the largest request must read
    # as a speed it knows
    cfg = make_microduck_nudge_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.dist_range_m[1] / microduck_mdp.NUDGE_UNIT_M <= 0.4
    assert cmd.yaw_range_rad[1] / microduck_mdp.NUDGE_UNIT_RAD <= 1.0


def test_walking_terms_are_gone_and_slip_is_always_priced():
    cfg = make_microduck_nudge_env_cfg()
    for name in nudge.DROP_REWARDS:
        assert name not in cfg.rewards, name
    slip = cfg.rewards["nudge_slip"]
    assert slip.weight < 0.0
    assert slip.params["command_threshold"] < 0.0          # active at zero command too


def test_reward_signs():
    cfg = make_microduck_nudge_env_cfg()
    for name in ("nudge_progress", "nudge_hold", "nudge_hold_fine", "nudge_feet_down"):
        assert cfg.rewards[name].weight > 0.0, name
    for name in ("nudge_still", "termination"):
        assert cfg.rewards[name].weight < 0.0, name
    # the hold terms only pay once the move should be over (after the go window)
    assert cfg.rewards["nudge_hold"].params["after_s"] > nudge.GO_S
    assert cfg.rewards["nudge_hold_fine"].params["std"] < cfg.rewards["nudge_hold"].params["std"]


def test_no_pushes_and_standing_spawns():
    cfg = make_microduck_nudge_env_cfg()
    assert "push_robot" not in cfg.events
    z = cfg.events["reset_base"].params["pose_range"]["z"]
    assert 0.11 < z[0] < z[1] < 0.125


def test_actor_obs_contract_is_untouched():
    cfg = make_microduck_nudge_env_cfg()
    actor = list(cfg.observations["actor"].terms)
    assert actor[-3:] == ["command", "head_command", "body_command"]
    assert "nudge_state" not in actor and "nudge_state" in cfg.observations["critic"].terms
