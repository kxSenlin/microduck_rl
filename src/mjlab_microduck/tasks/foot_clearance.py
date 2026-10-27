"""Left-foot ↔ right-foot clearance from the 10 leg joint angles (torch, batched, differentiable).

Why (branch standup_foot_catch, 2026-10-27): on the real robot the v8 get-up from the
back sometimes stalls because the swinging foot catches the other foot. In sim it never
does: the swing passes the stance foot with p50 4-9 mm of mesh clearance (rise battery +
CPU mj_geomDistance replay), which sim tolerates and hardware slop (backlash, low-kp
tracking error, compliance) does not. The tight swing comes from the stand teacher
(alpha_stand: p50 2-6 mm) through BC. Measuring it needs no simulator state: the relative
pose of the two feet depends only on the leg joint angles, so the clearance of a JOINT
TARGET (default + action, action scale 1.0) is a pure function of the action — usable both
as a reward on the measured joints and as a constraint on the BC teacher's labels.

Model: each foot = the union bounding box of its ankle-body collision hulls (ankle body
frame), placed by product-of-exponentials FK from the zero configuration of
robot_allcollisions.xml (backlash hinges at 0; identical kinematics on every model of the
family). Clearance = min over surface sample points of one box of the signed distance to
the other box, both directions. Box ⊇ hull, so the proxy reads ~3-5 mm BELOW the exact mesh
distance (calibrated on v8 rises: corr 0.97; mesh 5-10 mm → proxy p50 3.8; 10-20 → 7.3).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from mjlab_microduck.robot.microduck_constants import MICRODUCK_ALLCOLLISIONS_XML

# Servo-order indices (14-D action / joint layout) of each leg chain, root → tip.
LEFT_LEG = (0, 1, 2, 3, 4)
RIGHT_LEG = (9, 10, 11, 12, 13)
_CHAINS = {
    "left": (("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle"), "ankle_left"),
    "right": (("right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"), "ankle_right"),
}
_GRID = (7, 4, 4)  # surface sample grid per box (x, y, z): ≤ 11 mm spacing on a 54×34×41 mm box


def _box_surface(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    axes = [np.linspace(l, h, k) for l, h, k in zip(lo, hi, _GRID)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return g[(np.isclose(g, lo) | np.isclose(g, hi)).any(1)]


def _build() -> dict[str, dict[str, np.ndarray]]:
    m = mujoco.MjModel.from_xml_path(str(MICRODUCK_ALLCOLLISIONS_XML))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)  # qpos0: all joints 0, trunk at its XML pose
    trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    Rt, pt = d.xmat[trunk].reshape(3, 3), d.xpos[trunk]
    to_trunk = lambda v: (v - pt) @ Rt  # world → trunk frame (zero config)
    out = {}
    for side, (joints, body_name) in _CHAINS.items():
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints]
        body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        Rb, pb = d.xmat[body].reshape(3, 3), d.xpos[body]
        verts = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != body or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            verts.append(((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]) - pb) @ Rb)  # ankle body frame
        verts = np.concatenate(verts)
        lo, hi = verts.min(0), verts.max(0)
        out[side] = {
            "axis": np.stack([d.xaxis[j] @ Rt for j in jid]),          # (5, 3) trunk frame
            "anchor": np.stack([to_trunk(d.xanchor[j]) for j in jid]),  # (5, 3)
            "R0": Rt.T @ Rb, "p0": to_trunk(pb),                        # ankle frame at zero config
            "center": (lo + hi) / 2, "half": (hi - lo) / 2,             # box, ankle frame
            "points": _box_surface(lo, hi),                              # (P, 3) ankle frame
        }
    return out


_PARAMS = _build()
_CACHE: dict[torch.device, dict] = {}


def _params(device: torch.device) -> dict:
    if device not in _CACHE:  # first call may come from a reward under inference_mode: keep these autograd-usable
        with torch.inference_mode(False):
            _CACHE[device] = {s: {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in p.items()}
                              for s, p in _PARAMS.items()}
    return _CACHE[device]


def _rodrigues(axis: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """axis (3,), q (B,) → (B, 3, 3)."""
    k = axis.expand(q.shape[0], 3)
    K = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    s, c = torch.sin(q)[:, None, None], torch.cos(q)[:, None, None]
    return torch.eye(3, device=q.device, dtype=q.dtype) + s * K + (1 - c) * (K @ K)


def _ankle_pose(p: dict, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Product of exponentials: q (B, 5) → ankle frame (R (B,3,3), t (B,3)) in the trunk frame."""
    B = q.shape[0]
    R = torch.eye(3, device=q.device, dtype=q.dtype).expand(B, 3, 3)
    t = torch.zeros(B, 3, device=q.device, dtype=q.dtype)
    for i in range(5):
        Ri = _rodrigues(p["axis"][i], q[:, i])
        ti = p["anchor"][i] - (Ri @ p["anchor"][i])  # rotation about the axis through the anchor
        t = t + (R @ ti.unsqueeze(-1)).squeeze(-1)
        R = R @ Ri
    return R @ p["R0"], t + R @ p["p0"]


def _one_way(pa: dict, Ra, ta, pb: dict, Rb, tb) -> torch.Tensor:
    """min over box-A surface points of the signed distance to box B → (B,)."""
    w = torch.einsum("bij,pj->bpi", Ra, pa["points"]) + ta[:, None, :]       # trunk frame
    lb = torch.einsum("bji,bpj->bpi", Rb, w - tb[:, None, :])                 # B's ankle frame
    qd = (lb - pb["center"]).abs() - pb["half"]
    sd = qd.clamp(min=0).norm(dim=-1) + qd.max(dim=-1).values.clamp(max=0)
    return sd.min(dim=1).values


def foot_clearance(joint_pos: torch.Tensor) -> torch.Tensor:
    """Signed left/right foot-box clearance (m) for absolute servo joint angles (B, 14) → (B,).

    Negative = boxes interpenetrate. Differentiable w.r.t. ``joint_pos``.
    """
    p = _params(joint_pos.device)
    Rl, tl = _ankle_pose(p["left"], joint_pos[:, list(LEFT_LEG)])
    Rr, tr = _ankle_pose(p["right"], joint_pos[:, list(RIGHT_LEG)])
    return torch.minimum(_one_way(p["left"], Rl, tl, p["right"], Rr, tr), _one_way(p["right"], Rr, tr, p["left"], Rl, tl))


def lateral_gap(joint_pos: torch.Tensor) -> torch.Tensor:
    """Left foot-box centre minus right foot-box centre along the trunk y axis (m), (B,).
    Positive = feet in their natural left/right order; it goes negative when legs cross."""
    p = _params(joint_pos.device)
    Rl, tl = _ankle_pose(p["left"], joint_pos[:, list(LEFT_LEG)])
    Rr, tr = _ankle_pose(p["right"], joint_pos[:, list(RIGHT_LEG)])
    return (tl + Rl @ p["left"]["center"])[:, 1] - (tr + Rr @ p["right"]["center"])[:, 1]


def path_clearance(start: torch.Tensor, target: torch.Tensor, samples: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)) -> torch.Tensor:
    """min foot clearance along the straight joint-space path start → target (start excluded), (B,)."""
    return torch.stack([foot_clearance(start + s * (target - start)) for s in samples], 1).min(1).values


def project_min_clearance(
    joint_pos: torch.Tensor, min_clearance: float, joint_ids: tuple[int, ...],
    iters: int = 4, max_delta: float = 0.35, start: torch.Tensor | None = None,
) -> torch.Tensor:
    """Smallest move of ``joint_ids`` (servo order) of the absolute joint target
    ``joint_pos`` (B, 14) that brings its foot clearance up to ``min_clearance`` (m).

    With ``start`` (the current joints), the clearance is taken along the joint-space
    path start → target instead of at the target: the stand teacher's swing throws one
    leg ACROSS the body (left hip_roll target ≈ −1.08) and the feet meet on the way, not
    at the target pose, so a target-only constraint misses it.

    Gauss-Newton on the scalar clearance: q += deficit · ∇c / |∇c|², a few iterations,
    total change clamped to ±``max_delta`` rad per joint. Rows already clear are returned
    unchanged (bit-identical), so it is a pure, label-consistent function of its inputs.
    With ``start``, a row whose projection does not improve the path clearance is left
    unprojected; without it, a row whose projection lowers the left/right lateral gap (pushed a
    leg THROUGH the other, possible from an interpenetrating target) is left unprojected.
    """
    with torch.inference_mode(False):  # rollout-storage tensors are inference tensors
        return _project(joint_pos.detach().clone(), min_clearance, joint_ids, iters, max_delta,
                        None if start is None else start.detach().clone())


def _project(q0, min_clearance, joint_ids, iters, max_delta, start):
    ids = torch.as_tensor(joint_ids, device=q0.device, dtype=torch.long)
    clr = (lambda q: foot_clearance(q)) if start is None else (lambda q: path_clearance(start, q))
    q = q0.clone()
    for _ in range(iters):
        with torch.enable_grad():  # BC labels are computed under no_grad
            sub = q[:, ids].clone().requires_grad_(True)
            c = clr(q.index_copy(1, ids, sub))
            (g,) = torch.autograd.grad(c.sum(), sub)
        deficit = (min_clearance - c.detach()).clamp(min=0.0)
        if not (deficit > 0).any():
            break
        step = deficit[:, None] * g / (g.pow(2).sum(1, keepdim=True) + 1e-6)
        new = (sub.detach() + step).clamp(q0[:, ids] - max_delta, q0[:, ids] + max_delta)
        q = q.index_copy(1, ids, new)
    if start is not None:  # Gauss-Newton on a min of distances can step the wrong way from deep overlap
        keep = clr(q) > clr(q0)
        return torch.where(keep[:, None], q, q0)
    keep = lateral_gap(q) >= lateral_gap(q0) - 1e-6
    return torch.where(keep[:, None], q, q0)


SERVO_NAMES = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
               "neck_pitch", "head_pitch", "head_yaw", "head_roll",
               "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")


def home_servo_pos() -> tuple[float, ...]:
    """HOME_FRAME in servo order: action targets are HOME + action (scale 1.0)."""
    import re

    from mjlab_microduck.robot.microduck_constants import HOME_FRAME
    out = []
    for n in SERVO_NAMES:
        out.append(next(float(v) for pat, v in HOME_FRAME.joint_pos.items() if re.fullmatch(pat, n)))
    return tuple(out)
