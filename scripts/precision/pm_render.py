"""Replay recorded rollouts (pm_record.py npz: qpos (envs, T, 21)) as ducks side by side in one scene kit scene.

  ~/mujoco_scenes/.venv/bin/python pm_render.py LOG.npz OUT.mp4 [--spacing 0.3] [--preset sim_grid] [--size 1920x1080]
      [--az 205 --el -32 --dist 1.2] [--label "text"] [--slow 1.0]
Every duck starts on the same line (x = 0), facing +x, one per lane; the camera is fixed.
"""
import argparse
import subprocess
import sys

import mujoco
import numpy as np

sys.path.insert(0, "/Users/remi/mujoco_scenes")
from scenekit import camera, compose, render, replay  # noqa: E402


def annotate(img, a, sc, ys):
    """Title and lane labels (Pillow: this ffmpeg has no drawtext). Lane labels sit at the projected start line."""
    if not (a.labels or a.title):
        return img
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    W, H = im.size
    f_big = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", int(H * 0.04))
    f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", int(H * 0.028))
    if a.title:
        d.text((int(W * 0.025), int(H * 0.03)), a.title, font=f_big, fill=(255, 255, 255), stroke_width=3, stroke_fill=(10, 20, 30))
    if a.labels:
        labels = a.labels.split("|")
        n = len(labels)
        for i, lab in enumerate(labels):
            x = W * (i + 0.5) / n
            d.text((x, H * 0.93), lab, font=f, fill=(255, 255, 255), anchor="mm", stroke_width=3, stroke_fill=(10, 20, 30))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("out")
    ap.add_argument("--spacing", type=float, default=0.3)
    ap.add_argument("--preset", default="sim_grid")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--az", type=float, default=200.0)
    ap.add_argument("--el", type=float, default=-30.0)
    ap.add_argument("--dist", type=float, default=1.3)
    ap.add_argument("--cx", type=float, default=0.06, help="look-at x (m, forward of the start line)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--slow", type=float, default=1.0)
    ap.add_argument("--ssaa", type=float, default=1.5)
    ap.add_argument("--labels", default=None, help="one label per lane, | separated, drawn under the ducks")
    ap.add_argument("--title", default=None, help="a line of text at the top-left")
    a = ap.parse_args()
    z = np.load(a.log)
    q = z["qpos"]                                     # (envs, T, 21)
    names = [str(s) for s in z["joint_names"]]
    fps_log = float(z["fps"])
    n, T, _ = q.shape
    W, H = (int(v) for v in a.size.split("x"))
    ys = [(i - (n - 1) / 2) * a.spacing for i in range(n)]
    sc = compose.build(a.preset, robots=["microduck"] * n, places=[(0.0, y, 0.0) for y in ys],
                       size=(int(W * a.ssaa), int(H * a.ssaa)))
    maps = [replay._robot_qpos_map(sc, i, names) for i in range(n)]
    # each log starts near (0, 0) heading +x: put it on its lane
    t_log = np.arange(T) / fps_log
    t_out = np.arange(0, t_log[-1] * a.slow, 1.0 / a.fps) / a.slow
    idx = np.clip(np.round(t_out * fps_log).astype(int), 0, T - 1)
    rig = camera.Fixed(center=(a.cx, 0.0, 0.08), distance=a.dist, azimuth=a.az, elevation=a.el)
    with render.Renderer(sc, size=(W, H), ssaa=a.ssaa) as r, render.Video(a.out, fps=a.fps, size=(W, H)) as v:
        for k in idx:
            for i, (cols, adrs, _) in enumerate(maps):
                row = q[i, k].copy()
                row[1] += ys[i] - q[i, 0, 1]          # lane offset (keep the duck's own lateral motion)
                row[0] -= q[i, 0, 0]
                sc.data.qpos[adrs] = row[cols]
            mujoco.mj_forward(sc.model, sc.data)
            v.write(annotate(r.frame(rig(0.0, sc)), a, sc, ys))
    print("wrote", a.out, len(idx), "frames")


if __name__ == "__main__":
    main()
