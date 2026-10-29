"""What a duck sees, in the format `mediad` captures.

MuJoCo renders RGB; `mediad` pins its pipeline to UYVY because that is what `v4l2src` can drive at
full rate off the rkisp — so the conversion happens here, on the side that knows it is a simulator.

**Frames do not go down the JSON link.** A 640x360 UYVY frame is 460,800 bytes, and a camera is its
own TCP port carrying length-prefixed raw frames: four bytes of little-endian length, then the bytes,
forever. No handshake, because there is nothing to negotiate that both ends do not already have to
agree on to be useful.

**Opt in, per duck.** Rendering is the most expensive thing in the simulator by a wide margin — tens
of milliseconds per frame against a fraction of a millisecond to step physics. A camera is therefore
only created for ducks named in `--cameras`, and it only renders while somebody is reading: the
frame servers count their readers and a camera with none renders nothing.

## Local changes (this fork)

All additive and controlled by environment variables:

* `DUCK_SIM_CAMERA_WIDTH` / `DUCK_SIM_CAMERA_HEIGHT` — render size, default 640x360. The client asks
  for 1280x720 to match a real duck's default `720p30` rung; MuJoCo's offscreen framebuffer is grown
  to fit before the renderer is built.
* The renderer is created lazily on the first `render()`, so its OpenGL context belongs to the thread
  that will use it. Building it on the physics thread and rendering on a camera thread fails silently.
* Rendering is skipped while `FrameServer.readers == 0`, so a duck nobody is watching costs nothing.
* `DUCK_SIM_CAMERA_SPLIT=1` — time `mjr_render`, `mjr_readPixels` and the flip separately and print
  them every five seconds. The WSL D3D12 path spends its time somewhere inside `Renderer.render()`,
  and this is what tells the submission apart from the readback.
* `DUCK_SIM_CAMERA_SHADOW` / `DUCK_SIM_CAMERA_REFLECTION` / `DUCK_SIM_CAMERA_SKYBOX` /
  `DUCK_SIM_CAMERA_HAZE` — the camera scene's passes. Shadow and reflection default to **off**:
  measured on the apartment at 720p, they are 126 ms of the 171 ms frame, and dropping them takes
  the camera from 5.8 fps to 22.2 fps while the room still reads correctly. The extra passes are a
  mapping/teleop nicety, not a control input; `=1` turns any of them back on.
"""

from __future__ import annotations

import os
import socket
import socketserver
import struct
import threading
import time

import mujoco
import numpy as np
from PIL import Image

# 16:9. The default is what upstream renders; the client raises it to 720p through the environment.
WIDTH = int(os.environ.get("DUCK_SIM_CAMERA_WIDTH", "1280"))
HEIGHT = int(os.environ.get("DUCK_SIM_CAMERA_HEIGHT", "720"))

# The sensor's rate is 30, but a rendered frame is expensive and a duck being watched is usually not
# being raced. `duck-sim` passes its own `--camera-fps`; this is the fallback.
FPS = 30

# Per-frame diagnostics: where inside `Renderer.render()` the time went.
SPLIT = bool(int(os.environ.get("DUCK_SIM_CAMERA_SPLIT", "0")))

# Scene passes, for the camera only. The shadow and reflection passes are *off* by default in this
# fork: at 720p on this machine the apartment costs 171 ms a frame with them on and 45 ms without
# (measured with `render_bench.py`), and the room reads the same without a shadow map or a mirror
# floor. The env vars re-enable them for a fidelity check.
ENABLE_SHADOW = bool(int(os.environ.get("DUCK_SIM_CAMERA_SHADOW", "0")))
ENABLE_REFLECTION = bool(int(os.environ.get("DUCK_SIM_CAMERA_REFLECTION", "0")))
ENABLE_SKYBOX = bool(int(os.environ.get("DUCK_SIM_CAMERA_SKYBOX", "1")))
ENABLE_HAZE = bool(int(os.environ.get("DUCK_SIM_CAMERA_HAZE", "1")))

# BT.601, the same coefficients `duck_detect`'s one-pass sampler uses on the robot.
_Y = np.array([0.299, 0.587, 0.114])
_U = np.array([-0.168736, -0.331264, 0.5])
_V = np.array([0.5, -0.418688, -0.081312])


def to_uyvy(rgb: np.ndarray) -> bytes:
    """RGB to packed UYVY: `U Y0 V Y1` per pixel pair, chroma averaged across the pair.

    PIL converts RGB to YCbCr in C, which is 9 ms for a 720p frame against 41 ms for the same
    arithmetic in float32 numpy — and at 30 fps that difference is the whole frame budget. Chroma is
    still averaged across each pair rather than dropped, because a subsampler that takes the left
    pixel's chroma puts a half-pixel colour shift into every frame; the detector notices even when a
    person does not.
    """
    ycbcr = np.asarray(Image.fromarray(rgb).convert("YCbCr"), dtype=np.uint8)
    height, width = ycbcr.shape[:2]
    packed = np.empty((height, width // 2, 4), dtype=np.uint8)
    packed[:, :, 1] = ycbcr[:, 0::2, 0]
    packed[:, :, 3] = ycbcr[:, 1::2, 0]
    chroma_u = ycbcr[:, 0::2, 1].astype(np.uint16) + ycbcr[:, 1::2, 1]
    chroma_v = ycbcr[:, 0::2, 2].astype(np.uint16) + ycbcr[:, 1::2, 2]
    packed[:, :, 0] = (chroma_u // 2).astype(np.uint8)
    packed[:, :, 2] = (chroma_v // 2).astype(np.uint8)
    return packed.tobytes()


class Camera:
    """One duck's head camera.

    The renderer is not thread-safe and is expensive to build, so it is created once, on the thread
    that renders, and only that thread touches it. `render()` takes the world lock only to copy the
    scene; the draw itself touches no shared state.
    """

    def __init__(self, model: mujoco.MjModel, name: str, width: int = WIDTH, height: int = HEIGHT):
        self.camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if self.camera < 0:
            raise SystemExit(f"the model has no camera {name!r}")

        # **The model's head camera faces backwards.** The camera's view direction is -x where the
        # duck's forward axis is +x, exactly 180 degrees out, so an uncorrected picture shows what is
        # behind the duck. Turned 180 degrees about the camera's own right axis: yaw would leave it
        # upside down, and roll would leave the image on its side (which is what a real camera's
        # quarter-turn mount produces and what `mediad --rotate 90` describes).
        turn = np.array([0.0, 1.0, 0.0, 0.0])  # 180 degrees about x, scalar-first
        fixed = np.zeros(4)
        mujoco.mju_mulQuat(fixed, model.cam_quat[self.camera], turn)
        model.cam_quat[self.camera] = fixed

        self.model = model
        self.width = width
        self.height = height
        self.renderer: mujoco.Renderer | None = None
        self.latest: bytes | None = None
        self.lock = threading.Lock()
        # 帧服务器的读者在“有新帧”时被唤醒，而不是按固定周期重发同一张图。
        self.updated = threading.Condition(self.lock)
        self.timings: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.split: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._split_frames = 0
        self._split_sum = [0.0, 0.0, 0.0]
        self._split_next = time.perf_counter() + 5.0

    def _ensure_renderer(self) -> None:
        if self.renderer is not None:
            return
        model = self.model
        # MuJoCo caps offscreen rendering at the model's <global offwidth/offheight>, 640x480 unless
        # the scene says otherwise; a larger request fails when the Renderer is constructed.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, self.height)

        # Offscreen rending is fill-rate bound, and the defaults are generous: 4x MSAA samples and a
        # 4096x4096 shadow map, redrawn for every frame. A camera that exists to be looked at, not
        # photographed, can pay much less. `offsamples=0` turns MSAA off, `shadowsize=0` keeps the
        # model's own shadow map size.
        quality = model.vis.quality
        quality.offsamples = int(os.environ.get("DUCK_SIM_CAMERA_OFFSAMPLES", "0"))
        shadowsize = int(os.environ.get("DUCK_SIM_CAMERA_SHADOWSIZE", "0"))
        if shadowsize:
            quality.shadowsize = shadowsize
        print(
            f"== camera: {self.width}x{self.height} renderer, offsamples={quality.offsamples},"
            f" shadowsize={quality.shadowsize}, split={int(SPLIT)},"
            f" shadow={int(ENABLE_SHADOW)}, reflection={int(ENABLE_REFLECTION)},"
            f" skybox={int(ENABLE_SKYBOX)}, haze={int(ENABLE_HAZE)}",
            flush=True,
        )
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)

    def _render_pixels(self) -> np.ndarray:
        """`Renderer.render()` with the scene passes we want, and the split timings when asked."""
        renderer = self.renderer
        flags = renderer._scene.flags
        flags[mujoco.mjtRndFlag.mjRND_SHADOW] = ENABLE_SHADOW
        flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = ENABLE_REFLECTION
        flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = ENABLE_SKYBOX
        flags[mujoco.mjtRndFlag.mjRND_HAZE] = ENABLE_HAZE
        if not SPLIT:
            return renderer.render()

        if renderer._gl_context:
            renderer._gl_context.make_current()
        out = np.empty((self.height, self.width, 3), dtype=np.uint8)
        started = time.perf_counter()
        mujoco.mjr_render(renderer._rect, renderer._scene, renderer._mjr_context)
        drawn = time.perf_counter()
        mujoco.mjr_readPixels(out, None, renderer._rect, renderer._mjr_context)
        read = time.perf_counter()
        if renderer._gl_context:
            out[:] = np.flipud(out)
        flipped = time.perf_counter()
        self.split = (
            (drawn - started) * 1000.0,
            (read - drawn) * 1000.0,
            (flipped - read) * 1000.0,
        )
        for index in range(3):
            self._split_sum[index] += self.split[index]
        self._split_frames += 1
        if flipped >= self._split_next:
            frames = self._split_frames
            print(
                f"== camera split: draw {self._split_sum[0] / frames:.1f} ms,"
                f" read {self._split_sum[1] / frames:.1f} ms,"
                f" flip {self._split_sum[2] / frames:.1f} ms",
                flush=True,
            )
            self._split_frames = 0
            self._split_sum = [0.0, 0.0, 0.0]
            self._split_next = time.perf_counter() + 5.0
        return out

    def render(self, world) -> None:
        """Render one frame. `update_scene` reads the whole of `MjData`, so it holds the world lock.

        Sensor reads run on socket threads and can catch `MjData` mid-write: a ToF read that found a
        zero-length ray direction once aborted the simulator. The lock covers the scene copy only;
        the draw touches no shared state.
        """
        self._ensure_renderer()
        scene_started = time.perf_counter()
        with world.lock:
            self.renderer.update_scene(world.data, camera=self.camera)
        draw_started = time.perf_counter()
        pixels = self._render_pixels()
        encode_started = time.perf_counter()
        packed = to_uyvy(pixels)
        self.timings = (
            (draw_started - scene_started) * 1000.0,
            (encode_started - draw_started) * 1000.0,
            (time.perf_counter() - encode_started) * 1000.0,
        )
        with self.updated:
            self.latest = packed
            self.updated.notify_all()

    def wait_frame(self, previous: bytes | None, timeout: float = 5.0) -> bytes | None:
        """The next *new* frame, or None if the camera stayed silent for `timeout`."""
        with self.updated:
            self.updated.wait_for(
                lambda: self.latest is not None and self.latest is not previous, timeout
            )
            return self.latest


class FrameHandler(socketserver.BaseRequestHandler):
    """Length-prefixed frames, at the camera's rate, until the reader goes away."""

    def handle(self) -> None:
        self.server.readers += 1
        try:
            self._serve()
        finally:
            self.server.readers -= 1

    def _serve(self) -> None:
        camera: Camera = self.server.camera
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"== camera: a reader connected from {self.client_address}", flush=True)
        last: bytes | None = None
        try:
            while True:
                frame = camera.wait_frame(last)
                if frame is None:
                    continue
                last = frame
                self.request.sendall(struct.pack("<I", len(frame)) + frame)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        print("== camera: the reader went away", flush=True)


class FrameServer(socketserver.ThreadingTCPServer):
    """Serves one camera's latest frame. `readers` is what makes an unwatched camera free."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.readers = 0
