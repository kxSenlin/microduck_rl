# 本机 fork 说明（kxSenlin/microduck_rl，分支 `local/wsl`）

这是 [pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl) 的 fork。
**上游默认分支是 `develop`，不是 `main`**；本仓库的本地改动都在 `local/wsl` 分支，基于
`develop`，方便随时 merge：`git merge upstream/develop`。

换机复现时先照
[duck-client 的 docs/local-setup.md](https://github.com/kxSenlin/duck-client/blob/master/docs/local-setup.md)
准备系统（WSLg、GPU、`~/Pollen/` 布局、bashrc 托管块），再回来看本文件。

## 本 fork 改了什么

相对上游 `develop` 只改了两个仿真文件（`git diff --stat upstream/develop...HEAD`：

| 文件 | 改动 |
|---|---|
| `src/mjlab_microduck/sim/camera.py` | 独立的相机渲染线程、惰性启动、720p、PIL JPEG 编码；每只鸭子一个帧端口；渲染 pass 可关（`DUCK_SIM_CAMERA_SHADOW/REFLECTION/SKYBOX/HAZE/OFFSAMPLES/SHADOWSIZE/SPLIT`）；`medial` 帧端口协议与 duck-client `video.py` 对接 |
| `src/mjlab_microduck/sim/body_server.py` | `--viewer native\|viser` + `--viewer-port`；`DUCK_SIM_TRUTH_PATH` 真值发布：每个物理步（50 Hz）写相机/躯干/散落物件位姿（`cameras`/`bodies`/`objects`），供 duck-client 的 `/ws/pose` 和 WebGL 场景使用 |

这些改动是 duck-client 头摄（WebGL 自己画房间、真值位姿、动态物件跟随）的仿真端依赖；
`duck-client` 在真机上不走这条路径（`is_sim_endpoint` 隔离）。

## fork 引入/使用的环境变量

权威表在 [duck-client/docs/env.md](https://github.com/kxSenlin/duck-client/blob/master/docs/env.md)。
本仓库仿真侧相关的：

| 变量 | 默认值 | 作用 |
|---|---|---|
| `DUCK_SIM_TRUTH_PATH` | 未设置 | 设置后 `body_server` 每个物理步写真值 JSON（相机/躯干/散落物件）；客户端用 `DUCK_TRUTH_PATH` 读它。**不设置就没有真值通道** |
| `DUCK_SIM_CAMERA_WIDTH` | `1280` | 相机渲染宽 |
| `DUCK_SIM_CAMERA_HEIGHT` | `720` | 相机渲染高 |
| `DUCK_SIM_CAMERA_OFFSAMPLES` | `0` | 离屏 MSAA 采样数；0 = 关闭 |
| `DUCK_SIM_CAMERA_SHADOWSIZE` | `0` | 阴影贴图大小；0 = 模型默认 |
| `DUCK_SIM_CAMERA_STATS` | `0` | 1 = 每 5 秒打印渲染帧率/耗时 |
| `DUCK_SIM_CAMERA_SPLIT` | `0` | 1 = 分别打印 `mjr_render` / `mjr_readPixels` / flip 耗时 |
| `DUCK_SIM_CAMERA_SHADOW` / `REFLECTION` / `SKYBOX` / `HAZE` | `0`/`0`/`1`/`1` | 相机渲染 pass 开关 |

`--cameras` / `--frame-port` / `--camera-fps` 由 `microduck` fork 的 `scripts/duck-sim` 传入；
`DUCK_SIM_CAMERAS` / `DUCK_SIM_FRAME_PORT` / `DUCK_SIM_CAMERA_FPS` 也在那张表里。

## 安装

```bash
cd ~/Pollen
git clone git@github.com:kxSenlin/microduck_rl.git
cd microduck_rl
git checkout local/wsl

# Python 环境由 uv.lock 固定（Python 3.12）
uv sync
uv pip install mjviser        # viser 场景查看器需要
```

## 运行

推荐用 `microduck` fork 的 `scripts/duck-sim` 一起启动（它会自动带相机参数和真值路径）：

```bash
cd ~/Pollen/microduck
DUCK_SIM_TRUTH_PATH=$HOME/.cache/duck-sim/truth.json scripts/duck-sim boot 2
```

只想单独起 body_server：

```bash
cd ~/Pollen/microduck_rl
DUCK_SIM_TRUTH_PATH=$HOME/.cache/duck-sim/truth.json \
  .venv/bin/python -m mjlab_microduck.sim.body_server \
  --port 7801 --ducks 1 --keyframe SIT --viewer viser --viewer-port 8091 \
  --cameras all --frame-port 7901 --camera-fps 30
```

## 与上游同步

```bash
cd ~/Pollen/microduck_rl
git fetch upstream
git checkout local/wsl
git merge upstream/develop     # 2026-10-30 已同步（5 个上游提交：training bundles 等，无冲突）
uv sync                        # 上游改过 pyproject 时
```

冲突通常只会出现在 `src/mjlab_microduck/sim/` 下我们改过的两个文件。解决原则：**保留相机/
真值/viser 扩展，再吸收上游的 sim 逻辑**；改完必须用 `duck-sim boot` 验证一次。

## 快速验证（换机后）

```bash
cd ~/Pollen/microduck_rl
git branch --show-current                                # local/wsl
uv run python -c "import mujoco, mjviser; print('deps ok')"

cd ~/Pollen/microduck
DUCK_SIM_TRUTH_PATH=$HOME/.cache/duck-sim/truth.json scripts/duck-sim boot 1
python3 -c "import json;d=json.load(open('$HOME/.cache/duck-sim/truth.json'));print(sorted(d), sorted(d['objects']))"
scripts/duck-sim down
```
