# 复现说明

下列命令从仓库根目录执行。推荐 Linux 或 WSL2 Ubuntu、Python 3.12。保存的轨迹视频可在 GitHub 页面直接下载查看，结果 JSON 可用任何文本编辑器打开。两阶段使用独立虚拟环境。

## 1. 单摆 · CPU

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements/requirements.txt
.venv/bin/python lessons/01_simulation_step.py
.venv/bin/python lessons/02_pd_control.py
.venv/bin/python lessons/03_gravity_compensation.py
.venv/bin/python lessons/04_rl_environment.py
.venv/bin/python lessons/05_ppo.py --steps 200000
```

第一条实验生成电机 XML，后续 PD 和重力补偿读同一个模型。结果写到本地 `artifacts/`。训练后的 `latest_run.txt` 记录实际保存位置，可以用 `lessons/05_ppo.py --evaluate` 复测，或用 `--view` 打开观察窗口。发布时保存的策略与独立评测在 [单摆 PPO 证据](../evidence/lesson05/)。

要复测发布权重，可先复制到本地输出目录，再用 `--run` 指向该目录。评测会将新的 JSON 写到复制目录。

```bash
mkdir -p artifacts/pendulum-release
cp evidence/lesson05/{initial,best_model,final}.zip artifacts/pendulum-release/
.venv/bin/python lessons/05_ppo.py --evaluate --run artifacts/pendulum-release
```

## 2. 小鸭子 · 上游环境与依赖

```bash
python3.12 bootstrap_duck.py
python3.12 -m venv .venv-duck
.venv-duck/bin/python -m pip install -r requirements/requirements-duck.lock
.venv-duck/bin/python -m pip install onnxruntime==1.20.1
```

初始化从固定 Git 提交获取上游环境，校验参考动作和适配后的 `joystick.py`，并叠加仓库里的 [`extensions/`](../extensions/)。生成的 `third_party/` 与 `duck_training/` 留在本地。训练入口检查包版本、GPU 和保存环境的源码哈希。锁文件面向 Linux x86-64 与 CUDA 12。CPU 回放显式设 `JAX_PLATFORMS=cpu`，GUI 需要本地显示环境。

### 已保存的双足策略与手柄

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-duck/bin/python demo/native_assisted_gamepad.py --moving-entry
```

MuJoCo Viewer 打开后，终端显示本机手柄地址。在浏览器访问该地址，使用 `W/S/A/D` 或面板控制，`空格` 归零，`R` 重置。以下无界面验证用同一初态和命令逐步对照原生轨迹。

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-duck/bin/python demo/native_assisted_gamepad.py \
  --moving-entry --verify --sequence arcs --seed 6003
```

### 小批量 PPO 训练与检查点续训

在支持 CUDA 12 的 NVIDIA 环境中，用下面的命令检查 PPO 更新、写盘和回载。低步数用于验证路径。

```bash
.venv-duck/bin/python lessons/07_duck_ppo.py \
  --train --task-profile gait_forward --envs 32 --steps 1280 --skip-rollouts
```

也可用已发布的 [`ppo-round-21`](../checkpoints/ppo-round-21/) 接续当前环境。下面跑的是小批量示例。第 22 轮完整实验另使用 2,048 个环境、六组示范监督数据和相应学习率，完整参数与数据哈希在 [保存配置](../evidence/duck/rounds/round-22/config.json)。公开 Git 包提供训练代码、选定检查点和评测结果，历史大样本采集集未打入 Git。

```bash
.venv-duck/bin/python lessons/07_duck_ppo.py \
  --train --preset transfer --envs 32 --steps 100000 --evals 1 --skip-rollouts \
  --warm-start checkpoints/ppo-round-21/checkpoints/step_5038080
```

### 原生命令 A/B 与作者权重

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-duck/bin/python demo/benchmark_author.py \
  --run checkpoints/ppo-round-22 --checkpoint step_10076160 --cpu-restore \
  --paired-ramp --extended-switches --output artifacts/round22_recheck.json

python3.12 bootstrap_duck.py --author-only
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-duck/bin/python demo/compare_moving_entry_author.py
```

`compare_moving_entry_author.py` 读取手柄配对用的命令路线，对作者 ONNX 运行相同种子，输出逐段记录。安装脚本下载的参考轨迹是固定来源的 pickle 文件。使用默认固定提交与哈希，可避免把未知来源的 pickle 传给训练环境。

## 3. 复核发布数据

```bash
python3 scripts/summarize_evidence.py
PYTHONPATH="$PWD/duck_training:$PWD/lessons" JAX_PLATFORMS=cpu \
  .venv-duck/bin/python -m unittest discover -s tests -q
```

`evidence/` 是历史实验的导出快照，`artifacts/` 是这次运行生成的结果。已整理的训练和控制器检查点见 [`checkpoints/`](../checkpoints/)。旧后退课程代码可在 [`history/environment_profiles/`](../history/environment_profiles/) 阅读。 `demo/` 保留了当前手柄用到的原始实验组件，其较早的探针入口可能引用未随发布包分发的历史轮次。

**发布版核验。** 单摆第 01 至 04 课重跑通过，保存的三个 PPO 模型和 PD 都在原来的 10 个种子上重新评估。双足侧 53 项测试通过。公开仓库重新克隆后可重建训练环境，指定种子的弧线后退控制逐步复现 1,432 帧轨迹。作者 ONNX 的 80 局对照逐段复现原始记录。32 个 CUDA 并行环境的小批量训练完成 1,280 步，检查点重载动作最大误差为 `0.0`。[小批量训练检查](../evidence/duck/publication_gpu_smoke.json)
