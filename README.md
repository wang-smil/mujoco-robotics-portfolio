# Open Duck Mini v2 · MuJoCo 复现与 PPO 后训练

基于 Open Duck Mini v2 的机器人模型、参考步态和训练环境，在平地仿真中复现 PPO 训练与回放，并追加头颈稳定、低速后退和连续换向实验。仓库也包含用于验证控制和训练流程的单摆基线。

[![同种子直走与后退换向对照](docs/media/moving_reverse_comparison.png)](docs/media/moving_reverse_comparison.mp4)

## 核心修改与结果

| 问题 | 实现 | 实验结果 |
|---|---|---|
| 行走时头颈摆动明显 | 限制 4 个头颈执行器的目标角变化率，再从前进检查点接续 PPO。[限速代码](extensions/playground/open_duck_mini_v2/profiles/locomotion_neck.py) | 实际头颈关节速度 RMS 从 **1.2473 降到 0.3481 rad/s**，接续训练组基础测试 **80/80** 通过。[三组对照](evidence/duck/neck/final_metrics.json) |
| 策略容易站住，低速后退和换向难学 | 增加后退与换向命令样本，用实测过渡状态初始化部分训练局，在 PPO 中加入示范动作监督。[重置](extensions/playground/open_duck_mini_v2/locomotion_snapshot_reset.py) · [示范损失](lessons/ppo_demonstration_anchor.py) | 第 22 轮单 Actor 直接后退 **8/8 跑满**，平均约 **−0.053 m/s**。同权重使用缓变命令时 **0/8 跑满**。[直接命令](evidence/duck/round22_native_final.json) · [缓变命令](evidence/duck/round22_native_final_ramp.json) |
| 手柄控制从行进切入后退时不稳 | 演示使用示范初始化的 Actor，在运行时加入后退切入相位、俯仰反馈和髋踝速度修正。[手柄控制](demo/native_assisted_gamepad.py) · [过渡逻辑](demo/probe_moving_reverse.py) | 相对上一版辅助控制器，20 个相同种子上的直走与后退互换**整条路线达标从 1/20 到 15/20**，跑满从 **20/20 到 18/20**。[配对数据](evidence/duck/reverse_moving_entry_matched.json) |

另以作者发布的 ONNX 权重运行相同平地路线。标准路线的 `−0.05 m/s` 后退段，作者权重平均速度约 `−0.0001 m/s`，本地辅助版为 `−0.0508 m/s`。[作者权重评测](evidence/duck/author_moving_entry_matched.json) · [实验日志与原始记录](docs/03-timeline.md)

## 运行

从仓库根目录执行。小鸭子回放使用 Linux 或 WSL2 Ubuntu、Python 3.12 和图形显示。下面的 PPO 训练示例使用 CUDA 12 GPU。

```bash
python3.12 bootstrap_duck.py
python3.12 -m venv .venv-duck
.venv-duck/bin/python -m pip install -r requirements/requirements-duck.lock
.venv-duck/bin/python -m pip install onnxruntime==1.20.1
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-duck/bin/python demo/native_assisted_gamepad.py --moving-entry
```

打开终端显示的本地控制页。用 `W/S/A/D` 行进与转向，空格归零，`R` 重置。

```bash
# 小批量 PPO 训练，验证更新、保存和回载
.venv-duck/bin/python lessons/07_duck_ppo.py \
  --train --task-profile gait_forward --envs 32 --steps 1280 --skip-rollouts

# 从已发布的检查点接续训练
.venv-duck/bin/python lessons/07_duck_ppo.py \
  --train --preset transfer --envs 32 --steps 100000 --evals 1 --skip-rollouts \
  --warm-start checkpoints/ppo-round-21/checkpoints/step_5038080
```

上述命令用于验证训练入口。第 22 轮正式实验使用额外的示范数据，大样本采集集未随公开仓库分发。[正式配置与日志](evidence/duck/rounds/round-22/)

## 代码入口

| 入口 | 用途 |
|---|---|
| [环境扩展](extensions/playground/open_duck_mini_v2/) | 命令课程、任务奖励、头颈限速与过渡重置 |
| [PPO 训练入口](lessons/07_duck_ppo.py) | 环境选择、MJX/Brax PPO 与检查点接续 |
| [评测与手柄](demo/) | 原生 MuJoCo 对照、上游 ONNX 基线与交互演示 |
| [检查点](checkpoints/) · [原始记录](evidence/) | 发布权重、参数、逐段评测及训练日志 |

单摆使用独立 CPU 环境。安装 [依赖](requirements/requirements.txt) 后，按顺序运行 [第 01–05 课脚本](lessons/)，复现物理步进、PD、重力补偿和 SB3 PPO。重力补偿将该实验的稳态误差从 `0.0884 rad` 降至约 `3.4×10⁻⁷ rad`。[单摆实验日志](docs/03-timeline.md#单摆)

## 八足小蜘蛛

从单腿 MJCF 搭建八足机器人，用交错步态的动作示范热启动 MJX/Brax PPO。预训练策略支持前进、后退、横移、转向和弧线，两套已发布权重均通过 **19/19** 条原生 MuJoCo 平地路线。Windows 可直接用手柄驾驶。

[演示、快速运行与训练复现](docs/04-spider.md) · [平地 A/B 评测与训练日志](evidence/spider/)

## 上游来源

机器人、MJCF、参考步态和基础任务来自 Antoine Pirrone 与 Open Duck Mini 贡献者。`bootstrap_duck.py` 从固定提交获取 Open Duck Playground，再叠加本地扩展。作者权重基线为 `BEST_WALK_ONNX_2.onnx`。[Open Duck Mini v2](https://github.com/apirrone/Open_Duck_Mini/tree/v2) · [固定环境提交](https://github.com/apirrone/Open_Duck_Playground/tree/b9be205ac64488c23504ca42e5ec790337adeec3) · [作者权重](https://github.com/apirrone/Open_Duck_Mini/blob/b23317a485b3cec7d8417f352478778b3475173c/BEST_WALK_ONNX_2.onnx) · [上游许可](https://github.com/apirrone/Open_Duck_Mini/blob/v2/LICENSE)
