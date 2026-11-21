# MuJoCo Control & Locomotion

两个可复现的机器人仿真项目。单摆实现闭环控制与 PPO，Open Duck Mini v2 实现双足训练、速度跟踪和交互控制。

[![Open Duck Mini 同种子换向对照](docs/media/moving_reverse_comparison.png)](docs/media/moving_reverse_comparison.mp4)

[双足换向](docs/media/moving_reverse_comparison.mp4) · [头颈限速](docs/media/neck-ablation.mp4) · [单摆轨迹](docs/media/pendulum-controls.png)

## 项目结果

| 项目 | 实现 | 验证结果 |
|---|---|---|
| [单摆控制](docs/01-pendulum.md) | MJCF、PD、重力补偿、Gymnasium、SB3 PPO | 重力补偿将稳态误差从 `0.0884 rad` 降到 `3.4×10⁻⁷ rad`。PPO 训练 `200,704` 步，10 个测试种子平均回报 `375.90`。 |
| [双足运动](docs/02-duck.md) | MJX/Brax PPO、动作示范、姿态反馈、交互控制 | 头颈速度 RMS 从 `1.2473` 降到 `0.3481 rad/s`；同种子前后换向从 `1/20` 提升到 `15/20` 达标，`18/20` 跑完整条路线。 |

结果来自仿真。跟踪达标率、跑满率和逐段误差见 [评测数据](evidence/duck/)。

## 运行预训练策略

按 [安装说明](docs/reproduce.md#2-小鸭子--上游环境与依赖) 初始化 Ubuntu / WSL 环境后，从仓库根目录运行：

```bash
JAX_PLATFORMS=cpu .venv-duck/bin/python demo/native_assisted_gamepad.py --moving-entry
```

MuJoCo 打开后，从终端链接打开控制面板。[单摆运行](docs/reproduce.md#1-单摆--cpu) · [训练与续训](docs/reproduce.md#小批量-ppo-训练与检查点续训) · [检查点](checkpoints/)

## 工程入口

[`lessons/`](lessons/) 提供训练入口，双足环境与奖励在 [`extensions/`](extensions/)，评测和交互控制在 [`demo/`](demo/)。

[实验时间线](docs/03-timeline.md) · [逐轮记录 Release](https://github.com/wang-smil/mujoco-robotics-portfolio/releases/tag/experiment-archive) · [上游与资产来源](docs/credits.md)
