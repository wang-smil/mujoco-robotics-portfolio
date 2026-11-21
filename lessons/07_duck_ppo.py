"""第7课：真实MJX小鸭子的PPO训练、评测与策略回放。

在.venv-duck中运行，默认smoke；噪声、推力、参考与控制限制由task_profile及run保存配置确定。
本课当前未开启参数domain randomization。
"""

import argparse
from contextlib import nullcontext
import copy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import queue
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "duck_training"
OUTPUT = ROOT / "artifacts" / "lesson07"
LATEST = OUTPUT / "latest_run.txt"
EXPECTED = {"jax": "0.5.0", "mujoco": "3.2.7", "mujoco-mjx": "3.2.7",
            "playground": "0.0.3", "brax": "0.12.1", "flax": "0.10.2",
            "orbax-checkpoint": "0.11.5", "optax": "0.2.4", "chex": "0.1.88",
            "numpy": "1.26.4", "scipy": "1.14.1"}
COMBINED_PROFILES = ("neck_combined", "neck_combined_slew", "neck_combined_slew_stable",
                     "neck_combined_balance", "neck_combined_reversal_pairs", "neck_combined_snapshot_reset")
TASK_PROFILES = ("upstream", "forward", "joystick", "gait_forward", "neck_forward", "neck_turn",
                 "neck_turn_tracking", "neck_reverse", "neck_reverse_tracking", "neck_reverse_progress",
                 "neck_reverse_reset", "neck_arc", "neck_arc_switches") + COMBINED_PROFILES


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    for mode in ("train", "evaluate", "view"):
        modes.add_argument(f"--{mode}", action="store_true")
    parser.add_argument("--steps", type=int, default=1280)
    parser.add_argument("--envs", type=int, choices=(8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192), default=32)
    parser.add_argument("--seed", type=int, default=0, help="PPO training random seed; evaluation seeds are separate")
    parser.add_argument("--evals", type=int, default=1,
                        help="训练分成几段评价并存档；1 表示只在结束时存一次，10 可每 1/10 步存一次")
    parser.add_argument("--unroll", type=int, default=None,
                        help="每轮采样/更新的时间步数；默认由 --preset 决定")
    parser.add_argument("--preset", choices=("lesson", "transfer", "official"), default="lesson",
                        help="lesson=64x2；transfer=接续旧模型128x4/256x5；official为transfer的历史兼容名称，并非作者完整配置")
    parser.add_argument("--run", type=Path, help="已有训练目录；评价/窗口默认使用 latest_run")
    parser.add_argument("--task-profile", choices=TASK_PROFILES, default=None,
                        help="upstream=保留原任务；forward=新版前进/停稳；joystick=新版多方向任务")
    parser.add_argument("--command", type=float, nargs=3, metavar=("VX", "VY", "WZ"),
                        help="评价/回放固定本体速度命令，头部目标置零；例如 --command .1 0 0")
    parser.add_argument("--checkpoint", help="评价/回放指定 run/checkpoints/ 下的 step_*，默认最后一个")
    parser.add_argument("--skip-rollouts", action="store_true",
                        help="容量短测可跳过训练结束后的长轨迹；仍执行 PPO 评价、存档与重载检查")
    parser.add_argument("--warm-start", type=Path,
                        help="训练从已有 checkpoints/step_* 的网络和归一化器接续；优化器重置，新建 run")
    parser.add_argument("--turn-yaw-error-multiplier", type=float,
                        help="Override the pure-turn yaw-tracking penalty for a continuation of neck_turn_tracking")
    parser.add_argument("--reverse-probability", type=float,
                        help="Override the reverse command frequency for a saved reverse curriculum")
    parser.add_argument("--reverse-speed-min", type=float,
                        help="Lower the minimum sampled backward speed in a saved curriculum")
    parser.add_argument("--learning-rate", type=float,
                        help="Set PPO learning rate for a training run")
    parser.add_argument("--exploration-scale", type=float,
                        help="Scale the Gaussian standard deviation throughout PPO")
    parser.add_argument("--imitation-anchor", action="append", type=Path, default=[],
                        help="Validated demonstration NPZ, for auxiliary imitation during PPO training")
    parser.add_argument("--imitation-weight", type=float, default=4., help="Weight of demonstration action error")
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps 必须为正数")
    if args.evals < 1:
        parser.error("--evals 必须 ≥ 1")
    if args.unroll is not None and args.unroll <= 0:
        parser.error("--unroll 必须为正数")
    if not (args.evaluate or args.view) and args.steps < args.envs * (args.unroll or 1) * args.evals:
        parser.error("--steps 太小：至少要 --envs × --unroll × --evals 才能分成这么多段")
    if args.run and not (args.evaluate or args.view):
        parser.error("--run 只用于 --evaluate 或 --view；训练总是创建新目录")
    if args.command is not None:
        if not (args.evaluate or args.view) or not all(math.isfinite(x) for x in args.command):
            parser.error("--command 仅用于评价/回放，且三个数必须有限")
    if args.checkpoint and not (args.evaluate or args.view):
        parser.error("--checkpoint 仅用于评价/回放")
    if args.warm_start:
        if not args.train:
            parser.error("--warm-start 仅用于训练")
        args.warm_start = args.warm_start.resolve()
    if args.turn_yaw_error_multiplier is not None:
        if not args.train or not args.warm_start or not 1. <= args.turn_yaw_error_multiplier <= 8.:
            parser.error("--turn-yaw-error-multiplier requires turn continuation and a multiplier in [1, 8]")
    if args.reverse_probability is not None:
        if not args.train or not args.warm_start or not 0. < args.reverse_probability <= .4:
            parser.error("--reverse-probability requires reverse continuation and a probability in (0, 0.4]")
    if args.reverse_speed_min is not None:
        if not args.train or not args.warm_start or not .01 <= args.reverse_speed_min <= .05:
            parser.error("--reverse-speed-min requires continuation and a speed in [0.01, 0.05]")
    if args.learning_rate is not None and (not args.train or not 0. < args.learning_rate <= .001):
        parser.error("--learning-rate requires --train and a rate in (0, .001]")
    if args.exploration_scale is not None and (not args.train or not 0. < args.exploration_scale <= 1.):
        parser.error("--exploration-scale requires --train and a scale in (0, 1]")
    if args.imitation_anchor:
        if not args.train or not 0. < args.imitation_weight <= 20.:
            parser.error("--imitation-anchor requires --train and an imitation weight in (0, 20]")
        args.imitation_anchor = [path.resolve() for path in args.imitation_anchor]
    return args


def unique_name(prefix):
    return prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:6]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                              default=lambda x: np.asarray(x).tolist()) + "\n", encoding="utf-8")


def load_runtime():
    """先设进程内显存策略，再导入 JAX；不改系统配置或其他虚拟环境。"""
    global jax, jp, np, mujoco, joystick, ppo, ppo_networks, ppo_losses, ocp, wrapper
    if Path(sys.prefix).resolve() != (ROOT / ".venv-duck").resolve():
        raise RuntimeError(f"请使用 {ROOT / '.venv-duck/bin/python'} 运行本课")
    if not (WORK / "playground/open_duck_mini_v2/joystick.py").is_file():
        raise FileNotFoundError(f"训练副本未准备好：{WORK}")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.65")
    sys.path.insert(0, str(WORK))
    os.chdir(WORK)  # 上游 PolyReferenceMotion 按 cwd 读取参考 pickle。
    versions = {name: metadata.version(name) for name in EXPECTED}
    if versions != EXPECTED:
        raise RuntimeError(f"依赖版本与本课固定 API 不同：{versions}")
    import jax
    jax.config.update("jax_compilation_cache_dir", str(OUTPUT / "jax_cache"))
    import jax.numpy as jp
    import numpy as np
    import mujoco
    from playground.open_duck_mini_v2 import joystick
    from brax.training.agents.ppo import train as ppo, networks as ppo_networks, losses as ppo_losses
    from mujoco_playground import wrapper
    from orbax import checkpoint as ocp
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"本课要求 GPU，实际设备：{jax.devices()}")
    result = jax.jit(lambda x: x @ x)(jp.eye(4)).block_until_ready()
    assert result.device.platform == "gpu" and np.allclose(np.asarray(result), np.eye(4))
    return versions


def environment_settings(recorded=None, task_profile="upstream"):
    """Old runs have no profile. Return a detached copy of saved parameters."""
    profile = recorded.get("task_profile", "upstream") if recorded is not None else task_profile
    if profile not in TASK_PROFILES:
        raise ValueError(f"Unknown task profile: {profile}")
    task = recorded.get("task", "flat_terrain") if recorded is not None else "flat_terrain"
    values = copy.deepcopy(recorded.get("environment")) if recorded is not None else None
    return profile, task, values


def disable_assisted_resets(env):
    if "reverse_snapshot_probability" in env._config:
        env._config.reverse_snapshot_probability = 0.


def patch_evaluation_command(env, state, target):
    from playground.open_duck_mini_v2.locomotion import patch_command_observation

    command = jp.concatenate([target, jp.zeros(4)])
    reference = env.PRM.get_reference_motion(*target, state.info["imitation_i"])
    info = {**state.info, "command": command, "current_reference_motion": reference}
    if "command_target" in info:
        info.update(command_target=command, command_ramp_active=jp.asarray(False))
    obs = patch_command_observation(state.obs, command, reference)
    state = state.replace(info=info, obs=obs)
    return env._observe_balance(state) if env._config.get("balance_observations", False) else state


def make_environment(recorded=None, task_profile="upstream", command=None):
    from ml_collections import config_dict
    profile, task, values = environment_settings(recorded, task_profile)
    if profile == "upstream":
        cls, default = joystick.Joystick, joystick.default_config
    elif profile == "neck_forward":
        from playground.open_duck_mini_v2 import locomotion_neck
        cls, default = locomotion_neck.NeckLimitedGait, locomotion_neck.default_config
    elif profile == "neck_turn":
        from playground.open_duck_mini_v2 import locomotion_turn
        cls, default = locomotion_turn.ConservativeTurn, locomotion_turn.default_config
    elif profile == "neck_turn_tracking":
        from playground.open_duck_mini_v2 import locomotion_turn_tracking
        cls, default = locomotion_turn_tracking.TurnTracking, locomotion_turn_tracking.default_config
    elif profile == "neck_reverse":
        from playground.open_duck_mini_v2 import locomotion_reverse
        cls, default = locomotion_reverse.ReverseCurriculum, locomotion_reverse.default_config
    elif profile == "neck_reverse_tracking":
        from playground.open_duck_mini_v2 import locomotion_reverse_tracking
        cls, default = locomotion_reverse_tracking.ReverseTracking, locomotion_reverse_tracking.default_config
    elif profile == "neck_reverse_progress":
        from playground.open_duck_mini_v2 import locomotion_reverse_progress
        cls, default = locomotion_reverse_progress.ReverseProgress, locomotion_reverse_progress.default_config
    elif profile == "neck_reverse_reset":
        from playground.open_duck_mini_v2 import locomotion_reverse_reset
        cls, default = locomotion_reverse_reset.ReverseReferenceReset, locomotion_reverse_reset.default_config
    elif profile == "neck_arc":
        from playground.open_duck_mini_v2 import locomotion_arc
        cls, default = locomotion_arc.ArcCurriculum, locomotion_arc.default_config
    elif profile == "neck_arc_switches":
        from playground.open_duck_mini_v2 import locomotion_arc_switches
        cls, default = locomotion_arc_switches.ArcSwitches, locomotion_arc_switches.default_config
    elif profile == "neck_combined":
        from playground.open_duck_mini_v2 import locomotion_combined
        cls, default = locomotion_combined.CombinedLocomotion, locomotion_combined.default_config
    elif profile == "neck_combined_slew":
        from playground.open_duck_mini_v2 import locomotion_combined_slew
        cls, default = locomotion_combined_slew.SlewedLocomotion, locomotion_combined_slew.default_config
    elif profile == "neck_combined_slew_stable":
        from playground.open_duck_mini_v2 import locomotion_combined_slew_stable
        cls, default = locomotion_combined_slew_stable.StableReverseLocomotion, locomotion_combined_slew_stable.default_config
    elif profile == "neck_combined_balance":
        from playground.open_duck_mini_v2 import locomotion_combined_balance
        cls, default = locomotion_combined_balance.BalanceLocomotion, locomotion_combined_balance.default_config
    elif profile == "neck_combined_reversal_pairs":
        from playground.open_duck_mini_v2 import locomotion_reversal_pairs
        cls, default = locomotion_reversal_pairs.ReversalPairs, locomotion_reversal_pairs.default_config
    elif profile == "neck_combined_snapshot_reset":
        from playground.open_duck_mini_v2 import locomotion_snapshot_reset
        cls, default = locomotion_snapshot_reset.SnapshotReset, locomotion_snapshot_reset.default_config
    elif profile == "gait_forward":
        from playground.open_duck_mini_v2 import locomotion_gait
        cls, default = locomotion_gait.GaitLocomotion, locomotion_gait.default_config
    else:
        from playground.open_duck_mini_v2 import locomotion
        cls, default = locomotion.Locomotion, lambda: locomotion.default_config(profile)
    if profile in COMBINED_PROFILES[1:] and values is not None:
        cfg = config_dict.ConfigDict({**default().to_dict(), **values})
    else:
        cfg = config_dict.ConfigDict(values) if values is not None else default()
    if command is not None:
        if profile in COMBINED_PROFILES[1:]:
            cfg.command_ramp_probability = 0.
        if profile in COMBINED_PROFILES[4:]:
            cfg.reverse_pair_probability = 0.
            cfg.stand_reverse_probability = 0.
        if profile == "neck_combined_snapshot_reset":
            cfg.reverse_snapshot_probability = 0.
        fixed = tuple(command) + (0., 0., 0., 0.)
        class FixedCommand(cls):
            def sample_command(self, rng):
                return jp.array(fixed)
        cls = FixedCommand
    return cls(task=task, config=cfg)


def environment_source_paths(profile):
    paths = [WORK / "playground/open_duck_mini_v2" / name for name in
             ("base.py", "joystick.py", "custom_rewards.py", "constants.py")]
    paths += [WORK / "playground/common" / name for name in ("rewards.py", "poly_reference_motion.py")]
    paths.append(WORK / "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl")
    if profile != "upstream":
        paths.append(WORK / "playground/open_duck_mini_v2/locomotion.py")
    if profile == "gait_forward" or profile.startswith("neck_"):
        paths.append(WORK / "playground/open_duck_mini_v2/locomotion_gait.py")
    if profile.startswith("neck_"):
        paths.append(WORK / "playground/open_duck_mini_v2/locomotion_neck.py")
    if profile.startswith("neck_") and profile != "neck_forward":
        paths.append(WORK / "playground/open_duck_mini_v2/locomotion_turn.py")
    if profile in ("neck_turn_tracking", "neck_reverse", "neck_reverse_tracking", "neck_reverse_progress",
                   "neck_reverse_reset", "neck_arc", "neck_arc_switches") + COMBINED_PROFILES:
        paths.append(WORK / "playground/open_duck_mini_v2/locomotion_turn_tracking.py")
    if profile in ("neck_reverse", "neck_reverse_tracking", "neck_reverse_progress", "neck_reverse_reset"):
        paths.append(WORK / "playground/open_duck_mini_v2/locomotion_reverse.py")
    if profile in ("neck_reverse_tracking", "neck_reverse_progress", "neck_reverse_reset"):
        paths.append(WORK / "playground/open_duck_mini_v2/locomotion_reverse_tracking.py")
    if profile == "neck_reverse_progress":
        paths.append(WORK / "playground/open_duck_mini_v2/locomotion_reverse_progress.py")
    if profile == "neck_reverse_reset":
        paths.append(WORK / "playground/open_duck_mini_v2/locomotion_reverse_reset.py")
    if profile in ("neck_arc", "neck_arc_switches") + COMBINED_PROFILES:
        paths.append(WORK / "playground/open_duck_mini_v2/locomotion_arc.py")
    if profile in ("neck_arc_switches",) + COMBINED_PROFILES:
        paths.append(WORK / "playground/open_duck_mini_v2/locomotion_arc_switches.py")
    if profile in COMBINED_PROFILES:
        paths.append(WORK / "playground/open_duck_mini_v2/locomotion_combined.py")
    if profile in COMBINED_PROFILES[1:]:
        paths.append(WORK / "playground/open_duck_mini_v2/locomotion_combined_slew.py")
    if profile in COMBINED_PROFILES[2:]:
        paths.append(WORK / "playground/open_duck_mini_v2/locomotion_combined_slew_stable.py")
    if profile in COMBINED_PROFILES[3:]:
        paths.append(WORK / "playground/open_duck_mini_v2/locomotion_combined_balance.py")
    if profile in COMBINED_PROFILES[4:]:
        paths.append(WORK / "playground/open_duck_mini_v2/locomotion_reversal_pairs.py")
    if profile == "neck_combined_snapshot_reset":
        paths += [WORK / "playground/open_duck_mini_v2/locomotion_snapshot_reset.py",
                  WORK / "playground/open_duck_mini_v2/data/balance_reverse_snapshots.npz"]
    paths += sorted((WORK / "playground/open_duck_mini_v2/xmls").rglob("*.xml"))
    return paths


def environment_source_file(root, relative):
    """Find a recorded source by its stable, pre-organization name."""
    relative = Path(relative)
    source = root / relative
    package = Path("duck_training/playground/open_duck_mini_v2")
    if source.is_file() or relative.parent != package or not relative.name.startswith("locomotion_"):
        return source
    current = root / package / "profiles" / relative.name
    if current.is_file():
        return current
    return root / "history" / "environment_profiles" / relative.name


def source_manifest(profile):
    paths = environment_source_paths(profile)
    return {str(path.relative_to(ROOT)): hashlib.sha256(environment_source_file(ROOT, path.relative_to(ROOT)).read_bytes()).hexdigest()
            for path in paths}


def validate_sources(recorded, root=None):
    root = ROOT if root is None else root
    profile = recorded.get("task_profile", "upstream")
    if profile != "upstream":
        required = {str(path.relative_to(ROOT)) for path in environment_source_paths(profile)}
        missing = required - recorded.get("source_sha256", {}).keys()
        if missing:
            raise RuntimeError(f"新版任务缺少完整环境/参考数据版本清单：{sorted(missing)}")
    for relative, expected in recorded.get("source_sha256", {}).items():
        path = environment_source_file(root, relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"训练环境源码已变化，禁止静默换任务评价：{relative}；请恢复 run/sources 中的对应版本后再比较")


def warm_start_metadata(checkpoint, network):
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir() or checkpoint.parent.name != "checkpoints":
        raise ValueError("warm-start 必须指向已有 run/checkpoints/ 内的模型目录")
    source_run = checkpoint.parent.parent
    recorded = json.loads((source_run / "config.json").read_text(encoding="utf-8"))
    if network_from_config(recorded) != network:
        raise ValueError("接续训练的策略/价值网络结构或观测键与源模型不一致")
    environment_settings(recorded)
    validate_sources(recorded)
    name = checkpoint.name
    if name == "initial":
        step = 0
    elif name.startswith("step_") and name[5:].isdigit():
        step = int(name[5:])
    else:
        raise ValueError("无法从 checkpoint 名称确定已训练步数")
    return dict(checkpoint=str(checkpoint), source_run=str(source_run),
        source_task_profile=recorded.get("task_profile", "upstream"),
        prior_environment_steps=int(recorded.get("prior_environment_steps", 0)) + step,
        optimizer_state_restored=False)


def training_environment_source(task_profile, checkpoint):
    if checkpoint is None:
        return task_profile or "upstream", None
    recorded = json.loads((checkpoint.parent.parent / "config.json").read_text(encoding="utf-8"))
    source_profile = environment_settings(recorded)[0]
    selected = task_profile or source_profile
    if selected == "neck_combined_balance" and source_profile != selected:
        raise ValueError("For balance inputs migrate the Actor and its normalizer first")
    if selected in COMBINED_PROFILES[4:] and source_profile not in COMBINED_PROFILES[3:]:
        raise ValueError("This continuation expects a checkpoint trained with the balance inputs")
    if (source_profile in COMBINED_PROFILES and selected in COMBINED_PROFILES
            and COMBINED_PROFILES.index(source_profile) < COMBINED_PROFILES.index(selected)):
        return selected, dict(recorded, task_profile=selected)
    return selected, recorded if selected == source_profile else None


def finite_state(state):
    values = (state.obs, state.data.qpos, state.data.qvel, state.data.ctrl,
              state.reward, state.done, state.info["current_reference_motion"])
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(values)):
        raise FloatingPointError("MJX 状态、参考运动、观测或奖励出现非有限值")


def smoke(env):
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    state = reset(jax.random.PRNGKey(0))
    finite_state(state)
    shapes = {key: list(value.shape) for key, value in state.obs.items()}
    assert shapes == {"state": [101], "privileged_state": [212]}, shapes
    assert env.action_size == 14 and env.n_substeps == 10 and np.isclose(env.dt, 0.02)
    assert joystick.USE_IMITATION_REWARD, "模仿奖励开关必须开启"
    assert state.info["current_reference_motion"].shape == (40,)
    assert np.isclose(env.PRM.fps, 1 / env.dt), "参考帧率与控制帧率不一致"
    before = float(state.data.time)
    state = step(state, jp.zeros(env.action_size))
    finite_state(state)
    elapsed = float(state.data.time) - before
    assert np.isclose(elapsed, 0.02, atol=1e-6), elapsed
    reference = WORK / "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
    return {"gpu_jit": True, "devices": [str(x) for x in jax.devices()],
            "observation_shapes": shapes, "action_size": env.action_size,
            "control_dt": elapsed, "imitation_enabled": True,
            "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            "allocator_bytes_limit": jax.devices()[0].memory_stats().get("bytes_limit"),
            "reward_after_zero_action": float(state.reward)}


PRESETS = {
    # 教学档：小网络、每批只更新一次，用来跑通链路。
    "lesson": dict(policy_hidden=(64, 64), value_hidden=(64, 64), num_updates_per_batch=1,
                   num_minibatches=4, num_eval_envs=1, unroll_length=10),
    # 接续档的历史名称official保留兼容；不是作者Berkeley预设（512/256/128）。
    "official": dict(policy_hidden=(128, 128, 128, 128), value_hidden=(256, 256, 256, 256, 256),
                     num_updates_per_batch=4, num_minibatches=8, num_eval_envs=8, unroll_length=20),
}


PRESETS["transfer"] = PRESETS["official"]


def ppo_settings(envs, evals=1, unroll=None, preset="lesson", seed=0):
    profile = PRESETS[preset]
    minibatches = profile["num_minibatches"]
    return dict(episode_length=1000, action_repeat=1, num_envs=envs,
                batch_size=envs // minibatches, num_minibatches=minibatches,
                unroll_length=unroll or profile["unroll_length"],
                num_updates_per_batch=profile["num_updates_per_batch"], num_evals=evals,
                num_eval_envs=profile["num_eval_envs"],
                max_devices_per_host=1, normalize_observations=True,
                reward_scaling=1.0, learning_rate=3e-4, discounting=0.97,
                entropy_cost=0.005, clipping_epsilon=0.2, max_grad_norm=1.0, seed=seed)


def network_of(preset):
    profile = PRESETS[preset]
    return dict(policy_hidden=list(profile["policy_hidden"]),
                value_hidden=list(profile["value_hidden"]),
                policy_obs_key="state", value_obs_key="privileged_state")


def network_from_config(config):
    """兼容旧 run：只有 hidden_layers 时，策略和价值共用同一组宽度。"""
    recorded = config.get("network", {})
    if "policy_hidden" in recorded:
        return recorded
    hidden = recorded.get("hidden_layers", [64, 64])
    return dict(policy_hidden=hidden, value_hidden=hidden,
                policy_obs_key=recorded.get("policy_obs_key", "state"),
                value_obs_key=recorded.get("value_obs_key", "privileged_state"))


def ppo_kwargs(settings, network):
    from functools import partial
    from brax.training.distribution import NormalTanhDistribution

    settings = dict(settings)
    exploration_scale = settings.pop("exploration_scale", 1.)
    factory = partial(ppo_networks.make_ppo_networks,
                      policy_hidden_layer_sizes=tuple(network["policy_hidden"]),
                      value_hidden_layer_sizes=tuple(network["value_hidden"]),
                      policy_obs_key=network["policy_obs_key"],
                      value_obs_key=network["value_obs_key"])

    def make_networks(observation_size, action_size, preprocess_observations_fn):
        models = factory(observation_size, action_size, preprocess_observations_fn=preprocess_observations_fn)
        if exploration_scale != 1.:
            return models.replace(parametric_action_distribution=NormalTanhDistribution(
                event_size=action_size, var_scale=exploration_scale))
        return models

    return dict(settings, network_factory=make_networks, wrap_env_fn=wrapper.wrap_for_brax_training,
                randomization_fn=None)


def latest_checkpoint(run, recorded=None):
    """训练可能仍在进行：优先用 config 记录的最终 checkpoint，否则取已写出的最大 step。"""
    if recorded and (run / recorded).is_dir():
        return (run / recorded).resolve()
    steps = sorted((int(p.name.split("_")[1]), p) for p in (run / "checkpoints").glob("step_*")
                   if p.is_dir() and p.name.split("_")[1].isdigit())
    if not steps:
        raise FileNotFoundError(f"{run} 下还没有 step_* checkpoint，训练可能尚未写出第一段")
    return steps[-1][1].resolve()


def restore(env, settings, checkpoint, network):
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"找不到原生 checkpoint：{checkpoint}")
    return ppo.train(environment=env, num_timesteps=0,
                     restore_checkpoint_path=str(checkpoint.resolve()),
                     **ppo_kwargs(settings, network))[:2]


def deterministic(make_policy, params):
    # 保留整个观测字典：Brax 先归一化整棵树，再选 state / privileged_state。
    return jax.jit(make_policy(params, deterministic=True))


def evaluate(env, make_policy, parameters, target):
    """固定种子；每步保存真实姿态、命令、动作和奖励，不把短训称为会走。"""
    target.mkdir(parents=True, exist_ok=False)
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    summary = {"purpose": "deterministic physical metrics; inspect commands and failure, not return alone", "episodes": []}
    for label, params in parameters.items():
        policy = deterministic(make_policy, params)
        for seed in (11, 22):
            state = reset(jax.random.PRNGKey(seed))
            command = np.asarray(state.info["command"]).copy()
            trace = {key: [] for key in ("qpos", "qvel", "command", "action", "reward", "done", "time", "local_velocity", "gyro", "up_vector")}
            for _ in range(1000):
                # 上游在 step 末尾可能重采 command；记录本步奖励实际使用的命令。
                command_before = np.asarray(state.info["command"]).copy()
                action, _ = policy(state.obs, jax.random.PRNGKey(0))
                state = step(state, action)
                finite_state(state)
                values = (state.data.qpos, state.data.qvel, command_before, action,
                          state.reward, state.done, state.data.time, env.get_local_linvel(state.data),
                          env.get_gyro(state.data), env.get_gravity(state.data))
                for key, value in zip(trace, values):
                    trace[key].append(np.asarray(value).copy())
                if bool(state.done):
                    break
            name = f"{label}_seed_{seed}.npz"
            velocity, commands, gyro = (np.asarray(trace[key]) for key in ("local_velocity", "command", "gyro"))
            np.savez_compressed(target / name, **{k: np.asarray(v) for k, v in trace.items()})
            summary["episodes"].append(dict(policy=label, seed=seed, initial_command=command,
                steps=len(trace["reward"]), episode_return=float(np.sum(trace["reward"])),
                terminated=bool(state.done), final_qpos=np.asarray(state.data.qpos), trajectory=name,
                duration_s=len(trace["reward"])*env.dt, mean_local_velocity=velocity.mean(0),
                rms_planar_velocity_error=float(np.sqrt(np.mean(np.sum((velocity[:,:2]-commands[:,:2])**2, axis=1)))),
                rms_yaw_error=float(np.sqrt(np.mean((gyro[:,2]-commands[:,2])**2))),
                max_tilt_degrees=float(np.degrees(np.arccos(np.clip(np.asarray(trace["up_vector"])[:,2], -1, 1))).max())))
    write_json(target / "summary.json", summary)
    return summary


def train(env, args, run, checks, config):
    settings = config["ppo"]
    network = config["network"]
    checkpoints = run / "checkpoints"
    checkpoints.mkdir()
    checkpointer = ocp.PyTreeCheckpointer()
    initialization = ({"restore_checkpoint_path": str(args.warm_start)} if args.warm_start else {})
    if initialization:
        print(f"WARM_START: {args.warm_start}; network+normalizer restored; optimizer reset", flush=True)
    print("PHASE: initialize networks and save initial checkpoint", flush=True)
    make_policy, initial, _ = ppo.train(environment=env, num_timesteps=0,
                                        **ppo_kwargs(settings, network), **initialization)
    # 0.12.1 返回三元组；callback / restore 使用 normalizer + PPONetworkParams 二元组。
    initial_native = (initial[0], ppo_losses.PPONetworkParams(policy=initial[1], value=initial[2]))
    checkpointer.save(str(checkpoints / "initial"), initial_native)
    if args.warm_start:
        _, source_parameters = restore(env, settings, args.warm_start, network)
        assert jax.tree_util.tree_structure(initial) == jax.tree_util.tree_structure(source_parameters)
        differences = [float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))
                       for left, right in zip(jax.tree_util.tree_leaves(initial), jax.tree_util.tree_leaves(source_parameters))]
        maximum = max(differences, default=0.)
        assert np.isfinite(maximum) and maximum == 0., "接续训练的初始网络/归一化器与源检查点不一致"
        checks["warm_start"] = dict(config["warm_start"], initial_max_parameter_difference=maximum)
        write_json(run / "checks.json", checks)
        print(f"WARM_START_VERIFIED: source and initial network/value/normalizer max_difference={maximum}", flush=True)
    last_checkpoint = []

    def save(step, unused_make_policy, params):
        path = checkpoints / f"step_{int(step)}"
        checkpointer.save(str(path), params)
        last_checkpoint[:] = [path]
        print(f"Checkpoint: {path}", flush=True)

    def progress(step, metrics):
        row = {"step": int(step), "metrics": metrics}
        with (run / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False,
                                   default=lambda x: np.asarray(x).tolist()) + "\n")
        print(f"PPO step={step}: {metrics}", flush=True)

    print(f"PHASE: PPO learning, requested environment steps={args.steps}", flush=True)
    anchor = nullcontext()
    if args.imitation_anchor:
        from ppo_demonstration_anchor import anchor_loss, load_demonstrations
        datasets, provenance = load_demonstrations(args.imitation_anchor, float(env._config.reverse_cadence))
        config['imitation_anchor'] = dict(data=provenance, weight=args.imitation_weight)
        write_json(run / 'config.json', config)
        shutil.copy2(Path(__file__).with_name('ppo_demonstration_anchor.py'), run / 'sources/ppo_demonstration_anchor.py')
        anchor = anchor_loss(ppo_losses, datasets, args.imitation_weight)
    with anchor:
        make_policy, final, metrics = ppo.train(environment=env, num_timesteps=args.steps,
            policy_params_fn=save, progress_fn=progress, **ppo_kwargs(settings, network), **initialization)
    assert last_checkpoint, "训练未保存最终 callback 参数"
    before, after = jax.tree_util.tree_leaves(initial[1]), jax.tree_util.tree_leaves(final[1])
    assert jax.tree_util.tree_structure(initial[1]) == jax.tree_util.tree_structure(final[1])
    weight_l2 = float(np.sqrt(sum(np.square(np.asarray(b, dtype=np.float64)
                                          - np.asarray(a, dtype=np.float64)).sum()
                                 for a, b in zip(before, after))))
    assert np.isfinite(weight_l2) and weight_l2 > 0, "策略权重没有发生有限的更新"
    probe = jax.jit(env.reset)(jax.random.PRNGKey(314))
    np.savez_compressed(run / "probe_observation.npz", **{k: np.asarray(v) for k, v in probe.obs.items()})
    initial_action = deterministic(make_policy, initial)(probe.obs, jax.random.PRNGKey(0))[0]
    final_action = deterministic(make_policy, final)(probe.obs, jax.random.PRNGKey(0))[0]
    print("PHASE: reload checkpoint and compare deterministic actions", flush=True)
    loaded_make, loaded = restore(env, settings, last_checkpoint[0], network)
    loaded_action = deterministic(loaded_make, loaded)(probe.obs, jax.random.PRNGKey(0))[0]
    assert np.allclose(final_action, loaded_action, rtol=1e-6, atol=1e-6), "重载后的动作不一致"
    action_l2 = float(np.linalg.norm(np.asarray(final_action - initial_action)))
    assert np.isfinite(action_l2) and action_l2 > 0, "固定观测上的策略动作没有变化"
    checks["training"] = dict(policy_weight_l2=weight_l2, probe_action_l2=action_l2,
        initial_action=np.asarray(initial_action), final_action=np.asarray(final_action),
        restore_max_action_error=float(np.max(np.abs(np.asarray(final_action - loaded_action)))),
        final_metrics=metrics)
    config["final_checkpoint"] = str(last_checkpoint[0].relative_to(run))
    config["cumulative_environment_steps"] = int(config.get("prior_environment_steps", 0)) + int(last_checkpoint[0].name[5:])
    write_json(run / "config.json", config)
    write_json(run / "checks.json", checks)
    if args.skip_rollouts:
        checks["evaluation"] = {"skipped": True, "reason": "capacity/update/reload check only; no walking-quality evidence"}
    else:
        print("PHASE: fixed-seed initial/trained evaluation", flush=True)
        checks["evaluation"] = evaluate(env, loaded_make, {"initial": initial, "trained": loaded}, run / "evaluation")


def view(env, make_policy, parameters):
    """MJX 执行策略与动力学；CPU MjData 只用于画面，不做 mj_step。"""
    import mujoco.viewer
    events = queue.SimpleQueue()
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    policies = {name: deterministic(make_policy, p) for name, p in parameters.items()}
    state = reset(jax.random.PRNGKey(11))
    print(f"本体速度命令 vx/vy/wz={np.asarray(state.info['command'])[:3]}", flush=True)
    label, paused, frames = "trained", True, 0
    spec = mujoco.MjSpec.from_file(str(WORK / "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"))
    spec.modelname = "Lesson 07 - PPO policy - 7 initial 8 trained"
    display_model = spec.compile()
    assert (display_model.nq, display_model.nv, display_model.nu) == (21, 20, 14)
    data = mujoco.MjData(display_model)
    print("真实策略闭环：7 初始 / 8 训练后 / R 重置 / Space 暂停。默认暂停；行走质量以评测为准。", flush=True)
    with mujoco.viewer.launch_passive(display_model, data, key_callback=events.put,
                                    show_left_ui=False, show_right_ui=False) as handle:
        with handle.lock():
            handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            handle.cam.trackbodyid = display_model.body("trunk_assembly").id
            handle.cam.lookat[:] = [-0.015, 0, 0.20]
            handle.cam.distance, handle.cam.azimuth, handle.cam.elevation = 0.78, 40, -15
            handle.opt.geomgroup[2], handle.opt.geomgroup[3] = 1, 0
        while handle.is_running():
            started = time.perf_counter()
            while not events.empty():
                key = events.get()
                if key in (55, 56, 82, 114):
                    if key in (55, 56):
                        label = "initial" if key == 55 else "trained"
                    state = reset(jax.random.PRNGKey(11))
                    print(f"本体速度命令 vx/vy/wz={np.asarray(state.info['command'])[:3]}", flush=True)
                    paused, frames = False, 0
                elif key == 32:
                    paused = not paused
                print(f"策略={label}, paused={paused}", flush=True)
            if not paused and frames < 1000 and not bool(state.done):
                action, _ = policies[label](state.obs, jax.random.PRNGKey(0))
                state = step(state, action)
                frames += 1
                finite_state(state)
                if bool(state.done) or frames == 1000:
                    paused = True
                    reason = "环境失败" if bool(state.done) else "1000步时间上限"
                    print(f"本局结束（{reason}）；已暂停，按 R 重置并开始。", flush=True)
            with handle.lock():
                data.qpos[:] = np.asarray(state.data.qpos)
                data.qvel[:] = np.asarray(state.data.qvel)
                data.ctrl[:] = np.asarray(state.data.ctrl)
                data.time = float(state.data.time)
                mujoco.mj_forward(display_model, data)
            handle.sync()
            time.sleep(max(0.0, 0.02 - (time.perf_counter() - started)))


def main(argv=None):
    args = parse_args(argv)
    mode = "train" if args.train else "evaluate" if args.evaluate else "view" if args.view else "smoke"
    args.task_profile, training_source = training_environment_source(args.task_profile, args.warm_start)
    if args.turn_yaw_error_multiplier is not None:
        if args.task_profile != 'neck_turn_tracking' or training_source is None:
            raise ValueError('Turn penalty override requires a saved neck_turn_tracking environment')
        training_source['environment']['turn_yaw_error_multiplier'] = args.turn_yaw_error_multiplier
    if args.reverse_probability is not None:
        if training_source is None or 'reverse_probability' not in training_source['environment']:
            raise ValueError('Reverse frequency override requires a saved reverse curriculum environment')
        training_source['environment']['reverse_probability'] = args.reverse_probability
    if args.reverse_speed_min is not None:
        if training_source is None or 'reverse_speed' not in training_source['environment']:
            raise ValueError('Reverse speed override requires a saved reverse curriculum environment')
        _, upper = training_source['environment']['reverse_speed']
        if args.reverse_speed_min >= upper:
            raise ValueError('The minimum reverse speed must be below the saved maximum')
        training_source['environment']['reverse_speed'] = [args.reverse_speed_min, upper]
    warm_metadata = warm_start_metadata(args.warm_start, network_of(args.preset)) if args.warm_start else None
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run, report, config = None, None, None
    if mode in ("evaluate", "view"):
        run = (args.run or Path(LATEST.read_text(encoding="utf-8").strip())).resolve()
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if mode == "evaluate":
            report = run / "evaluations" / unique_name("eval_")
            report.mkdir(parents=True)
    else:
        run = OUTPUT / unique_name("run_")
        run.mkdir()
        report = run
        sources = run / "sources"
        sources.mkdir()
        for source in (Path(__file__), ROOT / "requirements-duck.lock",
                       ROOT / "requirements-duck.freeze.txt", WORK / "SOURCE.json"):
            if source.is_file():
                shutil.copy2(source, sources / source.name)
        print(f"RUN_DIRECTORY: {run}", flush=True)
    checks = {"mode": mode, "status": "running"}
    try:
        versions = load_runtime()
        if config is not None:
            validate_sources(config)
            if not config.get("source_sha256"):
                print("历史 run 无环境源码哈希；本次恢复保存的参数，但无法证明 Python 奖励实现与训练时完全一致。", flush=True)
        env = make_environment(config if config is not None else training_source,
                               task_profile=args.task_profile, command=args.command)
        if mode in ("evaluate", "view"):
            disable_assisted_resets(env)
        print("PHASE: GPU and environment smoke check", flush=True)
        checks["smoke"] = smoke(env)
        if mode in ("smoke", "train"):
            config = dict(source_commit="b9be205ac64488c23504ca42e5ec790337adeec3", versions=versions,
                purpose="pipeline smoke; no walking-quality claim", requested_steps=args.steps,
                task="flat_terrain", environment=env._config.to_dict(),
                task_profile=args.task_profile, source_sha256=source_manifest(args.task_profile),
                preset=args.preset,
                ppo=ppo_settings(args.envs, args.evals, args.unroll, args.preset, seed=args.seed),
                network=network_of(args.preset),
                parameter_domain_randomization=False, reset_noise_push_imitation="see saved environment and task_profile",
                checkpoint_scope="network + normalizer; optimizer and environment state are not restored")
            if args.warm_start:
                config["warm_start"] = warm_metadata
                config["prior_environment_steps"] = config["warm_start"]["prior_environment_steps"]
                shutil.copy2(Path(config["warm_start"]["source_run"]) / "config.json", run / "sources/warm_start_config.json")
            if args.learning_rate is not None:
                config['ppo']['learning_rate'] = args.learning_rate
            if args.exploration_scale is not None:
                config['ppo']['exploration_scale'] = args.exploration_scale
            write_json(run / "config.json", config)
            for relative in config["source_sha256"]:
                target = run / "sources" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(environment_source_file(ROOT, relative), target)
        if mode == "train":
            train(env, args, run, checks, config)
        elif mode in ("evaluate", "view"):
            # 训练还在跑时也能看：退回使用已经写出的最新 step_* checkpoint。
            final_path = ((run / "checkpoints" / args.checkpoint).resolve() if args.checkpoint
                          else latest_checkpoint(run, config.get("final_checkpoint")))
            if not final_path.is_relative_to(run):
                raise ValueError("checkpoint 必须位于所选 run 目录内")
            print(f"使用 checkpoint：{final_path.relative_to(run)}", flush=True)
            network = network_from_config(config)
            make_policy, initial = restore(env, config["ppo"], run / "checkpoints/initial", network)
            make_policy, trained = restore(env, config["ppo"], final_path, network)
            parameters = {"initial": initial, "trained": trained}
            if mode == "view":
                view(env, make_policy, parameters)
            else:
                checks["evaluation"] = evaluate(env, make_policy, parameters, report / "trajectories")
        checks["status"] = "passed"
        if report:
            write_json(report / "checks.json", checks)
        if mode == "train" and not args.skip_rollouts:
            LATEST.write_text(str(run) + "\n", encoding="utf-8")
        print(f"{mode}: passed; run={run}", flush=True)
    except Exception as error:
        checks.update(status="failed", error=f"{type(error).__name__}: {error}")
        if report:
            write_json(report / "checks.json", checks)
        raise


if __name__ == "__main__":
    main()
