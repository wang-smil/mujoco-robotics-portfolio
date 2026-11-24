"""把小蜘蛛的 MJX 环境接入 Brax PPO，验收训练和策略重载。"""

import argparse
from functools import partial
from pathlib import Path
from tempfile import mkdtemp

from brax.io import model as brax_model
from brax.training.agents.ppo import losses, networks, train
import jax
import numpy as np
from mujoco_playground import wrapper
from orbax import checkpoint as ocp

from spider_mjx_env import SpiderMJXEnv


OUTPUT = Path(__file__).resolve().parents[1] / "artifacts" / "spider_ppo"
UNROLL_LENGTH = 8
NUM_MINIBATCHES = 2


def actor_parameter_change(initial, trained):
    before = jax.tree_util.tree_leaves(initial)
    after = jax.tree_util.tree_leaves(trained)
    return max(float(np.max(np.abs(np.asarray(new) - np.asarray(old)))) for old, new in zip(before, after))


def training_settings(env, num_envs, episode_length, seed):
    return {
        "environment": env,
        "wrap_env_fn": wrapper.wrap_for_brax_training,
        "episode_length": episode_length,
        "action_repeat": 1,
        "num_envs": num_envs,
        "num_eval_envs": 1,
        "max_devices_per_host": 1,
        "unroll_length": UNROLL_LENGTH,
        "batch_size": num_envs,
        "num_minibatches": NUM_MINIBATCHES,
        "num_updates_per_batch": 1,
        "num_evals": 1,
        "seed": seed,
        "learning_rate": 3e-4,
        "deterministic_eval": True,
        "network_factory": partial(
            networks.make_ppo_networks,
            policy_hidden_layer_sizes=(32, 32),
            value_hidden_layer_sizes=(32, 32),
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=4, help="Number of parallel MJX environments")
    parser.add_argument("--steps", type=int, help="Requested rollout steps across all environments")
    parser.add_argument("--episode-length", type=int, default=60, help="Time limit in 20 ms control steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrain", action="store_true", help="Warm-start the Actor with the 50 Hz demonstration")
    args = parser.parse_args()
    if args.envs < 1 or args.episode_length < 1 or (args.steps is not None and args.steps < 1):
        parser.error("envs, steps and episode-length must be positive")
    batch_steps = args.envs * UNROLL_LENGTH * NUM_MINIBATCHES
    requested_steps = args.steps if args.steps is not None else batch_steps

    env = SpiderMJXEnv()
    settings = training_settings(env, args.envs, args.episode_length, args.seed)
    trained_steps = 0

    def record_steps(steps, _make_policy, _params):
        nonlocal trained_steps
        trained_steps = int(steps)

    print(
        f"device={jax.devices()[0]}  envs={args.envs}  requested_steps={requested_steps}"
        f"  control_dt={env.dt:.3f} s",
        flush=True,
    )
    # 两次使用同样的种子和网络，0 步的 Actor 就是这轮 PPO 的起点。
    initial_policy, initial, _ = train.train(num_timesteps=0, **settings)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix="run-", dir=OUTPUT))
    key = jax.random.PRNGKey(args.seed)
    if args.pretrain:
        from spider_ppo_pretrain import make_takeover_check, pretrain_actor

        ppo_networks = settings["network_factory"](env.observation_size, env.action_size)
        warmed = pretrain_actor(env, ppo_networks, initial)
        check_takeover = make_takeover_check(env, initial_policy, key)
        check_takeover("pretrained", warmed)
        pretrain_checkpoint = run / "pretrain"
        warm_start = (warmed[0], losses.PPONetworkParams(policy=warmed[1], value=warmed[2]))
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(str(pretrain_checkpoint), warm_start)
        settings["restore_checkpoint_path"] = str(pretrain_checkpoint)
        _, initial, _ = train.train(num_timesteps=0, **settings)
        if actor_parameter_change(warmed[1], initial[1]) != 0:
            raise RuntimeError("Brax did not restore the pretrained Actor")
        print(f"PPO restored pretrained Actor: {pretrain_checkpoint}", flush=True)
    make_policy, params, metrics = train.train(
        num_timesteps=requested_steps,
        policy_params_fn=record_steps,
        **settings,
    )
    change = actor_parameter_change(initial[1], params[1])
    if trained_steps < requested_steps or not np.isfinite(change) or change <= 0:
        raise RuntimeError("PPO did not complete an Actor update")
    print(f"collected_steps={trained_steps}  actor_max_parameter_change={change:.6g}")
    for name in ("training/policy_loss", "training/v_loss", "eval/episode_reward"):
        if name in metrics:
            value = float(metrics[name])
            if not np.isfinite(value):
                raise RuntimeError(f"Non-finite PPO metric: {name}")
            print(f"{name}={value:.6g}")

    if args.pretrain:
        check_takeover("PPO", params)
    checkpoint = run / "policy.params"
    brax_model.save_params(str(checkpoint), params)
    restored = brax_model.load_params(str(checkpoint))
    state = env.reset(key)
    action, _ = make_policy(params, deterministic=True)(state.obs, key)
    reloaded_action, _ = make_policy(restored, deterministic=True)(state.obs, key)
    np.testing.assert_allclose(reloaded_action, action, atol=1e-7)
    if reloaded_action.shape != (env.action_size,) or not np.isfinite(reloaded_action).all():
        raise RuntimeError("Reloaded Actor returned an invalid action")
    step = jax.jit(env.step)(state, reloaded_action)
    np.testing.assert_allclose(step.data.time, env.dt, atol=1e-8)
    print(f"checkpoint={checkpoint}")
    print(f"reload_action={reloaded_action.shape}  t={float(step.data.time):.3f} s")
    print("ppo: PASS")


if __name__ == "__main__":
    main()
