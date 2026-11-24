"""用 50 Hz 专家动作预热 Brax PPO 的 Actor，并验证接管后的步态。"""

from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import optax

from spider_walk import CYCLES, make_expert, observation_names


DEMONSTRATION = Path(__file__).resolve().parents[1] / "artifacts" / "spider_imitation" / "demonstration_50hz.npz"


def pretrain_actor(env, ppo_networks, initial):
    names = [env.mj_model.actuator(index).name for index in range(env.action_size)]
    with np.load(DEMONSTRATION, allow_pickle=False) as demo:
        obs, action = demo["obs"], demo["action"]
        np.testing.assert_array_equal(demo["actuator_names"], names)
        np.testing.assert_array_equal(demo["obs_names"], observation_names(names))
        np.testing.assert_allclose(demo["home_ctrl"], env.home_ctrl, atol=1e-7)
        np.testing.assert_allclose(demo["control_dt"], env.dt)
        np.testing.assert_allclose(demo["cycle_steps"], env.cycle_steps)
    if (
        obs.shape != (CYCLES * env.cycle_steps, env.observation_size)
        or action.shape != (len(obs), env.action_size)
        or not np.isfinite(obs).all() or not np.isfinite(action).all()
        or np.max(np.abs(action)) > env.action_scale
    ):
        raise ValueError("Invalid 50 Hz demonstration for SpiderMJXEnv")

    obs = jp.asarray(obs)
    target = jp.asarray(action / env.action_scale)
    split = len(obs) - env.cycle_steps
    # 这批专家数据只有一套固定步态。预热只初始化相位通道，其余通道留给 PPO 学习。
    train_obs = obs[:split].at[:, :-2].set(0)
    train_target = target[:split]
    normalizer, actor, critic = initial
    distribution = ppo_networks.parametric_action_distribution
    optimizer = optax.adam(0.003)

    def predict(params, observation):
        logits = ppo_networks.policy_network.apply(normalizer, params, observation)
        return distribution.mode(logits), logits

    def loss(params):
        prediction, logits = predict(params, train_obs)
        # 动作均值模仿专家，初始探索标准差保持在较小的范围。
        std = jax.nn.softplus(logits[:, env.action_size:]) + 0.001
        return jp.mean(jp.square(prediction - train_target)) + 0.05 * jp.mean(jp.square(std - 0.10))

    @jax.jit
    def fit(actor):
        def update(_, carry):
            params, optimizer_state = carry
            gradients = jax.grad(loss)(params)
            updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
            return optax.apply_updates(params, updates), optimizer_state

        return jax.lax.fori_loop(0, 2000, update, (actor, optimizer.init(actor)))[0]

    layers = actor["params"]
    first = layers["hidden_0"]
    phase_kernel = first["kernel"].at[:-2].set(0)
    phase_seed = {**actor, "params": {**layers, "hidden_0": {**first, "kernel": phase_kernel}}}
    trained = fit(phase_seed)
    before, _ = predict(actor, obs[split:])
    after, _ = predict(trained, obs[split:])
    before_rmse = float(jp.sqrt(jp.mean(jp.square(before - target[split:]))) * env.action_scale)
    after_rmse = float(jp.sqrt(jp.mean(jp.square(after - target[split:]))) * env.action_scale)
    print(f"pretrain: samples={split}  held_out={len(obs) - split}  action_rmse={before_rmse:.5f}->{after_rmse:.5f} rad", flush=True)
    if not 0 <= after_rmse < min(0.025, before_rmse):
        raise RuntimeError("Expert action fitting check failed")
    return normalizer, trained, critic


def make_takeover_check(env, make_policy, key):
    home = np.asarray(env.home_ctrl)
    expert = make_expert(env.mj_model, home)
    targets = home.copy()
    length = (1 + CYCLES) * env.cycle_steps
    warmup = np.zeros((length, env.action_size), dtype=np.float32)
    for step in range(env.cycle_steps):
        expert(targets, step / env.cycle_steps)
        warmup[step] = (targets - home) / env.action_scale
    warmup = jp.asarray(warmup)

    @jax.jit
    def replay(params):
        actor = make_policy(params, deterministic=True)

        def advance(state, index):
            action = jax.lax.cond(
                index < env.cycle_steps,
                lambda: warmup[index],
                lambda: actor(state.obs, key)[0],
            )
            state = env.step(state, action)
            return state, (state.data.qpos[:2], state.done, state.metrics["torso_tilt_deg"])

        _, trace = jax.lax.scan(advance, env.reset(key), jp.arange(length))
        return trace

    def check(label, params):
        xy, done, tilt = map(np.asarray, replay(params))
        dx, dy = xy[-1] - xy[env.cycle_steps - 1]
        max_tilt = float(tilt[env.cycle_steps:].max())
        print(f"{label} takeover: dx={dx:+.4f} m  dy={dy:+.4f} m  max_tilt={max_tilt:.2f} deg", flush=True)
        if np.any(done) or not np.isfinite([dx, dy, max_tilt]).all() or dx <= 0.05 or abs(dy) >= 0.02 or max_tilt >= 5:
            raise RuntimeError(f"{label} Actor did not pass the three-cycle takeover check")

    return check
