"""用完整奖励和专家示范监督训练小蜘蛛，并复评每个检查点。"""

import argparse
import json
from pathlib import Path
from tempfile import mkdtemp

from brax.io import model as brax_model
from brax.training.agents.ppo import losses, train
import jax
import jax.numpy as jp
import numpy as np
from orbax import checkpoint as ocp

from ppo_demonstration_anchor import anchor_loss
from spider_diagnose_walk import render_comparison
from spider_full_env import SpiderFullEnv, WEIGHTS, Y_OBSERVATION_SCALE
from spider_mjx_env import SpiderMJXEnv
from spider_ppo import OUTPUT, actor_parameter_change, training_settings
from spider_ppo_pretrain import DEMONSTRATION
from spider_prm_compare import load_reference, make_evaluator
from spider_walk import TARGET_SPEED


DEFAULT_EPISODE_STEPS = 300


def checkpoint_schedule(requested_steps, batch_steps, save_every=None):
    if save_every is None:
        save_every = 1_000_000 if requested_steps >= 5_000_000 else 100_000
    saves = min((requested_steps + save_every - 1) // save_every, max(1, requested_steps // batch_steps))
    updates_per_save = max(1, requested_steps // (saves * batch_steps))
    return saves * updates_per_save * batch_steps, saves + 1 if saves > 1 else 1


def extend_input(params):
    layers = params["params"]
    first = layers["hidden_0"]
    weights = jp.pad(first["kernel"], ((0, 1), (0, 0)))
    return {**params, "params": {**layers, "hidden_0": {**first, "kernel": weights}}}


def load_demo(env):
    with np.load(DEMONSTRATION, allow_pickle=False) as demo:
        obs = demo["obs"].astype(np.float32)
        action = demo["action"].astype(np.float32) / env.action_scale
    if obs.shape[1:] != (52,) or action.shape != (len(obs), env.action_size):
        raise ValueError("Incompatible demonstration dimensions")
    if not np.isfinite(obs).all() or not np.isfinite(action).all() or np.max(np.abs(action)) > 1:
        raise ValueError("Invalid expert observation or normalized action")
    # 单一直走示范的标签对应中心线，横移初态由 PPO 自己体验。
    return jp.asarray(np.pad(obs, ((0, 0), (0, 1)))), jp.asarray(action)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rank(rows):
    failures = sum(not row["completed"] for row in rows)
    cost = np.mean([
        row["speed_mae_m_s"] / TARGET_SPEED
        + row["centerline_mae_m"] / 0.015
        + row["stance_contact_slip_m_s"] / 0.05
        for row in rows
    ])
    return failures, float(cost)


def prepare_initial(native_env, full_env, settings, source, seed):
    old_settings = training_settings(native_env, 1, settings["episode_length"], seed)
    old_infer, previous, _ = train.train(
        num_timesteps=0, restore_checkpoint_path=str(source), **old_settings,
    )
    init_settings = {**settings, "num_envs": 1, "batch_size": 1, "num_minibatches": 1}
    infer, initialized, _ = train.train(num_timesteps=0, **init_settings)
    warm = (initialized[0], extend_input(previous[1]), extend_input(previous[2]))
    key = jax.random.PRNGKey(seed)
    old_obs = native_env.reset(key).obs
    new_obs = jp.concatenate((old_obs, jp.zeros(1)))
    previous_action, _ = old_infer(previous, deterministic=True)(old_obs, key)
    extended_action, _ = infer(warm, deterministic=True)(new_obs, key)
    np.testing.assert_allclose(extended_action, previous_action, atol=1e-7)
    if warm[1]["params"]["hidden_0"]["kernel"].shape[0] != full_env.observation_size:
        raise RuntimeError("Warm-start input width does not match the environment")
    return warm


def check_environment(env):
    key = jax.random.PRNGKey(0)
    reset = jax.jit(env.reset_at_y_offset)
    centered = reset(key, 0.0)
    offset = reset(key, 0.015)
    assert centered.obs.shape == (53,)
    np.testing.assert_allclose(centered.obs[-1], 0, atol=1e-7)
    np.testing.assert_allclose(offset.obs[-1], .015 / Y_OBSERVATION_SCALE, atol=1e-7)
    step = jax.jit(env.step)
    mid = step(centered, jp.zeros(env.action_size))
    side = step(offset, jp.zeros(env.action_size))
    np.testing.assert_allclose(mid.metrics["alive_bonus"], WEIGHTS["alive"])
    np.testing.assert_allclose(mid.metrics["centerline_cost"], 0, atol=1e-7)
    expected_cost = env.centerline_weight * np.expm1(-1)
    np.testing.assert_allclose(side.metrics["centerline_cost"], expected_cost, atol=1e-5)
    np.testing.assert_allclose(side.reward - mid.reward, expected_cost, atol=1e-5)
    fallen = centered.replace(data=centered.data.replace(qpos=centered.data.qpos.at[2].set(.01)))
    terminal = step(fallen, jp.zeros(env.action_size))
    assert float(terminal.done) == 1 and float(terminal.metrics["alive_bonus"]) == 0
    print("Centerline observation, cost and alive reward: PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("warm_start", type=Path, help="Existing 52-D behavior-cloning Orbax checkpoint")
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=5_000_000)
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument("--evals", type=int, help="Number of evaluations including the initial policy")
    schedule.add_argument("--save-every", type=int, help="Approximate spacing in control steps between checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=DEFAULT_EPISODE_STEPS,
                        help="Episode and evaluation limit in 20 ms control steps")
    parser.add_argument("--batch-size", type=int, help="PPO minibatch size")
    parser.add_argument("--centerline-weight", type=float, default=WEIGHTS["centerline"])
    args = parser.parse_args()
    if (not args.warm_start.is_dir() or min(args.envs, args.steps, args.episode_length) < 1
            or (args.evals is not None and args.evals < 1)
            or (args.save_every is not None and args.save_every < 1)):
        parser.error("Need an existing warm start and positive counts and intervals")
    if not 0 <= args.centerline_weight <= 1:
        parser.error("Centerline weight must be in [0, 1]")
    minibatches = 8 if args.envs % 8 == 0 else 1
    batch_size = args.envs // minibatches if args.batch_size is None else args.batch_size
    if batch_size < 1 or batch_size * minibatches % args.envs:
        parser.error("Minibatch size times number of minibatches must be a positive multiple of envs")

    baseline = SpiderMJXEnv()
    qpos_reference, contact_reference = load_reference(baseline)
    env = SpiderFullEnv(qpos_reference, start_offset=.015, centerline_weight=args.centerline_weight)
    evaluation_env = SpiderFullEnv(qpos_reference, centerline_weight=args.centerline_weight)
    check_environment(evaluation_env)
    settings = training_settings(env, args.envs, args.episode_length, args.seed)
    batch_steps = batch_size * minibatches * settings["unroll_length"]
    if args.evals is None:
        target_steps, num_evals = checkpoint_schedule(args.steps, batch_steps, args.save_every)
    else:
        target_steps, num_evals = args.steps, args.evals
    settings.update(
        batch_size=batch_size,
        num_minibatches=minibatches,
        num_evals=num_evals,
        eval_env=evaluation_env,
        learning_rate=1e-4,
    )

    source = args.warm_start.resolve()
    warm = prepare_initial(baseline, env, settings, source, args.seed)
    demo = load_demo(env)
    imitation_weight = 2.0
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix="full-", dir=OUTPUT))
    checkpoints = run / "checkpoints"
    checkpoints.mkdir()
    initial_path = checkpoints / "initial"
    report = dict(
        requested_steps=args.steps, target_steps=target_steps, envs=args.envs, seed=args.seed, evals=num_evals,
        episode_steps=settings["episode_length"], episode_seconds=settings["episode_length"] * env.dt,
        control_dt=env.dt,
        unroll_length=settings["unroll_length"], batch_size=settings["batch_size"],
        num_minibatches=minibatches, learning_rate=settings["learning_rate"],
        initial_y_offset_max_m=.015, target_forward_speed_m_s=TARGET_SPEED,
        reward_weights={**WEIGHTS, "centerline": args.centerline_weight}, imitation_loss_weight=imitation_weight,
        demonstration=str(DEMONSTRATION), demonstration_samples=int(demo[0].shape[0]),
        warm_start=str(source),
    )
    write_json(run / "config.json", report)
    print(f"Run directory: {run}", flush=True)
    print(
        f"Device: {jax.devices()[0]}; envs={args.envs}; control steps={target_steps}; "
        f"episode={args.episode_length} steps ({args.episode_length * env.dt:.2f} s)",
        flush=True,
    )
    saved = {}

    def progress(step, metrics):
        row = dict(step=int(step), metrics={name: np.asarray(value).tolist() for name, value in metrics.items()})
        with (run / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        tracked = ("eval/episode_reward", "training/demonstration_mse", "training/sps")
        values = " ".join(f"{name}={float(metrics[name]):.5g}" for name in tracked if name in metrics)
        print(f"step={step} {values}", flush=True)

    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(str(initial_path), (warm[0], losses.PPONetworkParams(policy=warm[1], value=warm[2])))
        brax_model.save_params(str(run / "warm_start.params"), warm)

        def save(step, _, params):
            checkpointer.save(str(checkpoints / f"step_{step}"), params)
            path = run / f"step_{step}.params"
            brax_model.save_params(str(path), (params[0], params[1].policy, params[1].value))
            saved[int(step)] = path
            print(f"Checkpoint: {path}", flush=True)

        with anchor_loss(losses, [demo], imitation_weight, format_observation=lambda obs: obs):
            make_policy, final, metrics = train.train(
                num_timesteps=target_steps,
                restore_checkpoint_path=str(initial_path),
                progress_fn=progress,
                policy_params_fn=save,
                **settings,
            )

    if not saved or max(saved) < target_steps or actor_parameter_change(warm[1], final[1]) <= 0:
        raise RuntimeError("PPO did not complete the requested updates")
    imitation_mse = float(metrics["training/demonstration_mse"])
    if not np.isfinite(imitation_mse):
        raise RuntimeError("Non-finite demonstration loss")

    evaluate = make_evaluator(
        evaluation_env, make_policy, qpos_reference, contact_reference, args.seed,
        include_yaw_kicks=True, cross_track_offsets=(-.015, .015),
        horizon_steps=args.episode_length,
    )
    scores = {}
    clips = {}
    for step, path in [(0, run / "warm_start.params"), *saved.items()]:
        params = brax_model.load_params(str(path))
        rows, clip = evaluate(params, return_replay=True)
        scores[step] = dict(checkpoint=str(path), score=rank(rows), evaluation=rows)
        clips[step] = clip
        print(f"evaluated step={step} failures={scores[step]['score'][0]} score={scores[step]['score'][1]:.4f}", flush=True)

    best = min(saved, key=lambda step: scores[step]["score"])
    last = max(saved)
    report.update(training_steps=last, final_demonstration_mse=imitation_mse,
                  best_trained_checkpoint_step=best, checkpoints=scores)
    write_json(run / "comparison.json", report)
    panel_steps = (0, last) if best == last else (0, best, last)
    panels = ("Warm start", f"Final PPO {last:,}") if best == last else (
        "Warm start", f"Best PPO {best:,}", f"Final PPO {last:,}",
    )
    playback = {name: np.stack([clips[step][name] for step in panel_steps]) for name in clips[0]}
    render_comparison(evaluation_env.mj_model, playback, run / "walking_comparison.gif", panels)
    print(f"Results: {run / 'comparison.json'}", flush=True)
    print("full PPO: PASS", flush=True)


if __name__ == "__main__":
    main()
