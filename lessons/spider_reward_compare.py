"""从同一热启动比较基础速度奖励和包含中心线正向奖励的直走奖励。"""

import argparse
import json
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter

from brax.io import model as brax_model
from brax.training.agents.ppo import losses, train
import jax
import numpy as np
from orbax import checkpoint as ocp

from spider_diagnose_walk import render_comparison
from spider_full_env import CENTERLINE_CORRIDOR, CENTERLINE_REWARD_WIDTH, SpiderFullEnv, WEIGHTS
from spider_mjx_env import SpiderMJXEnv
from spider_ppo import OUTPUT, actor_parameter_change, training_settings
from spider_prm_compare import load_reference, make_evaluator
from spider_train_full import DEFAULT_EPISODE_STEPS, checkpoint_schedule, prepare_initial, write_json
from spider_walk import TARGET_SPEED


def train_policy(env, settings, run, target_steps, initial_path, warm):
    mode = env.reward_mode
    saved = {}
    save_seconds = 0.0

    def progress(step, metrics):
        row = {"step": int(step), "metrics": {name: np.asarray(value).tolist() for name, value in metrics.items()}}
        with (run / f"{mode}_progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(f"{mode}: step={step:,} train_sps={metrics.get('training/sps', 0):.0f}", flush=True)

    def save(step, _, params):
        nonlocal save_seconds
        started = perf_counter()
        path = run / f"{mode}_{step}.params"
        brax_model.save_params(str(path), (params[0], params[1].policy, params[1].value))
        saved[int(step)] = path.name
        save_seconds += perf_counter() - started

    start = perf_counter()
    make_policy, final, metrics = train.train(
        **settings, environment=env, num_timesteps=target_steps, restore_checkpoint_path=str(initial_path),
        progress_fn=progress, policy_params_fn=save,
    )
    if max(saved, default=0) != target_steps or actor_parameter_change(warm[1], final[1]) <= 0:
        raise RuntimeError(f"{mode}: PPO did not complete {target_steps} steps")
    timing = dict(
        total_seconds=perf_counter() - start, training_seconds=float(metrics["training/walltime"]),
        brax_eval_seconds=float(metrics["eval/walltime"]), save_seconds=save_seconds,
    )
    return make_policy, final, saved, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("warm_start", type=Path)
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int)
    args = parser.parse_args()
    if (not args.warm_start.is_dir() or args.envs <= 0 or args.steps <= 0
            or (args.save_every is not None and args.save_every <= 0)):
        parser.error("Need an existing warm start and positive environments, steps and save interval")

    baseline = SpiderMJXEnv()
    reference, contact = load_reference(baseline)
    forward = SpiderFullEnv(reference, reward_mode="forward")
    straight = SpiderFullEnv(reference, reward_mode="straight")
    settings = training_settings(forward, args.envs, DEFAULT_EPISODE_STEPS, args.seed)
    minibatches = 8 if args.envs % 8 == 0 else 1
    target_steps, num_evals = checkpoint_schedule(args.steps, args.envs * settings["unroll_length"], args.save_every)
    settings.update(batch_size=args.envs // minibatches, num_minibatches=minibatches, num_evals=num_evals,
                    learning_rate=1e-4)
    warm = prepare_initial(baseline, forward, settings, args.warm_start.resolve(), args.seed)
    del settings["environment"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix="reward-", dir=OUTPUT))
    initial_path = run / "initial"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(str(initial_path), (warm[0], losses.PPONetworkParams(policy=warm[1], value=warm[2])))
    report = dict(
        requested_steps=args.steps, training_steps_per_policy=target_steps, envs=args.envs, seed=args.seed,
        train_episode_steps=DEFAULT_EPISODE_STEPS, control_dt=forward.dt, start_y_m=0,
        warm_start=str(args.warm_start.resolve()), demonstration_loss_weight=0,
        target_forward_speed_m_s=TARGET_SPEED, straight_weights={**WEIGHTS, "centerline": 1.0},
        centerline_width_m=CENTERLINE_REWARD_WIDTH, centerline_corridor_m=CENTERLINE_CORRIDOR,
        batch_size=settings["batch_size"], unroll_length=settings["unroll_length"], num_minibatches=minibatches,
        profiles={},
    )
    write_json(run / "config.json", report)
    print(f"Run: {run}; device={jax.devices()[0]}; {args.envs} envs, {target_steps:,} steps each", flush=True)
    outcomes = {}
    for env in (forward, straight):
        make_policy, params, saved, timing = train_policy(env, settings, run, target_steps, initial_path, warm)
        outcomes[env.reward_mode] = params
        report["profiles"][env.reward_mode] = dict(checkpoints=saved, timing=timing)
        write_json(run / "comparison.json", report)

    eval_start = perf_counter()
    evaluate = make_evaluator(forward, make_policy, reference, contact, args.seed, include_lateral_kicks=False,
                              horizon_steps=DEFAULT_EPISODE_STEPS)
    clips = {}
    for mode, params in outcomes.items():
        rows, clips[mode] = evaluate(params, return_replay=True)
        report["profiles"][mode]["from_home_6s"] = rows[0]
    evaluate_long = make_evaluator(forward, make_policy, reference, contact, args.seed, include_lateral_kicks=False,
                                   horizon_steps=2 * DEFAULT_EPISODE_STEPS)
    for mode, params in outcomes.items():
        report["profiles"][mode]["from_home_12s"] = evaluate_long(params)[0]
    report["comparison_eval_seconds"] = perf_counter() - eval_start
    write_json(run / "comparison.json", report)
    replay = {name: np.stack([clip[name] for clip in clips.values()]) for name in clips["forward"]}
    render_start = perf_counter()
    render_comparison(forward.mj_model, replay, run / "walking_comparison.gif", ("A  Forward speed", "B  Straight rewards"))
    report["render_seconds"] = perf_counter() - render_start
    write_json(run / "comparison.json", report)
    print(f"Results: {run / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
