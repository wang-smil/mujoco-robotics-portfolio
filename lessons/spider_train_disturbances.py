"""在配对延时和传感器噪声下复测并续训已有 A/B 策略。"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from tempfile import mkdtemp

from brax.io import model as brax_model
from brax.training.agents.ppo import losses, networks
import jax
import jax.numpy as jp
import numpy as np

from ppo_demonstration_anchor import anchor_loss
from spider_command_curriculum import CURRICULUM, validation_routes
from spider_command_disturbance import MAX_DELAY_TICKS, NOISE_STD
from spider_command_env import SpiderCommandEnv
from spider_command_eval import make_native_command_evaluator
from spider_command_gait import build_command_gait
from spider_ppo import OUTPUT, training_settings
from spider_reward_compare import train_policy
from spider_standing import build_model
from spider_train_commands import (
    DEMONSTRATION_LOSS_WEIGHT, EPISODE_STEPS, POLICIES, capped_exploration_network, make_native_controller,
    validate_checkpoint,
)
from spider_train_full import write_json
from spider_train_speed import save_initial
from spider_walk import CONTROL_SUBSTEPS


SOURCE_RUN = OUTPUT / "command-ab-gutptkn_"
START_STEP = 524_288
EVAL_CASES = {
    "clean": dict(action_delay=0, sensor_delay=0, noise_seed=None),
    "noise_only": dict(action_delay=0, sensor_delay=0, noise_seed=131),
    "delay_40ms_only": dict(action_delay=2, sensor_delay=2, noise_seed=None),
    "noise_delay_20ms": dict(action_delay=1, sensor_delay=1, noise_seed=131),
    "noise_delay_40ms": dict(action_delay=2, sensor_delay=2, noise_seed=131),
}


def evaluate_with_disturbances(params, evaluate, routes, condition):
    def replay(policy, cases):
        return evaluate(policy, cases, **condition)

    return validate_checkpoint(params, replay, routes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    parser.add_argument("--start-step", type=int, default=START_STEP)
    parser.add_argument("--steps", type=int, default=524_288, help="PPO continuation steps per arm")
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if (args.envs < 8 or args.envs % 8 or args.steps < 16 * args.envs or args.steps % (16 * args.envs)):
        parser.error("Use an environment count divisible by eight and steps divisible by two PPO batches")
    source = args.source_run.resolve()
    if not (source / "comparison.json").is_file():
        parser.error("Source run must have an A/B comparison report")
    source_report = json.loads((source / "comparison.json").read_text(encoding="utf-8"))
    if not all(str(args.start_step) in source_report["policies"][arm] for arm in POLICIES):
        parser.error("Both source arms must have the requested starting checkpoint")

    model = build_model()
    observe, advance = make_native_controller(model)
    gait = build_command_gait(model)
    profiles = CURRICULUM["mixed"]
    routes = validation_routes()
    network = capped_exploration_network(61, model.nu)
    make_policy = networks.make_inference_fn(network)
    _, evaluate = make_native_command_evaluator(make_policy, observe=observe, advance=advance,
                                                seed=args.seed, model=model)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run = args.resume.resolve() if args.resume else Path(mkdtemp(prefix="command-disturbance-", dir=OUTPUT))
    report_path = run / "comparison.json"
    configuration = dict(source_run=str(source), start_step=args.start_step, continuation_steps=args.steps,
                         envs=args.envs, seed=args.seed, control_dt=model.opt.timestep * CONTROL_SUBSTEPS,
                         training_episode_steps=EPISODE_STEPS, train_delay_ticks=[0, MAX_DELAY_TICKS],
                         noise_standard_deviation=NOISE_STD, eval_conditions=EVAL_CASES, learning_rate=2e-5)
    if args.resume:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["configuration"] != configuration:
            parser.error("Resume settings must match the original run")
    else:
        report = dict(configuration=configuration, policies={arm: {} for arm in POLICIES})
        write_json(report_path, report)
    print(f"Run: {run}; {args.envs:,} environments; {args.steps:,} additional steps per arm", flush=True)

    with np.load(source / "demonstrations.npz", allow_pickle=False) as dataset:
        demo = (jp.asarray(dataset["observations"]), jp.asarray(dataset["actions"]))
    for arm in POLICIES:
        results = report["policies"][arm]
        initial = source / source_report["policies"][arm][str(args.start_step)]["checkpoint"]
        warm = brax_model.load_params(str(initial))
        results["source_checkpoint"] = str(initial)
        if "checkpoints" not in results:
            stage = run / arm
            stage.mkdir(exist_ok=True)
            restore = stage / "initial"
            if not restore.exists():
                save_initial(restore, warm)
            template = gait if arm == "rich" else None
            env = SpiderCommandEnv(profiles, EPISODE_STEPS, 10, reference=template, disturbances=True)
            settings = training_settings(env, args.envs, EPISODE_STEPS, args.seed)
            settings.update(network_factory=capped_exploration_network, batch_size=args.envs // 8, num_minibatches=8,
                            num_evals=3, learning_rate=configuration["learning_rate"], discounting=.97,
                            entropy_cost=.003, clipping_epsilon=.20, max_grad_norm=1.0, num_updates_per_batch=2)
            del settings["environment"]
            guard = (anchor_loss(losses, [demo], DEMONSTRATION_LOSS_WEIGHT, format_observation=lambda obs: obs)
                     if arm == "rich" else nullcontext())
            print(f"Training {arm} under latency and noise", flush=True)
            with guard:
                _, _, saved, timing = train_policy(env, settings, stage, args.steps, restore, warm)
            results["checkpoints"] = {str(step): str(Path(arm) / path) for step, path in saved.items()}
            results["timing"] = timing
            write_json(report_path, report)
            jax.clear_caches()

    for arm in POLICIES:
        results = report["policies"][arm]
        params_by_phase = {
            "before": brax_model.load_params(results["source_checkpoint"]),
            "after": brax_model.load_params(str(run / results["checkpoints"][str(args.steps)])),
        }
        for phase, params in params_by_phase.items():
            evaluation = results.setdefault(phase, {})
            for label, condition in EVAL_CASES.items():
                if label not in evaluation:
                    evaluation[label] = evaluate_with_disturbances(params, evaluate, routes, condition)
                    write_json(report_path, report)
                summary = evaluation[label]["summary"]
                print(f"{arm}/{phase}/{label}: {summary['passed_cases']}/{summary['routes']} pass, "
                      f"{summary['complete_cases']} complete", flush=True)
    print(f"Results: {report_path}", flush=True)


if __name__ == "__main__":
    main()
