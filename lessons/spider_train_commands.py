"""运行基础与完整奖励的三轴指令课程，并逐检查点做原生 MuJoCo 对照。"""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from tempfile import mkdtemp

from brax.io import model as brax_model
from brax.training.agents.ppo import losses, networks, train
from brax.training.distribution import NormalTanhDistribution
import jax
import jax.numpy as jp
import mujoco
import numpy as np

from ppo_demonstration_anchor import anchor_loss
from spider_command_curriculum import CURRICULUM, DEMONSTRATOR_PROFILES, validation_routes
from spider_command_env import RICH_WEIGHTS, SpiderCommandEnv
from spider_command_eval import aggregate_command_routes, make_native_command_evaluator, plot_command_progress
from spider_command_gait import build_command_gait
from spider_command_task import COMMAND_SLEW, advance_command, command_authority, observation
from spider_command_warm import fit_warm_actor
from spider_diagnose_walk import render_comparison
from spider_ppo import OUTPUT, training_settings
from spider_reward_compare import train_policy
from spider_standing import build_model, floor_contact_geoms
from spider_train_full import extend_input, write_json
from spider_train_speed import save_initial
from spider_walk import ACTION_LIMIT, CONTROL_SUBSTEPS, FALL_TILT_DEGREES, GROUPS, PERIOD, read_observation


EPISODE_STEPS = 400
DEFAULT_STAGE_STEPS = 1_048_576
POLICIES = ("base", "rich")
DEMONSTRATION_LOSS_WEIGHT = .70
BC_FEATURES = (50, 51, 54, 57, 58)  # 步态相位和实际下发的三轴指令


def capped_exploration_network(observation_size, action_size, *args, **kwargs):
    model = networks.make_ppo_networks(observation_size, action_size, *args, **kwargs, policy_hidden_layer_sizes=(32, 32),
                                       value_hidden_layer_sizes=(32, 32))
    return model.replace(parametric_action_distribution=NormalTanhDistribution(
        event_size=action_size, var_scale=.25,
    ))


def make_native_controller(model):
    feet = [model.geom(name + "_foot").id for group in GROUPS for name in group]
    joints = model.actuator_trnid[:, 0]
    qpos_ids = model.jnt_qposadr[joints]
    qvel_ids = model.jnt_dofadr[joints]
    floor = model.geom("floor").id
    home = model.key("home").ctrl.copy()
    dt = model.opt.timestep * CONTROL_SUBSTEPS

    def observe(data, requested, applied, target_yaw):
        contacts = floor_contact_geoms(data, floor)
        phase = (round(data.time / dt) * dt / PERIOD) % 1.0
        base = read_observation(data, qpos_ids, qvel_ids, feet, contacts, phase)
        rotation = data.body("torso").xmat.reshape(3, 3)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        return observation(base, yaw, requested, applied, target_yaw, np)

    def advance(data, action, requested, applied):
        applied = advance_command(applied, requested, dt, np)
        return home + ACTION_LIMIT * np.clip(action, -1., 1.) * command_authority(applied, np), applied

    return observe, advance


def collect_demonstrations(model, gait):
    observe, advance = make_native_controller(model)
    dt = model.opt.timestep * CONTROL_SUBSTEPS
    observations, actions = [], []
    floor, shell = model.geom("floor").id, model.geom("torso_shell").id
    for profile in DEMONSTRATOR_PROFILES:
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        applied = np.zeros(3)
        heading = 0.
        for tick, requested in enumerate(np.repeat(profile, EPISODE_STEPS // 4, axis=0)):
            obs = observe(data, requested, applied, heading)
            next_command = advance_command(applied, requested, dt, np)
            authority = command_authority(next_command, np)
            template = gait.normalized_action(next_command, tick * dt / PERIOD, np)
            action = np.clip(template / max(authority, 1e-6), -.99, .99)
            data.ctrl[:], applied = advance(data, action, requested, applied)
            mujoco.mj_step(model, data, nstep=CONTROL_SUBSTEPS)
            mujoco.mj_forward(model, data)
            heading += applied[2] * dt
            up = data.body("torso").xmat[8]
            if not np.isfinite(obs).all() or up < np.cos(np.radians(FALL_TILT_DEGREES)):
                raise RuntimeError(f"Demonstration failed: {profile}, step {tick}")
            if shell in floor_contact_geoms(data, floor):
                raise RuntimeError(f"Demonstrator hit the ground: {profile}, step {tick}")
            observations.append(obs)
            actions.append(action)
    return np.asarray(observations, np.float32), np.asarray(actions, np.float32)


def validate_checkpoint(params, evaluate, routes):
    results, _ = evaluate(params, routes)
    return {"summary": aggregate_command_routes(results), "routes": results}


def mask_warm_actor(actor, obs_size):
    mask = np.zeros(obs_size, dtype=np.float32)
    mask[list(BC_FEATURES)] = 1.0
    layers = actor["params"]
    first = layers["hidden_0"]
    weights = first["kernel"] * jp.asarray(mask[:, None])
    return {**actor, "params": {**layers, "hidden_0": {**first, "kernel": weights}}}, mask


def finish_report(run, report, evaluate, model, routes):
    series = {name: [dict(step=int(step), **result["summary"]) for step, result in entries.items()]
              for name, entries in report["policies"].items()}
    plot_command_progress(series, run / "training_progress.png")
    clips = []
    route = {"arc_left": routes["arc_left"]}
    for name in POLICIES:
        entries = report["policies"][name]
        selected = max(entries, key=lambda step: (
            entries[step]["summary"]["passed_cases"], entries[step]["summary"]["complete_cases"],
            -(entries[step]["summary"]["max_heading_error_degrees"] or 0),
        ))
        report.setdefault("best_steps", {})[name] = int(selected)
        final = max(map(int, entries))
        report.setdefault("final_steps", {})[name] = final
        params = brax_model.load_params(str(run / entries[str(final)]["checkpoint"]))
        _, traces = evaluate(params, route)
        clips.append(traces["arc_left"])
    count = max(len(clip["time"]) for clip in clips)
    if count < 2 or any(not len(clip["time"]) for clip in clips):
        return
    replay = {}
    for field in ("time", "qpos", "heading", "requested", "measured"):
        replay[field] = np.stack([clip[field][np.minimum(np.arange(count), len(clip[field]) - 1)] for clip in clips])
    render_comparison(model, replay, run / "arc_comparison.gif", ("A  Tracking only", "B  Tracking + imitation"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("warm_checkpoint", type=Path)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--stage-steps", type=int, default=DEFAULT_STAGE_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, help="Resume an interrupted experiment after its last saved stage")
    args = parser.parse_args()
    batch_steps = args.envs * 8
    if (not args.warm_checkpoint.is_file() or args.envs < 8 or args.envs % 8 or args.stage_steps < 2 * batch_steps
            or args.stage_steps % (2 * batch_steps) or args.stage_steps * 6 > 150_000_000):
        parser.error("Provide a checkpoint and a step budget aligned to two PPO batches per stage, within the total cap")

    native = build_model()
    template = build_command_gait(native)
    first = SpiderCommandEnv(next(iter(CURRICULUM.values())), EPISODE_STEPS, 10)
    settings = training_settings(first, args.envs, EPISODE_STEPS, args.seed)
    settings.update(network_factory=capped_exploration_network, batch_size=args.envs // 8, num_minibatches=8,
                    num_evals=3, learning_rate=3e-5, discounting=.97, entropy_cost=.003,
                    clipping_epsilon=.20, max_grad_norm=1.0, num_updates_per_batch=2)
    makes_policy, initialized, _ = train.train(
        num_timesteps=0, **{**settings, "num_envs": 1, "batch_size": 1, "num_minibatches": 1, "num_evals": 1},
    )
    observe, advance = make_native_controller(native)
    _, evaluate = make_native_command_evaluator(makes_policy, observe=observe, advance=advance, seed=args.seed, model=native)
    routes = validation_routes()
    del settings["environment"]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    run = args.resume.resolve() if args.resume else Path(mkdtemp(prefix="command-ab-", dir=OUTPUT))
    report_path = run / "comparison.json"
    if args.resume:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report["envs"], report["stage_steps"], report["seed"], report["source"]) != (
                args.envs, args.stage_steps, args.seed, str(args.warm_checkpoint.resolve())):
            parser.error("Resume requires the original experiment settings and source")
        with np.load(run / "demonstrations.npz", allow_pickle=False) as demo:
            dataset = (demo["observations"], demo["actions"])
    else:
        baseline = brax_model.load_params(str(args.warm_checkpoint))
        actor = baseline[1]
        for _ in range(first.observation_size - actor["params"]["hidden_0"]["kernel"].shape[0]):
            actor = extend_input(actor)
        actor, mask = mask_warm_actor(actor, first.observation_size)
        initial = (initialized[0], actor, initialized[2])
        dataset = collect_demonstrations(native, template)
        np.savez_compressed(run / "demonstrations.npz", observations=dataset[0], actions=dataset[1])
        print(f"Run: {run}; {jax.devices()[0]}; {args.envs:,} envs in each arm; {len(dataset[0]):,} demonstration samples",
              flush=True)
        shared, bc_report = fit_warm_actor(initial, makes_policy, capped_exploration_network,
                                           dataset[0] * mask, dataset[1], seed=args.seed, steps=7000)
        brax_model.save_params(str(run / "shared_warm.params"), shared)
        at_zero = {"checkpoint": "shared_warm.params", **validate_checkpoint(shared, evaluate, routes)}
        report = dict(source=str(args.warm_checkpoint.resolve()), envs=args.envs, stage_steps=args.stage_steps,
                      seed=args.seed, episode_steps=EPISODE_STEPS, dt=first.dt, observation_size=first.observation_size,
                      train_solver_line_search=10, native_eval_solver_line_search=int(native.opt.ls_iterations),
                      warm_fit=bc_report, warm_features=BC_FEATURES, command_slew=COMMAND_SLEW,
                      training_profiles=CURRICULUM, eval_routes=routes,
                      tracking_weights=(1.2, .7), rich_reward_weights=RICH_WEIGHTS,
                      rich_demonstration_loss_weight=DEMONSTRATION_LOSS_WEIGHT,
                      ppo_settings={name: settings[name] for name in ("learning_rate", "num_updates_per_batch",
                                                                     "num_minibatches", "unroll_length", "discounting")},
                      policies={name: {"0": at_zero} for name in POLICIES})
        write_json(report_path, report)

    report["velocity_measurement"] = "body-frame displacement and heading change per control interval"
    write_json(report_path, report)
    demonstration_batch = tuple(jp.asarray(array) for array in dataset)
    for index, (stage, profiles) in enumerate(CURRICULUM.items()):
        offset = index * args.stage_steps
        for name in POLICIES:
            entries = report["policies"][name]
            if str(offset + args.stage_steps) in entries:
                continue
            warm = brax_model.load_params(str(run / entries[str(offset)]["checkpoint"]))
            stage_dir = run / f"{name}_{stage}"
            stage_dir.mkdir(exist_ok=True)
            initial_path = stage_dir / "initial"
            if not initial_path.exists():
                save_initial(initial_path, warm)
            env = SpiderCommandEnv(profiles, EPISODE_STEPS, 10, reference=template if name == "rich" else None)
            print(f"Training {name}/{stage} from step {offset:,}", flush=True)
            guard = (anchor_loss(losses, [demonstration_batch], DEMONSTRATION_LOSS_WEIGHT, format_observation=lambda obs: obs)
                     if name == "rich" else nullcontext())
            with guard:
                _, _, checkpoints, timing = train_policy(env, settings, stage_dir, args.stage_steps, initial_path, warm)
            for step, checkpoint in checkpoints.items():
                relative = Path(stage_dir.name) / checkpoint
                params = brax_model.load_params(str(run / relative))
                result = validate_checkpoint(params, evaluate, routes)
                entries[str(offset + step)] = dict(checkpoint=str(relative), stage=stage, **result)
                print(f"{name} {offset + step:,}: {result['summary']['passed_cases']}/{len(routes)} routes pass, "
                      f"{result['summary']['complete_cases']} complete", flush=True)
                write_json(report_path, report)
            report.setdefault("timing", {})[stage_dir.name] = timing
            write_json(report_path, report)
            jax.clear_caches()

    finish_report(run, report, evaluate, native, routes)
    write_json(report_path, report)
    print(f"Results: {report_path}", flush=True)


if __name__ == "__main__":
    main()
