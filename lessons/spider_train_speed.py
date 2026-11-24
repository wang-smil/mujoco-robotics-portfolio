"""在基础直走 PPO 上逐阶段训练调速、后退和前后切换。"""

import argparse
from pathlib import Path
from tempfile import mkdtemp

from brax.io import model as brax_model
from brax.training.agents.ppo import losses, train
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from orbax import checkpoint as ocp

from spider_diagnose_walk import render_comparison
from spider_full_env import Y_OBSERVATION_SCALE
from spider_ppo import OUTPUT, training_settings
from spider_reward_compare import train_policy
from spider_speed_command import COMMAND_SLEW_M_S2, forward_speed, scale_action, slew_speed
from spider_speed_env import SpiderSpeedEnv
from spider_standing import build_model, floor_contact_geoms, torso_tilt_degrees
from spider_train_full import DEFAULT_EPISODE_STEPS, checkpoint_schedule, extend_input, write_json
from spider_walk import CONTROL_SUBSTEPS, FALL_TILT_DEGREES, GROUPS, TARGET_SPEED, read_observation


SPEED_PROFILES = ((0.,) * 4, (.018,) * 4, (.034,) * 4, (.046,) * 4,
                  (0., .018, .046, 0.), (.034, .018, .046, 0.))
REVERSE_PROFILES = ((-.018,) * 4, (-.024,) * 4, (-.034,) * 4, (0., -.018, -.034, 0.))
SWITCH_PROFILES = ((.034, -.024, .034, 0.), (-.024, .034, -.024, 0.), (.018, -.034, .046, 0.),
                   (.046, -.018, .034, 0.))
STAGES = {
    "speed": SPEED_PROFILES,
    "reverse": (*SPEED_PROFILES, *REVERSE_PROFILES),
    "switch": (*SPEED_PROFILES, *REVERSE_PROFILES, *SWITCH_PROFILES, *SWITCH_PROFILES),
}
ROUTES = {
    "speed_changes": (0., .018, .046, 0.),
    "nominal_to_slow": (.034, .018, .034, 0.),
    "reverse_changes": (0., -.018, -.034, 0.),
    "forward_reverse": (.034, -.024, .034, 0.),
    "reverse_forward": (-.024, .034, -.024, 0.),
}
STAGE_ROUTES = {"speed": 2, "reverse": 3, "switch": 5}
SECTION_STEPS = 150
MAX_HEADING_DEGREES = 5.0


def save_initial(path, params):
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(str(path), (params[0], losses.PPONetworkParams(policy=params[1], value=params[2])))


def make_native_evaluator(make_policy, env, seed, section_steps=SECTION_STEPS):
    if section_steps < env.cycle_steps:
        raise ValueError("Each segment must include at least one gait cycle")
    model = build_model()
    data = mujoco.MjData(model)
    torso = data.body("torso")
    floor = model.geom("floor").id
    shell = model.geom("torso_shell").id
    feet = [model.geom(name + "_foot").id for group in GROUPS for name in group]
    joint_ids = model.actuator_trnid[:, 0]
    qpos = model.jnt_qposadr[joint_ids]
    qvel = model.jnt_dofadr[joint_ids]
    hips = np.asarray(env.hips)
    nominal = np.asarray(env.home_ctrl)
    key = jax.random.PRNGKey(seed)

    @jax.jit
    def act(params, obs):
        return make_policy(params, deterministic=True)(obs, key)[0]

    def evaluate(params, route):
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        mujoco.mj_forward(model, data)
        contacts = floor_contact_geoms(data, floor)
        applied = 0.0
        traces = {k: [] for k in ("time", "qpos", "heading", "command", "speed", "support", "tilt", "failed")}
        for tick, request in enumerate(np.repeat(route, section_steps)):
            obs = read_observation(data, qpos, qvel, feet, contacts, (tick % env.cycle_steps) / env.cycle_steps)
            obs = np.append(obs, [np.clip(data.qpos[1] / Y_OBSERVATION_SCALE, -10, 10),
                                  request / TARGET_SPEED, applied / TARGET_SPEED])
            action = np.asarray(act(params, obs.astype(np.float32)))
            applied = slew_speed(applied, request, env.dt, np)
            data.ctrl[:] = nominal + env.action_scale * scale_action(action, hips, applied, np)
            before = data.qpos[:2].copy()
            shell_contact = False
            for _ in range(CONTROL_SUBSTEPS):
                mujoco.mj_step(model, data)
                shell_contact |= shell in floor_contact_geoms(data, floor)
            mujoco.mj_forward(model, data)
            contacts = floor_contact_geoms(data, floor)
            rotation = torso.xmat.reshape(3, 3)
            tilt = torso_tilt_degrees(torso)
            speed = forward_speed(data.qpos[:2] - before, rotation[:2, 0], env.dt, np)
            failed = shell_contact or shell in contacts or tilt > FALL_TILT_DEGREES or not np.isfinite(speed)
            values = (data.time, data.qpos.copy(), np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])),
                      request, speed, len(contacts.intersection(feet)), tilt, failed)
            for field, value in zip(traces, values):
                traces[field].append(value)
            if failed:
                break
        trace = {name: np.asarray(values) for name, values in traces.items()}
        completed = not bool(trace["failed"].any())
        segments = []
        for index, request in enumerate(route):
            start = (index + 1) * section_steps - env.cycle_steps
            end = (index + 1) * section_steps
            settled = trace["speed"][start:end]
            measured = float(settled.mean()) if len(settled) == env.cycle_steps and np.isfinite(settled).all() else None
            tolerance = .006 if request == 0 else .012
            segments.append(dict(target_m_s=request, mean_m_s=measured,
                                 passed=measured is not None and abs(measured - request) <= tolerance))
        max_heading = float(np.max(np.abs(trace["heading"])))
        support = trace["support"][min(25, len(trace["support"]) - 1):]
        supported = np.flatnonzero(np.r_[True, support > 0, True])
        longest_unsupported = int(max(np.diff(supported)) - 1)
        result = dict(completed=completed, passed=completed and max_heading <= MAX_HEADING_DEGREES
                      and all(row["passed"] for row in segments),
                      segments=segments, endpoint_x_m=float(data.qpos[0]), endpoint_y_m=float(data.qpos[1]),
                      max_heading_degrees=max_heading, min_support_after_settle=int(support.min()),
                      max_unsupported_control_steps=longest_unsupported)
        return result, trace

    return model, evaluate


def score(routes):
    return (sum(not route["passed"] for route in routes.values()),
            max(route["max_unsupported_control_steps"] for route in routes.values()),
            max(route["max_heading_degrees"] for route in routes.values()), sum(
        abs(row["mean_m_s"] - row["target_m_s"]) if row["mean_m_s"] is not None else 1.0
        for route in routes.values() for row in route["segments"]
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forward_checkpoint", type=Path)
    parser.add_argument("--resume-switch", action="store_true", help="Use a 55-D checkpoint to repeat only switch training")
    parser.add_argument("--envs", type=int, default=1024)
    parser.add_argument("--steps-per-stage", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--line-search-iterations", type=int, default=10)
    args = parser.parse_args()
    if (not args.forward_checkpoint.is_file() or min(args.envs, args.steps_per_stage, args.line_search_iterations) < 1):
        parser.error("Need an existing forward checkpoint and positive env, step and iteration counts")
    profiles = {"switch": STAGES["switch"]} if args.resume_switch else STAGES
    stage_envs = {name: SpiderSpeedEnv(commands, DEFAULT_EPISODE_STEPS, args.line_search_iterations)
                  for name, commands in profiles.items()}
    first = next(iter(stage_envs.values()))
    settings = training_settings(first, args.envs, DEFAULT_EPISODE_STEPS, args.seed)
    target_steps, num_evals = checkpoint_schedule(args.steps_per_stage, args.envs * settings["unroll_length"])
    minibatches = 8 if args.envs % 8 == 0 else 1
    settings.update(batch_size=args.envs // minibatches, num_minibatches=minibatches,
                    num_evals=num_evals, learning_rate=1e-4)
    make_policy, initialized, _ = train.train(
        num_timesteps=0, **{**settings, "num_envs": 1, "batch_size": 1, "num_minibatches": 1, "num_evals": 1},
    )
    baseline = brax_model.load_params(str(args.forward_checkpoint))
    expected_input = first.observation_size if args.resume_switch else first.observation_size - 2
    if any(params["params"]["hidden_0"]["kernel"].shape[0] != expected_input for params in baseline[1:]):
        parser.error(f"This run requires a {expected_input}-dimensional checkpoint")
    warm = baseline if args.resume_switch else (initialized[0], *(extend_input(extend_input(params)) for params in baseline[1:]))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix="speed-command-", dir=OUTPUT))
    report = dict(source=str(args.forward_checkpoint.resolve()), obs_size=first.observation_size, envs=args.envs,
                  train_episode_steps=DEFAULT_EPISODE_STEPS, dt=first.dt, training_steps_per_stage=target_steps,
                  seed=args.seed, train_line_search_iterations=args.line_search_iterations,
                  command_slew_m_s2=COMMAND_SLEW_M_S2, training_profiles=profiles, stages={})
    model, evaluate = make_native_evaluator(make_policy, first, args.seed)
    report["evaluation_line_search_iterations"] = model.opt.ls_iterations
    write_json(run / "comparison.json", report)
    del settings["environment"]
    print(f"Run: {run}; {jax.devices()[0]}; {target_steps:,} steps per stage", flush=True)
    for name, env in stage_envs.items():
        stage_dir = run / name
        stage_dir.mkdir()
        initial_path = stage_dir / "initial"
        save_initial(initial_path, warm)
        stage_settings = {**settings, "learning_rate": 1e-5 if name == "switch" else settings["learning_rate"]}
        print(f"Training stage: {name}, learning rate={stage_settings['learning_rate']:g}", flush=True)
        _, _, saved, timing = train_policy(env, stage_settings, stage_dir, target_steps, initial_path, warm)
        routes = list(ROUTES.items())[:STAGE_ROUTES[name]]
        before = {route_name: evaluate(warm, route)[0] for route_name, route in routes}
        candidates = {}
        for step, checkpoint in saved.items():
            params = brax_model.load_params(str(stage_dir / checkpoint))
            results = {route_name: evaluate(params, route)[0] for route_name, route in routes}
            candidates[step] = dict(checkpoint=checkpoint, routes=results)
        best_trained = min(candidates, key=lambda step: score(candidates[step]["routes"]))
        best = best_trained if score(candidates[best_trained]["routes"]) <= score(before) else 0
        best_results = candidates[best]["routes"] if best else before
        if best:
            best_path = stage_dir / saved[best]
        else:
            best_path = stage_dir / "initial.params"
            brax_model.save_params(str(best_path), warm)
        warm = brax_model.load_params(str(best_path))
        report["stages"][name] = dict(before=before, trained=candidates, selected=str(best_path),
                                      selected_step=best, selected_routes=best_results, best_trained_step=best_trained,
                                      learning_rate=stage_settings["learning_rate"], timing=timing)
        write_json(run / "comparison.json", report)
        print(f"{name}: selected {best:,}; routes passed {sum(result['passed'] for result in best_results.values())}/{len(best_results)}",
              flush=True)

    playback = [evaluate(warm, ROUTES[name])[1] for name in ("speed_changes", "reverse_changes", "forward_reverse")]
    common_frames = min(len(trace["time"]) for trace in playback)
    replay = {field: np.stack([trace[field][:common_frames] for trace in playback]) for field in playback[0]}
    render_comparison(model, replay, run / "speed_comparison.gif", ("Change speed", "Backward", "Forward / backward"))
    print(f"Results: {run / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
