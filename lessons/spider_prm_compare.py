"""从同一个 Actor 预热检查点对照速度奖励和速度加参考步态奖励。"""

import argparse
import json
from pathlib import Path
from tempfile import mkdtemp

from brax.io import model as brax_model
from brax.training.agents.ppo import train
import jax
import jax.numpy as jp
import numpy as np

from spider_mjx_env import SpiderMJXEnv
from spider_ppo import OUTPUT, actor_parameter_change, training_settings
from spider_ppo_pretrain import DEMONSTRATION
from spider_walk import CYCLES, TARGET_SPEED, observation_names


def load_reference(env):
    actuators = [env.mj_model.actuator(index).name for index in range(env.action_size)]
    with np.load(DEMONSTRATION, allow_pickle=False) as demo:
        names = observation_names(actuators)
        np.testing.assert_array_equal(demo["obs_names"], names)
        np.testing.assert_array_equal(demo["actuator_names"], actuators)
        np.testing.assert_allclose(demo["control_dt"], env.dt)
        np.testing.assert_allclose(demo["home_ctrl"], env.home_ctrl, atol=1e-7)
        expected_phases = np.tile(np.arange(env.cycle_steps) / env.cycle_steps, CYCLES)
        np.testing.assert_allclose(demo["phase"], expected_phases)
        obs = demo["obs"].reshape(CYCLES, env.cycle_steps, -1)
    qpos = obs[:, :, [i for i, name in enumerate(names) if name.startswith("angle_")]].mean(axis=0)
    contact = obs[:, :, [i for i, name in enumerate(names) if name.startswith("contact_")]].mean(axis=0) >= 0.5
    return qpos, contact


def make_evaluator(
    env, make_policy, reference_qpos, reference_contact, seed, include_yaw_kicks=False, cross_track_offsets=(),
    horizon_steps=None, include_lateral_kicks=True,
):
    key = jax.random.PRNGKey(seed)
    poses, contacts = jp.asarray(reference_qpos), jp.asarray(reference_contact)
    feet = env._feet
    total_steps = CYCLES * env.cycle_steps if horizon_steps is None else horizon_steps
    if total_steps < 1:
        raise ValueError("Evaluation horizon must be positive")
    kicks = [(0.0, 0.0, 0.0)]
    if include_lateral_kicks:
        kicks.extend([(-0.05, 0.0, 0.0), (0.05, 0.0, 0.0)])
    if include_yaw_kicks:
        kicks.extend([(0.0, -0.20, 0.0), (0.0, 0.20, 0.0)])
    kicks.extend((0.0, 0.0, offset) for offset in cross_track_offsets)

    def rollout(params, kick):
        actor = make_policy(params, deterministic=True)

        def advance(state, tick):
            def step(state):
                # 一圈后给机身一个速度增量，并把对应的观测更新给 Actor。
                velocity_delta = jp.where(tick == env.cycle_steps, kick[:2], 0)
                qvel = state.data.qvel.at[1].add(velocity_delta[0]).at[5].add(velocity_delta[1])
                obs = state.obs.at[5].set(qvel[1]).at[9].set(qvel[5])
                state = state.replace(data=state.data.replace(qvel=qvel), obs=obs)
                return env.step(state, actor(obs, key)[0])

            next_state = jax.lax.cond(state.done > 0, lambda state: state, step, state)
            phase = env._phase_index(next_state.data)
            grounded = env._floor_contact(next_state.data, feet)
            stance = grounded & env._floor_contact(state.data, feet)
            foot_delta = next_state.data.geom_xpos[feet, :2] - state.data.geom_xpos[feet, :2]
            slip_speed = jp.linalg.norm(foot_delta, axis=1) / env.dt
            contact_speed_squared, _ = env.foot_contact_speed_squared(next_state.data)
            contact_speed = jp.sqrt(contact_speed_squared)
            heading_error = jp.degrees(env.heading_error(next_state.data))
            lateral_speed = (next_state.data.qpos[1] - state.data.qpos[1]) / env.dt
            metrics = (
                next_state.data.qpos[:2],
                next_state.done,
                next_state.metrics["forward_speed"],
                next_state.metrics["torso_tilt_deg"],
                jp.mean(jp.square(next_state.data.qpos[env._qpos_addresses] - poses[phase])),
                jp.mean(grounded == contacts[phase]),
                jp.sum(jp.where(stance, slip_speed, 0)),
                jp.sum(stance),
                jp.sum(jp.where(stance, contact_speed, 0)),
                jp.sum(grounded),
                lateral_speed,
                heading_error,
                env.gyro_z(next_state.data),
                next_state.data.qpos,
                next_state.data.time,
            )
            return next_state, metrics

        initial = env.reset_at_y_offset(key, kick[2]) if cross_track_offsets else env.reset(key)
        _, trace = jax.lax.scan(advance, initial, jp.arange(total_steps))
        return trace

    evaluate = jax.jit(jax.vmap(rollout, in_axes=(None, 0)))

    def summarize(params, *, return_replay=False):
        device = jax.devices()[0]
        params = jax.tree_util.tree_map(lambda value: jax.device_put(np.asarray(value), device), params)
        traces = tuple(np.asarray(value) for value in evaluate(params, jp.asarray(kicks)))
        rows = []
        for case, (lateral_kick, yaw_kick, start_y) in enumerate(kicks):
            (
                xy, done, speed, tilt, pose_error, contact_match, slip, stance,
                tangential_slip, support, lateral, heading, gyro, _, _,
            ) = (trace[case] for trace in traces)
            failures = np.flatnonzero(done)
            length = int(failures[0] + 1) if failures.size else total_steps
            valid = slice(length)
            rows.append(dict(
                lateral_kick_m_s=lateral_kick,
                yaw_kick_rad_s=yaw_kick,
                start_y_m=start_y,
                completed=not bool(failures.size),
                steps=length,
                dx_m=float(xy[length - 1, 0]),
                dy_m=float(xy[length - 1, 1]),
                centerline_mae_m=float(np.mean(np.abs(xy[valid, 1]))),
                max_cross_track_m=float(np.max(np.abs(xy[valid, 1]))),
                speed_mae_m_s=float(np.mean(np.abs(speed[valid] - TARGET_SPEED))),
                lateral_mae_m_s=float(np.mean(np.abs(lateral[valid]))),
                max_heading_error_deg=float(np.max(np.abs(heading[valid]))),
                gyro_z_mae_rad_s=float(np.mean(np.abs(gyro[valid]))),
                max_tilt_deg=float(np.max(tilt[valid])),
                pose_rmse_rad=float(np.sqrt(np.mean(pose_error[valid]))),
                contact_match=float(np.mean(contact_match[valid])),
                stance_slip_m_s=float(slip[valid].sum() / max(1, stance[valid].sum())),
                stance_contact_slip_m_s=float(tangential_slip[valid].sum() / max(1, stance[valid].sum())),
                min_support_after_0_5s=int(np.min(support[min(round(0.5 / env.dt), length - 1):length])),
            ))
        if return_replay:
            return rows, {"qpos": traces[-2][0], "time": traces[-1][0], "heading": traces[-4][0]}
        return rows

    return summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Orbax pretraining directory")
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--weight", type=float, default=0.10)
    args = parser.parse_args()
    if not args.checkpoint.is_dir() or args.steps < 1 or not 0 < args.weight <= 1:
        parser.error("Need an existing pretraining directory, positive steps and a weight in (0, 1]")

    seed, num_envs = 0, 8
    baseline = SpiderMJXEnv()
    qpos, contact = load_reference(baseline)
    guided = SpiderMJXEnv(qpos, args.weight)
    home = baseline.reset(jax.random.PRNGKey(seed)).data
    addresses = baseline._qpos_addresses
    matched = home.replace(qpos=home.qpos.at[addresses].set(qpos[0]))
    offset = matched.replace(qpos=matched.qpos.at[addresses].add(0.30))
    np.testing.assert_allclose(guided._reward(matched, TARGET_SPEED, 0, True), 1 + args.weight)
    assert float(guided._reward(offset, TARGET_SPEED, 0, True)) < 1 + args.weight
    np.testing.assert_allclose(guided._reward(matched, TARGET_SPEED, 1, True), 0)
    np.testing.assert_allclose(baseline._reward(matched, TARGET_SPEED, 0, True), 1)

    settings = training_settings(baseline, num_envs, CYCLES * baseline.cycle_steps, seed)
    settings["restore_checkpoint_path"] = str(args.checkpoint.resolve())
    infer, initial, _ = train.train(num_timesteps=0, **settings)
    _, guided_initial, _ = train.train(num_timesteps=0, **{**settings, "environment": guided})
    for left, right in zip(jax.tree_util.tree_leaves(initial), jax.tree_util.tree_leaves(guided_initial)):
        np.testing.assert_array_equal(left, right)
    if actor_parameter_change(initial[1], train.train(num_timesteps=0, **{
        **settings, "restore_checkpoint_path": None,
    })[1][1]) == 0:
        raise RuntimeError("Training did not restore the pretrained Actor")

    print(f"device={jax.devices()[0]}  envs={num_envs}  requested_steps_per_arm={args.steps}", flush=True)
    evaluate = make_evaluator(baseline, infer, qpos, contact, seed)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix="prm-", dir=OUTPUT))
    results = {}
    for name, env in (("speed_only", baseline), ("speed_plus_prm", guided)):
        def report_progress(steps, _metrics):
            print(f"{name}: collected_steps={steps}", flush=True)

        print(f"training {name}", flush=True)
        _, params, _ = train.train(
            num_timesteps=args.steps,
            progress_fn=report_progress,
            **{**settings, "environment": env},
        )
        change = actor_parameter_change(initial[1], params[1])
        if not np.isfinite(change) or change <= 0:
            raise RuntimeError(f"Actor update failed for {name}")
        brax_model.save_params(str(run / f"{name}.params"), params)
        results[name] = {"actor_max_change": change, "evaluation": evaluate(params)}
        print(f"{name}: {json.dumps(results[name])}", flush=True)

    results["warm_start"] = {"evaluation": evaluate(initial)}
    report = {
        "seed": seed,
        "num_envs": num_envs,
        "requested_steps_per_arm": args.steps,
        "reference_weight": args.weight,
        "checkpoint": str(args.checkpoint.resolve()),
        "results": results,
    }
    path = run / "comparison.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"report={path}")
    print("prm comparison: PASS")


if __name__ == "__main__":
    main()
