"""重放小蜘蛛检查点，定位步态退化并生成同步的 MuJoCo 对照录像。"""

import argparse
import json
from pathlib import Path
from tempfile import mkdtemp

from brax.io import model as brax_model
from brax.training.agents.ppo import train
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spider_mjx_env import SpiderMJXEnv
from spider_ppo import OUTPUT, training_settings
from spider_ppo_pretrain import DEMONSTRATION
from spider_walk import CYCLES, TARGET_SPEED


POLICY_FILE = "speed_plus_lateral_gyro_heading.params"


def record_traces(env, make_policy, policies, key):
    with np.load(DEMONSTRATION, allow_pickle=False) as demo:
        names = demo["obs_names"]
        velocity_columns = np.flatnonzero(np.char.startswith(names, "velocity_"))
        period_velocities = demo["obs"][:, velocity_columns].reshape(CYCLES, env.cycle_steps, env.action_size)
        velocities = jp.asarray(period_velocities.mean(axis=0))
        action_deltas = np.abs(np.diff(demo["action"], axis=0))
        rate_limit = jp.asarray(1.25 * action_deltas.max(axis=0) / env.action_scale)

    def rollout(params, intervention):
        actor = make_policy(params, deterministic=True)

        def advance(carry, tick):
            state, previous = carry
            reference_velocity = velocities[env._phase_index(state.data)]
            observed_velocity = jp.where(intervention == 2, reference_velocity, state.obs[velocity_columns])
            obs = state.obs.at[velocity_columns].set(observed_velocity)
            action, _ = actor(obs, key)
            limited = jp.clip(action, previous - rate_limit, previous + rate_limit)
            applied = jp.where((intervention == 1) & (tick > 0), limited, action)
            state = jax.lax.cond(state.done > 0, lambda: state, lambda: env.step(state, applied))
            data = state.data
            trace = {
                "time": data.time,
                "qpos": data.qpos,
                "qvel": data.qvel,
                "target": data.ctrl,
                "force": data.actuator_force,
                "feet_xy": data.geom_xpos[env._feet, :2],
                "contact": env._floor_contact(data, env._feet),
                "heading": jp.degrees(env.heading_error(data)),
                "tilt": state.metrics["torso_tilt_deg"],
                "reward": state.reward,
                "forward_speed": state.metrics["forward_speed"],
                "done": state.done,
            }
            return (state, applied), trace

        _, trace = jax.lax.scan(
            advance, (env.reset(key), jp.zeros(env.action_size)), jp.arange(CYCLES * env.cycle_steps),
        )
        return trace

    batch = jax.tree_util.tree_map(lambda *values: jp.stack(values), *policies)
    # 最后两条轨迹使用同一长跑策略，只在回放时改一个因素。
    modes = jp.array([0, 0, 0, 1, 2])
    return jax.tree_util.tree_map(np.asarray, jax.jit(jax.vmap(rollout))(batch, modes))


def summarize(trace, env):
    time = trace["time"]
    failed = np.flatnonzero(trace["done"])
    count = int(failed[0] + 1) if failed.size else len(time)
    steady = slice(min(env.cycle_steps, count - 1), count)
    target_change = np.diff(trace["target"], axis=0, prepend=trace["target"][:1])
    target_second_change = np.diff(target_change, axis=0, prepend=target_change[:1])
    contact = trace["contact"]
    stance = contact[1:count] & contact[:count - 1]
    foot_speed = np.linalg.norm(np.diff(trace["feet_xy"][:count], axis=0), axis=-1) / env.dt
    stable_stance = stance[env.cycle_steps - 1:]
    stable_slip = foot_speed[env.cycle_steps - 1:][stable_stance]
    return dict(
        completed=not bool(failed.size),
        forward_m=float(trace["qpos"][count - 1, 0]),
        drift_m=float(trace["qpos"][count - 1, 1]),
        max_heading_deg=float(np.max(np.abs(trace["heading"][:count]))),
        max_tilt_deg=float(np.max(trace["tilt"][:count])),
        forward_mae_m_s=float(np.mean(np.abs(trace["forward_speed"][:count] - TARGET_SPEED))),
        mean_reward=float(np.mean(trace["reward"][:count])),
        target_delta_rms_rad=float(np.sqrt(np.mean(target_change[steady] ** 2))),
        target_delta_p95_rad=float(np.quantile(np.abs(target_change[steady]), 0.95)),
        target_second_delta_rms_rad=float(np.sqrt(np.mean(target_second_change[steady] ** 2))),
        joint_speed_rms_rad_s=float(np.sqrt(np.mean(trace["qvel"][steady][:, env._qvel_addresses] ** 2))),
        actuator_force_rms=float(np.sqrt(np.mean(trace["force"][steady] ** 2))),
        stance_slip_m_s=float(stable_slip.mean()) if stable_slip.size else None,
    )


def render_comparison(model, traces, path, panels=("Warm start", "PPO 4,096", "PPO 32,768"), *, lookat=(.045, 0, .05)):
    width, height, header, footer = 360, 264, 38, 32
    for index in range(model.ngeom):
        geom = model.geom(index)
        body = model.body(int(model.geom_bodyid[index])).name
        if body.startswith("left_"):
            geom.rgba[:] = (.20, .64, .73, 1)
        elif body.startswith("right_"):
            geom.rgba[:] = (.80, .40, .20, 1)
    model.geom("torso_shell").rgba[:] = (.21, .24, .32, 1)
    model.geom("floor").rgba[:] = (.67, .71, .73, 1)
    model.vis.headlight.ambient[:] = (.55, .55, .55)
    model.vis.headlight.diffuse[:] = (.75, .75, .75)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance, camera.azimuth, camera.elevation = .94, 115, -58
    data = mujoco.MjData(model)

    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 16)
        stats_font = ImageFont.truetype("DejaVuSansMono.ttf", 12)
    except OSError:
        title_font = stats_font = ImageFont.load_default()
    frames = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for tick in range(1, traces["time"].shape[1], 2):
            canvas = Image.new("RGB", (len(panels) * width, header + height + footer), "#141b25")
            draw = ImageDraw.Draw(canvas)
            for index, label in enumerate(panels):
                data.qpos[:] = traces["qpos"][index, tick]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                canvas.paste(Image.fromarray(renderer.render()), (index * width, header))
                x, y = data.qpos[:2] * 100
                yaw = traces["heading"][index, tick]
                draw.text((index * width + 12, 9), label, fill="#f2f5fa", font=title_font)
                if "requested" in traces:
                    vx, vy, wz = traces["measured"][index, max(0, tick - 19):tick + 1].mean(axis=0)
                    _, _, target_wz = traces["requested"][index, tick]
                    status = f"vx {vx*100:+4.1f} vy {vy*100:+4.1f} cm/s  yaw {wz:+.2f}/{target_wz:+.2f}"
                elif "command" in traces:
                    request = traces["command"][index, tick] * 100
                    speed = traces["speed"][index, max(0, tick - 19):tick + 1].mean() * 100
                    status = f"Goal {request:+4.1f}   speed {speed:+4.1f} cm/s   X {x:+5.1f} cm"
                else:
                    status = f"X {x:+5.1f} cm   Y {y:+5.1f} cm   yaw {yaw:+4.1f} deg"
                draw.text((index * width + 12, header + height + 9), status, fill="#d2e0ee", font=stats_font)
                if index:
                    draw.line((index * width, 0, index * width, canvas.height), fill="#e0e3e8")
            draw.text((canvas.width - 64, 12), f"{traces['time'][0, tick]:.2f}s", fill="#e0e3e8", font=stats_font)
            frames.append(canvas.quantize(colors=96))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=40, loop=0, optimize=False)
    frames[-1].convert("RGB").save(path.with_name("last_frame.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("long_report", type=Path, help="Long training comparison.json")
    args = parser.parse_args()
    report = json.loads(args.long_report.read_text(encoding="utf-8"))
    short_report = Path(report["control_report"])
    env = SpiderMJXEnv(
        lateral_weight=report["lateral_weight"], yaw_rate_weight=report["gyro_weight"],
        heading_weight=report["heading_weight"],
    )
    settings = training_settings(env, report["num_envs"], CYCLES * env.cycle_steps, report["seed"])
    settings["restore_checkpoint_path"] = str(Path(report["checkpoint"]).resolve(strict=True))
    infer, warm, _ = train.train(num_timesteps=0, **settings)
    short = brax_model.load_params(str(short_report.parent / POLICY_FILE))
    long = brax_model.load_params(str(args.long_report.parent / POLICY_FILE))
    labels = ("warm_start", "ppo_4096", "ppo_32768", "long_with_rate_limit", "long_with_reference_joint_velocity")
    print(f"device={jax.devices()[0]}  replaying={len(labels)}", flush=True)
    traces = record_traces(env, infer, (warm, short, long, long, long), jax.random.PRNGKey(report["seed"]))
    results = {
        label: summarize({name: values[index] for name, values in traces.items()}, env)
        for index, label in enumerate(labels)
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix="diagnosis-", dir=OUTPUT))
    path = run / "diagnosis.json"
    path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    np.savez_compressed(run / "replay.npz", labels=labels, **traces)
    print(f"diagnosis={path}", flush=True)
    for label, metrics in results.items():
        print(f"{label}: {json.dumps(metrics, allow_nan=False)}", flush=True)
    video = run / "walking_comparison.gif"
    render_comparison(env.mj_model, traces, video)
    print(f"video={video}", flush=True)


if __name__ == "__main__":
    main()
