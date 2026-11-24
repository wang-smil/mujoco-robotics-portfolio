"""用原生 MuJoCo 验收三轴速度指令，并比较训练检查点。"""

from math import atan2, cos, degrees, pi, sin
from zlib import crc32

import jax
import mujoco
import numpy as np

from spider_standing import build_model, floor_contact_geoms, torso_tilt_degrees
from spider_command_disturbance import HISTORY_LENGTH, MAX_DELAY_TICKS, push_history, sensor_observation
from spider_walk import CONTROL_SUBSTEPS, FALL_TILT_DEGREES, GROUPS, PERIOD


# x/y 单位为 m/s，yaw 单位为 rad/s。非零指令还要求同向且达到一半幅值。
MOVING_TOLERANCE = np.array([0.010, 0.006, 0.120])
IDLE_TOLERANCE = np.array([0.006, 0.006, 0.050])
MAX_HEADING_ERROR_DEG = 20.0
MIN_SUPPORT = 2
MAX_UNSUPPORTED_SECONDS = 0.06


def _rotation(yaw):
    return np.array(((cos(yaw), -sin(yaw)), (sin(yaw), cos(yaw))))


def _angle_delta(angle, previous):
    return (angle - previous + pi) % (2 * pi) - pi


def _stance_slip(model, data, floor, stance):
    """按接触点而非足底球心计算与地面平行的滑速。"""
    speeds = []
    velocity = np.empty(6)
    for contact in data.contact:
        if contact.geom1 == floor and contact.geom2 in stance:
            geom = contact.geom2
        elif contact.geom2 == floor and contact.geom1 in stance:
            geom = contact.geom1
        else:
            continue
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, geom, velocity, 0)
        offset = contact.pos - data.geom_xpos[geom]
        contact_velocity = velocity[3:] + np.cross(velocity[:3], offset)
        speeds.append(float(np.linalg.norm(contact_velocity[:2])))
    return speeds


def _expand_route(route, control_dt):
    segments = []
    for seconds, command in route:
        seconds = float(seconds)
        command = np.asarray(command, dtype=np.float64)
        if not np.isfinite(seconds) or seconds <= 0 or command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("Each segment needs a positive duration and finite [vx, vy, yaw_rate]")
        steps = round(seconds / control_dt)
        if steps < 1:
            raise ValueError("A segment cannot be shorter than one control step")
        segments.append((steps, command))
    if not segments:
        raise ValueError("Evaluation requires at least one command segment")
    return segments


def _summarize(trace, segments, slip_sums, slip_counts, reason, dt):
    result_segments = []
    end = 0
    for count, requested in segments:
        end += count
        # 用每段的最后一个完整步态周期评估稳态，切换瞬态保留在逐步误差里。
        start = end - min(round(PERIOD / dt), count)
        available = len(trace["time"]) >= end
        target_moving = np.abs(requested) > 1.0e-8
        tolerance = np.where(target_moving, MOVING_TOLERANCE, IDLE_TOLERANCE)
        if available:
            settled = trace["measured"][start:end]
            mean = settled.mean(axis=0)
            error = np.abs(mean - requested)
            direction_ok = ~target_moving | ((mean * np.sign(requested)) >= 0.5 * np.abs(requested))
            passed = bool(np.all(direction_ok & (error <= tolerance)))
            actual = mean.tolist()
            abs_error = error.tolist()
        else:
            actual = abs_error = None
            passed = False
        result_segments.append({
            "target": requested.tolist(),
            "mean_velocity": actual,
            "abs_error": abs_error,
            "passed": passed,
        })

    valid_steps = len(trace["time"])
    completed = reason is None and valid_steps == end
    if valid_steps:
        settled_from = min(round(0.5 / dt), valid_steps - 1)
        support = trace["support"][settled_from:]
        longest_air = int(max(np.diff(np.flatnonzero(np.r_[True, support > 0, True]))) - 1) * dt
        valid_slip_count = sum(slip_counts[settled_from:])
        slip = sum(slip_sums[settled_from:]) / valid_slip_count if valid_slip_count else None
        step_abs_error = np.abs(trace["applied"] - trace["measured"]).mean(axis=0).tolist()
        heading_error = float(np.abs(trace["heading_error_deg"]).max())
        min_support = int(support.min())
        low_support_fraction = float(np.mean(support < MIN_SUPPORT))
        tilt = float(trace["tilt_deg"].max())
        endpoint = trace["qpos"][-1, :2].tolist()
        goal_endpoint = trace["target_xy"][-1].tolist()
        path_error = np.linalg.norm(trace["qpos"][:, :2] - trace["target_xy"], axis=1)
        path_rmse = float(np.sqrt(np.mean(path_error ** 2)))
    else:
        longest_air = slip = step_abs_error = heading_error = min_support = low_support_fraction = None
        tilt = endpoint = goal_endpoint = path_rmse = None

    tracking_passed = completed and all(section["passed"] for section in result_segments)
    stability_passed = (completed and heading_error <= MAX_HEADING_ERROR_DEG
                        and longest_air <= MAX_UNSUPPORTED_SECONDS and low_support_fraction <= 0.05)
    return dict(
        completed=completed, failure=reason, passed=tracking_passed and stability_passed,
        tracking_passed=tracking_passed, stability_passed=stability_passed,
        simulated_seconds=valid_steps * dt, expected_seconds=end * dt, segments=result_segments,
        step_tracking_mae=step_abs_error, max_heading_error_degrees=heading_error,
        min_support_after_settle=min_support, longest_unsupported_seconds=longest_air,
        low_support_fraction=low_support_fraction, stance_contact_slip_m_s=slip, max_tilt_degrees=tilt,
        endpoint_xy_m=endpoint, reference_endpoint_xy_m=goal_endpoint, planned_path_rmse_m=path_rmse,
    )


def make_native_command_evaluator(make_policy, *, observe, advance, seed=0, model=None):
    """返回 (model, evaluate)，对各检查点重放同一组指令。

    observe(data, requested, applied, heading_reference) 产生 Actor 观测。
    advance(data, action, requested, applied) 返回本步关节目标和实际执行指令。
    evaluate(params, routes) 返回按路线名索引的统计和原始轨迹。
    可选延时以控制步为单位。噪声种子按路线派生，A/B 的同路线使用同一采样流。
    扰动只作用于 Actor 观测和关节目标，评测指标仍从原生物理真值计算。
    """
    model = build_model() if model is None else model
    dt = float(model.opt.timestep) * CONTROL_SUBSTEPS
    floor = model.geom("floor").id
    shell = model.geom("torso_shell").id
    feet = {model.geom(name + "_foot").id for group in GROUPS for name in group}
    key = jax.random.PRNGKey(seed)

    @jax.jit
    def act(params, observation):
        return make_policy(params, deterministic=True)(observation, key)[0]

    def evaluate(params, routes, *, initial_yaw=0.0, action_delay=0, sensor_delay=0, noise_seed=None):
        if not np.isfinite(initial_yaw):
            raise ValueError("Initial yaw must be finite")
        if any(not isinstance(delay, (int, np.integer)) or not 0 <= delay <= MAX_DELAY_TICKS
               for delay in (action_delay, sensor_delay)):
            raise ValueError("Action and sensor delay must be integers from zero to two control ticks")
        if noise_seed is not None and (not isinstance(noise_seed, (int, np.integer)) or noise_seed < 0):
            raise ValueError("Noise seed must be a nonnegative integer")
        summaries = {}
        replays = {}
        for name, route in routes.items():
            noise_rng = None
            if noise_seed is not None:
                route_seed = crc32(str(name).encode("utf-8"))
                noise_rng = np.random.default_rng(np.random.SeedSequence([noise_seed, route_seed]))
            summaries[name], replays[name] = _replay(
                model, dt, floor, shell, feet, observe, advance, act, params, route, initial_yaw,
                action_delay=action_delay, sensor_delay=sensor_delay, noise_rng=noise_rng,
            )
            if action_delay or sensor_delay or noise_seed is not None:
                summaries[name]["perturbation"] = dict(
                    action_delay_steps=int(action_delay), sensor_delay_steps=int(sensor_delay),
                    action_delay_seconds=action_delay * dt, sensor_delay_seconds=sensor_delay * dt,
                    noise_seed=None if noise_seed is None else int(noise_seed),
                )
        return summaries, replays

    return model, evaluate


def _replay(model, dt, floor, shell, feet, observe, advance, act, params, route, initial_yaw,
            *, action_delay=0, sensor_delay=0, noise_rng=None):
    segments = _expand_route(route, dt)
    fields = (
        "time", "qpos", "heading", "requested", "applied", "measured",
        "heading_error_deg", "target_xy", "support", "tilt_deg", "stance_slip_m_s",
    )
    trace = {field: [] for field in fields}
    slip_sums, slip_counts = [], []
    data = mujoco.MjData(model)
    torso = data.body("torso")
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    if initial_yaw:
        orientation = np.empty(4)
        yaw_rotation = np.array([cos(initial_yaw / 2), 0.0, 0.0, sin(initial_yaw / 2)])
        mujoco.mju_mulQuat(orientation, yaw_rotation, data.qpos[3:7])
        data.qpos[3:7] = orientation
    mujoco.mj_forward(model, data)
    contacts = floor_contact_geoms(data, floor)
    applied = np.zeros(3)
    heading_ref = atan2(torso.xmat[3], torso.xmat[0])
    heading = heading_ref
    target_xy = data.qpos[:2].copy()
    reason = None
    ctrl_history = np.tile(data.ctrl, (HISTORY_LENGTH, 1)) if action_delay else None
    sensor_history = None

    for steps, requested in segments:
        for _ in range(steps):
            obs = np.asarray(observe(data, requested, applied, heading_ref), np.float32)
            if not np.isfinite(obs).all():
                reason = "nonfinite_observation"
                break
            if sensor_delay or noise_rng is not None:
                if sensor_history is None:
                    sensor_history = np.tile(obs, (HISTORY_LENGTH, 1))
                sensor_history, delayed = push_history(sensor_history, obs, sensor_delay, np)
                noise = (np.zeros_like(obs) if noise_rng is None
                         else noise_rng.standard_normal(obs.shape, dtype=obs.dtype))
                obs = sensor_observation(obs, delayed, noise, np)
            action = np.asarray(act(params, obs))
            if not np.isfinite(action).all():
                reason = "nonfinite_action"
                break
            targets, applied = advance(data, action, requested, applied)
            applied = np.asarray(applied, dtype=np.float64)
            if not np.isfinite(targets).all() or not np.isfinite(applied).all():
                reason = "nonfinite_control"
                break
            if ctrl_history is not None:
                ctrl_history, targets = push_history(ctrl_history, targets, action_delay, np)
            data.ctrl[:] = targets
            before = data.qpos[:2].copy()
            shell_hit = False
            for _ in range(CONTROL_SUBSTEPS):
                mujoco.mj_step(model, data)
                shell_hit |= shell in floor_contact_geoms(data, floor)
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                reason = "nonfinite_state"
                break
            rotation = torso.xmat.reshape(3, 3)
            yaw = atan2(rotation[1, 0], rotation[0, 0])
            dyaw = _angle_delta(yaw, heading)
            velocity_xy = _rotation(heading + dyaw / 2).T @ (data.qpos[:2] - before) / dt
            measured = np.r_[velocity_xy, dyaw / dt]
            heading += dyaw
            ref_dyaw = float(applied[2]) * dt
            target_xy += _rotation(heading_ref + ref_dyaw / 2) @ applied[:2] * dt
            heading_ref += ref_dyaw
            height_tilt = torso_tilt_degrees(torso)
            next_contacts = floor_contact_geoms(data, floor)
            speeds = _stance_slip(model, data, floor, feet & contacts & next_contacts)
            slip_sums.append(sum(speeds))
            slip_counts.append(len(speeds))
            heading_error_deg = degrees(heading - heading_ref)
            values = (data.time, data.qpos.copy(), degrees(heading), requested, applied.copy(), measured,
                      heading_error_deg, target_xy.copy(), len(feet & next_contacts), height_tilt,
                      float(np.mean(speeds)) if speeds else 0.0)
            for field, value in zip(fields, values):
                trace[field].append(value)
            contacts = next_contacts
            if shell_hit or shell in contacts:
                reason = "torso_contact"
                break
            if height_tilt > FALL_TILT_DEGREES:
                reason = "excessive_tilt"
                break
        if reason:
            break

    trace = {field: np.asarray(values) for field, values in trace.items()}
    return _summarize(trace, segments, slip_sums, slip_counts, reason, dt), trace


def aggregate_command_routes(summaries):
    """给检查点曲线提供相同的路线权重和分段验收指标。"""
    if not summaries:
        raise ValueError("Cannot aggregate an empty set of routes")
    routes = list(summaries.values())
    segment_errors = [segment["abs_error"] for route in routes for segment in route["segments"]
                      if segment["abs_error"] is not None]
    average = np.asarray(segment_errors).mean(axis=0).tolist() if segment_errors else [None] * 3
    slip = [route["stance_contact_slip_m_s"] for route in routes if route["stance_contact_slip_m_s"] is not None]
    heading = [route["max_heading_error_degrees"] for route in routes if route["max_heading_error_degrees"] is not None]
    return dict(
        routes=len(routes), complete_cases=sum(route["completed"] for route in routes),
        passed_cases=sum(route["passed"] for route in routes),
        failed_cases=sum(not route["completed"] for route in routes),
        scored_segments=len(segment_errors),
        total_segments=sum(len(route["segments"]) for route in routes),
        vx_mae_m_s=average[0], vy_mae_m_s=average[1], yaw_mae_rad_s=average[2],
        mean_stance_slip_m_s=float(np.mean(slip)) if slip else None,
        max_heading_error_degrees=max(heading, default=None),
    )


def plot_command_progress(series, output):
    """绘制各组检查点的路线通过数、分段误差、打滑、航向和跌倒。"""
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    if not series or not all(series.values()):
        raise ValueError("Each policy needs at least one evaluation checkpoint")
    figure, panels = plt.subplots(2, 3, figsize=(12, 6.3), layout="constrained")
    metrics = (
        ("passed_cases", "Routes passed", "routes", 1),
        ("vx_mae_m_s", "Forward tracking error", "cm/s", 100),
        ("vy_mae_m_s", "Lateral tracking error", "cm/s", 100),
        ("yaw_mae_rad_s", "Turn-rate tracking error", "rad/s", 1),
        ("mean_stance_slip_m_s", "Stance contact slip", "cm/s", 100),
        ("max_heading_error_degrees", "Worst heading error", "degrees", 1),
    )
    for axis, (metric, title, unit, scale) in zip(panels.flat, metrics):
        for name, rows in series.items():
            steps = np.array([row["step"] for row in rows])
            values = np.array([row.get(metric, np.nan) for row in rows], dtype=np.float64)
            axis.plot(steps, values * scale, marker="o", linewidth=1.6, markersize=4, label=name)
        axis.set(title=title, xlabel="Environment steps", ylabel=unit, ylim=(0, None))
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.xaxis.set_major_formatter(FuncFormatter(
            lambda steps, _: f"{steps / 1e6:g}M" if steps >= 1e6 else f"{steps / 1e3:g}k" if steps else "0",
        ))
        axis.grid(alpha=0.2)
        if metric == "passed_cases":
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            axis.set_ylim(0, max(row[metric] for rows in series.values() for row in rows) + 1)
    handles, labels = panels[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", ncols=len(series), frameon=False)
    figure.savefig(output, dpi=140)
    plt.close(figure)
