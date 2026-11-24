"""平地八足三轴指令环境，支持基础奖励和参考步态奖励。"""

import jax
import jax.numpy as jp
from mujoco import mjx
import numpy as np

from spider_command_disturbance import HISTORY_LENGTH, push_history, sensor_observation
from spider_command_task import (
    LINEAR_TRACK_WIDTH, YAW_TRACK_WIDTH, advance_command, body_velocity, command_authority, heading_error, observation,
)
from spider_mjx_env import SpiderMJXEnv


RICH_WEIGHTS = dict(reference_pose=.30, reference_contact=.05, slip=.12, heading=.20, tilt=.08,
                    action_rate=.025, support=.10, alive=.04, fall=1.0)


class SpiderCommandEnv(SpiderMJXEnv):
    def __init__(self, profiles, episode_steps, line_search_iterations, reference=None, *, disturbances=False, model=None):
        super().__init__(model=model)
        profiles = np.asarray(profiles, np.float32)
        if (profiles.ndim != 3 or profiles.shape[-1] != 3 or not profiles.size or not np.isfinite(profiles).all()
                or episode_steps < profiles.shape[1] or episode_steps % profiles.shape[1]):
            raise ValueError("Each profile must provide xyz commands for equal segments of the episode")
        self.profiles = jp.asarray(profiles)
        self.segment_steps = episode_steps // profiles.shape[1]
        self.reference = reference
        self.disturbances = disturbances
        self.reward_mode = "rich" if reference is not None else "base"
        self.home_joints = self._home_data.qpos[self._qpos_addresses]
        if reference is not None:
            self._slip_weight = RICH_WEIGHTS["slip"]
        if self.mj_model.opt.ls_iterations != line_search_iterations:
            self.mj_model.opt.ls_iterations = line_search_iterations
            self._mjx_model = mjx.put_model(self.mj_model)

    @property
    def observation_size(self):
        return 61

    def _command(self, profile, time):
        tick = jp.rint(time / self.dt).astype(jp.int32)
        segment = jp.minimum(tick // self.segment_steps, self.profiles.shape[1] - 1)
        return self.profiles[profile, segment]

    def reset(self, rng):
        state = super().reset(rng)
        profile = jax.random.randint(rng, (), 0, self.profiles.shape[0])
        applied = jp.zeros(3, dtype=state.obs.dtype)
        yaw = self._yaw(state.data)
        obs = observation(state.obs, yaw, self.profiles[profile, 0], applied, yaw, jp)
        info = {**state.info, "profile": profile, "applied_command": applied, "command_yaw": yaw}
        if self.disturbances:
            info.update(
                noise_key=jax.random.fold_in(rng, 9989), delivery_delays=jp.zeros(2, dtype=jp.int32),
                initial_sensor=obs, sensor_history=jp.broadcast_to(obs, (HISTORY_LENGTH, self.observation_size)),
                motor_history=jp.zeros((HISTORY_LENGTH, self.action_size), dtype=state.obs.dtype),
            )
        metrics = {**state.metrics, "tracking_vx_abs": state.reward, "tracking_vy_abs": state.reward,
                   "tracking_wz_abs": state.reward}
        return state.replace(obs=obs, info=info, metrics=metrics)

    def step(self, state, action):
        yaw_before = self._yaw(state.data)
        fresh = state.data.time < self.dt / 2
        requested = self._command(state.info["profile"], state.data.time)
        previous = jp.where(fresh, 0, state.info["applied_command"])
        applied = advance_command(previous, requested, self.dt, jp)
        target_yaw = jp.where(fresh, yaw_before, state.info["command_yaw"]) + self.dt * applied[2]
        valid_action = jp.all(jp.isfinite(action))
        motor_action = jp.where(valid_action, action * command_authority(applied, jp), jp.nan)
        if self.disturbances:
            noise_key, delay_rng, obs_rng = jax.random.split(state.info["noise_key"], 3)
            sampled_delays = jax.random.randint(delay_rng, (2,), 0, HISTORY_LENGTH)
            delays = jp.where(fresh, sampled_delays, state.info["delivery_delays"])
            motor_history, motor_action = push_history(
                jp.where(fresh, 0.0, state.info["motor_history"]), motor_action, delays[0], jp,
            )
        updated = super().step(state, motor_action)
        yaw = self._yaw(updated.data)
        yaw_delta = heading_error(yaw, yaw_before, jp)
        xy_delta = updated.data.qpos[:2] - state.data.qpos[:2]
        measured = jp.concatenate((body_velocity(xy_delta, yaw_before + yaw_delta / 2, self.dt, jp),
                                   jp.atleast_1d(yaw_delta / self.dt)))
        error = applied - measured
        valid = valid_action & jp.all(jp.isfinite(error)) & jp.all(jp.isfinite(updated.data.qpos))
        done = jp.maximum(updated.done, (~valid).astype(updated.done.dtype))
        reward = (1.2 * jp.exp(-jp.sum(jp.square(error[:2] / jp.asarray(LINEAR_TRACK_WIDTH))))
                  + .7 * jp.exp(-jp.square(error[2] / YAW_TRACK_WIDTH)) - done)
        if self.reference is not None:
            reward += self._rich_reward(state, updated, applied, target_yaw, done)
        next_request = self._command(state.info["profile"], updated.data.time)
        obs = observation(updated.obs, jp.where(valid, yaw, 0.0), next_request, applied, target_yaw, jp)
        metrics = {
            **updated.metrics, "forward_speed": jp.where(valid, measured[0], 0.0),
            "tracking_vx_abs": jp.where(valid, jp.abs(error[0]), 0.0),
            "tracking_vy_abs": jp.where(valid, jp.abs(error[1]), 0.0),
            "tracking_wz_abs": jp.where(valid, jp.abs(error[2]), 0.0),
        }
        info = {**updated.info, "applied_command": applied, "command_yaw": target_yaw}
        if self.disturbances:
            history = jp.where(fresh, state.info["initial_sensor"], state.info["sensor_history"])
            sensor_history, delayed = push_history(history, obs, delays[1], jp)
            obs = sensor_observation(obs, delayed, jax.random.normal(obs_rng, obs.shape), jp)
            info.update(noise_key=noise_key, delivery_delays=delays, motor_history=motor_history,
                        sensor_history=sensor_history)
        return updated.replace(obs=obs, reward=jp.where(valid & jp.isfinite(reward), reward, -2.),
                               done=done, metrics=metrics, info=info)

    def _rich_reward(self, previous, current, command, yaw_goal, done):
        weights = RICH_WEIGHTS
        phase = ((current.data.time / self.dt - 2) / self.cycle_steps) % 1.0
        pose = self.home_joints + self.reference.offsets(command, phase, jp)
        pose_error = jp.mean(jp.square((current.data.qpos[self._qpos_addresses] - pose) / .14))
        contacts = self._floor_contact(current.data, self._feet)
        reference_contact = self.reference.stance_mask(phase, jp)
        moving = command_authority(command, jp)
        imitation = moving * (weights["reference_pose"] * jp.exp(-pose_error)
                              + weights["reference_contact"] * jp.mean(contacts == reference_contact))
        tracking_heading = heading_error(self._yaw(current.data), yaw_goal, jp)
        tilt = current.metrics["torso_tilt_deg"]
        action_rate = current.data.ctrl - previous.data.ctrl
        return ((1 - done) * (imitation + weights["alive"])
                + weights["heading"] * jp.expm1(-jp.square(tracking_heading / .25))
                + weights["tilt"] * jp.expm1(-jp.square(tilt / 8.0))
                + weights["action_rate"] * jp.expm1(-jp.mean(jp.square(action_rate / .045)))
                - weights["support"] * jp.maximum(0., 3. - jp.sum(contacts)) / 3.0
                - self._stance_slip_penalty(previous.data, current.data) - weights["fall"] * done)
