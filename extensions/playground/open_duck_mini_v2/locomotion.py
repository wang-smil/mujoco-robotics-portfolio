"""Locomotion reward v1, alongside (without editing) the user's Joystick task.

All command and tracking velocities use the body frame. A signed failure cost
survives the legacy step's nonnegative clipping. Weights are starting values,
not evidence of learned locomotion; validate physical metrics after training.
"""
import jax
import jax.numpy as jp
from ml_collections import config_dict

from playground.open_duck_mini_v2 import joystick
from playground.open_duck_mini_v2.custom_rewards import reward_imitation
from playground.common.rewards import cost_action_rate, cost_torques

REWARD_VERSION = 'locomotion-v1'


def default_config(stage='forward'):
    if stage not in ('forward', 'joystick'):
        raise ValueError(f'Unknown locomotion stage: {stage}')
    cfg = joystick.default_config()
    cfg.reward_config = config_dict.create(
        scales=config_dict.create(tracking_lin_vel=8., tracking_ang_vel=2.,
            alive=.5, orientation=-4., lin_vel_z=-.5, ang_vel_xy=-.1,
            torques=-.001, action_rate=-.05, stand_pose=-2., stand_motion=-2.,
            feet_slip=-.5, imitation=1., termination=-10.),
        linear_sigma=.05, angular_sigma=.3,
    )
    cfg.head_range_factor = 0.
    if stage == 'forward':
        cfg.lin_vel_x = [.05, .15]
        cfg.lin_vel_y = [0., 0.]
        cfg.ang_vel_yaw = [0., 0.]
        cfg.noise_config.level = .3
        cfg.push_config.enable = False
    else:
        cfg.lin_vel_x = [-.15, .15]
        cfg.lin_vel_y = [-.12, .12]
        cfg.ang_vel_yaw = [-.6, .6]
    return cfg


def tracking_rewards(command, velocity, gyro, linear_sigma=.05, angular_sigma=.3):
    linear_error = jp.sum(jp.square(command[:2] - velocity[:2]))
    angular_error = jp.square(command[2] - gyro[2])
    return jp.exp(-linear_error / linear_sigma**2), jp.exp(-angular_error / angular_sigma**2)


def stability_costs(command, velocity, gyro, up, pose_error, foot_velocity, contact):
    stand = jp.linalg.norm(command[:3]) < .01
    return dict(
        orientation=jp.square(1. - up[2]),
        ang_vel_xy=jp.sum(jp.square(gyro[:2])),
        stand_pose=jp.mean(jp.square(pose_error)) * stand,
        stand_motion=(jp.sum(jp.square(velocity[:2])) + jp.square(gyro[2])) * stand,
        feet_slip=jp.sum(jp.sum(jp.square(foot_velocity[:, :2]), axis=1) * contact),
    )


def signed_reward(terms, scales, dt):
    return sum(terms[key] * scale for key, scale in scales.items()) * dt


def is_failure(up_z, height, qpos, qvel):
    return (up_z < .5) | (height < .10) | ~jp.isfinite(qpos).all() | ~jp.isfinite(qvel).all()


def patch_command_observation(obs, command, reference):
    """Fixed101/212 schema: patch targets without drawing noise/history again."""
    actor = obs['state'].at[6:13].set(command)
    critic = obs['privileged_state'].at[:101].set(actor)
    critic = critic.at[-43:-3].set(reference)
    return dict(state=actor, privileged_state=critic)


def sanitize_terms(terms, invalid):
    invalid = invalid | ~jp.stack([jp.isfinite(value) for value in terms.values()]).all()
    terms = {key: jp.where(invalid, 1. if key == 'termination' else 0.,
                          jp.nan_to_num(value, nan=0., posinf=0., neginf=0.))
             for key, value in terms.items()}
    return terms, invalid


class Locomotion(joystick.Joystick):
    def __init__(self, task='flat_terrain', config=None, config_overrides=None):
        super().__init__(task=task, config=default_config() if config is None else config,
                         config_overrides=config_overrides)

    def reset(self, rng):
        state = super().reset(rng)
        state.info['signed_reward'] = jp.zeros(())
        state.info['invalid_transition'] = jp.array(False)
        return state

    def step(self, state, action):
        previous = state
        # The inherited step mutates info/metrics dictionaries. Keep an intact
        # previous transition for finite terminal fallback and batched evaluation.
        state = state.replace(info=dict(state.info), metrics=dict(state.metrics))
        state = super().step(state, action)
        invalid = state.info['invalid_transition']
        def finite_fallback(current, prior):
            return jp.where(invalid, prior, current)
        data = jax.tree_util.tree_map(finite_fallback, state.data, previous.data)
        obs = jax.tree_util.tree_map(finite_fallback, state.obs, previous.obs)
        info = jax.tree_util.tree_map(finite_fallback, state.info, previous.info)
        info['signed_reward'] = state.info['signed_reward']
        info['invalid_transition'] = invalid
        # Parent resamples the NEXT command after building obs/reward for the
        # completed transition. Align the next observation/reference only now.
        info['current_reference_motion'] = self.PRM.get_reference_motion(
            info['command'][0], info['command'][1], info['command'][2], info['imitation_i'])
        obs = patch_command_observation(obs, info['command'], info['current_reference_motion'])
        # Parent computes all weighted metrics, but clips their sum at zero.
        # _get_reward stores the same sum before clipping, with the same command.
        return state.replace(data=data, obs=obs, info=info,
            reward=info['signed_reward'], done=jp.where(invalid, 1., state.done),
            metrics={key: jp.nan_to_num(value, nan=0., posinf=0., neginf=0.)
                     for key, value in state.metrics.items()})

    def _invalid_data(self, data):
        return ~jp.stack([jp.isfinite(value).all() for value in
            (data.qpos, data.qvel, data.ctrl, data.sensordata, data.actuator_force)]).all()

    def _get_termination(self, data):
        return is_failure(self.get_gravity(data)[2],
            self.get_floating_base_qpos(data.qpos)[2], data.qpos, data.qvel) | self._invalid_data(data)

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        velocity = self.get_local_linvel(data)
        gyro = self.get_gyro(data)
        joints = self.get_actuator_joints_qpos(data.qpos)
        leg_error = jp.concatenate([(joints-self._default_actuator)[:5],
                                    (joints-self._default_actuator)[9:]])
        linear, angular = tracking_rewards(info['command'], velocity, gyro,
            self._config.reward_config.linear_sigma, self._config.reward_config.angular_sigma)
        terms = stability_costs(info['command'], velocity, gyro, self.get_gravity(data),
            leg_error, data.sensordata[self._foot_linvel_sensor_adr], contact)
        # Reference velocities are commanded in body axes. Free-joint qvel[:3]
        # is world translation and would disagree after random initial yaw.
        body_qvel = jp.concatenate([velocity, gyro])
        terms.update(tracking_lin_vel=linear, tracking_ang_vel=angular, alive=jp.ones(()),
            lin_vel_z=jp.square(self.get_global_linvel(data)[2]),
            torques=cost_torques(data.actuator_force),
            action_rate=cost_action_rate(action, info['last_act']),
            imitation=reward_imitation(self.get_floating_base_qpos(data.qpos), body_qvel,
                joints, self.get_actuator_joints_qvel(data.qvel), contact,
                info['current_reference_motion'], info['command'], joystick.USE_IMITATION_REWARD),
            termination=done.astype(jp.float32))
        terms, invalid = sanitize_terms(terms, self._invalid_data(data))
        info['invalid_transition'] = invalid
        info['signed_reward'] = signed_reward(terms, self._config.reward_config.scales, self.dt)
        return terms
