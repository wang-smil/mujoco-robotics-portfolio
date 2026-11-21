"""Use a blend of ordinary and reference gait resets for reverse training."""
import jax
import jax.numpy as jp
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding

from playground.open_duck_mini_v2 import locomotion_gait, locomotion_reverse_tracking


def default_config():
    config = locomotion_reverse_tracking.default_config()
    config.reverse_reference_reset_probability = .45
    config.reverse_reference_reset_blend = .35
    return config


class ReverseReferenceReset(locomotion_reverse_tracking.ReverseTracking):
    def __init__(self, task='flat_terrain', config=None, config_overrides=None):
        super().__init__(task=task, config=default_config() if config is None else config,
                         config_overrides=config_overrides)

    def reset(self, rng):
        state = super().reset(rng)
        choice, phase_key = jax.random.split(jax.random.fold_in(rng, 817))
        phase = jax.random.randint(phase_key, (), 0, self.PRM.nb_steps_in_period)
        reverse = state.info['command'][0] < -.01
        aided = reverse & jax.random.bernoulli(choice, self._config.reverse_reference_reset_probability)
        return jax.lax.cond(aided, lambda value: self._reference_reset(value, phase), lambda value: value, state)

    def _reference_reset(self, state, phase):
        command = state.info['command']
        reference = self.PRM.get_reference_motion(*command[:3], phase)
        blend = self._config.reverse_reference_reset_blend
        leg_ids = jp.array([0, 1, 2, 3, 4, 9, 10, 11, 12, 13])
        source_leg_ids = jp.array([0, 1, 2, 3, 4, 11, 12, 13, 14, 15])
        positions = self.get_actuator_joints_qpos(state.data.qpos)
        velocities = self.get_actuator_joints_qvel(state.data.qvel)
        deltas = jp.clip(blend * (reference[source_leg_ids] - positions[leg_ids]), -.16, .16)
        positions = positions.at[leg_ids].add(deltas)
        velocities = velocities.at[leg_ids].add(jp.clip(blend * reference[16 + source_leg_ids], -1., 1.))
        qpos = self.set_actuator_joints_qpos(positions, state.data.qpos)
        qvel = self.set_actuator_joints_qvel(velocities, state.data.qvel)
        world_velocity = state.data.site_xmat[self._site_id] @ jp.array([command[0], 0., 0.])
        qvel = qvel.at[:2].set((1. - blend) * qvel[:2] + blend * world_velocity[:2])
        ctrl = state.data.ctrl.at[leg_ids].set(positions[leg_ids])
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)
        old_z = jp.min(state.data.site_xpos[self._feet_site_id, 2])
        new_z = jp.min(data.site_xpos[self._feet_site_id, 2])
        qpos = qpos.at[2].add(jp.clip(old_z - new_z, -.015, .015))
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)
        feet = jp.array([geoms_colliding(data, geom, self._floor_geom_id) for geom in self._feet_geom_id])
        info = {key: value for key, value in state.info.items() if key != '_episode_reset'}
        info.update(imitation_i=phase, current_reference_motion=reference, motor_targets=ctrl, last_contact=feet)
        initial_action = (ctrl - self._default_actuator) / self._config.action_scale
        delay = self._config.noise_config.action_max_delay
        info.update(last_act=initial_action, last_last_act=initial_action, last_last_last_act=initial_action,
                    action_history=jp.tile(initial_action, delay))
        obs = self._get_obs(data, info, feet)
        info['filtered_twist'] = jp.concatenate([self.get_local_linvel(data)[:2], self.get_gyro(data)[2:3]])
        info['_episode_reset'] = dict(info)
        assisted = state.replace(data=data, info=info,
            obs=locomotion_gait.patch_filtered_observation(obs, info['filtered_twist']))
        return jax.lax.cond(self._get_termination(data), lambda: state, lambda: assisted)
