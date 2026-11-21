"""Train the combined policy on both step and gamepad-slewed commands."""
import jax
import jax.numpy as jp

from playground.open_duck_mini_v2 import locomotion_combined
from playground.open_duck_mini_v2.locomotion_gait import coherent_task_info


def default_config():
    config = locomotion_combined.default_config()
    config.command_slew = [.3, .3, 1.]
    config.command_ramp_probability = .65
    return config


def applied_command(previous, target, limits, ramp):
    twist = previous[:3] + jp.clip(target[:3] - previous[:3], -limits, limits)
    return target.at[:3].set(jp.where(ramp, twist, target[:3]))


def observation_for_command(obs, command):
    return {name: value.at[6:13].set(command) for name, value in obs.items()}


class SlewedLocomotion(locomotion_combined.CombinedLocomotion):
    def __init__(self, task='flat_terrain', config=None, config_overrides=None):
        super().__init__(task=task, config=default_config() if config is None else config,
                         config_overrides=config_overrides)

    def reset(self, rng):
        environment_key, ramp_key = jax.random.split(rng)
        state = super().reset(environment_key)
        target = state.info['command']
        ramp = jax.random.bernoulli(ramp_key, self._config.command_ramp_probability)
        zero_twist = target.at[:3].set(0.)
        command = jp.where(ramp, zero_twist, target)
        context = dict(command=command, command_target=target, command_ramp_active=ramp)
        snapshot = {**state.info['_episode_reset'], **context}
        return state.replace(info={**state.info, **context, '_episode_reset': snapshot},
                             obs=observation_for_command(state.obs, command))

    def step(self, state, action):
        info = coherent_task_info(state.info, state.obs['state'][-2:])
        due = info['step'] >= self._config.turn_command_hold_steps - 1
        last_command = info['command']
        last_target = info['command_target']
        last_ramp = info['command_ramp_active']
        state = super().step(state.replace(info=info), action)
        next_key, ramp_key = jax.random.split(state.info['rng'])
        new_ramp = jax.random.bernoulli(ramp_key, self._config.command_ramp_probability)
        resampled = due & (state.done == 0.)
        target = jp.where(resampled, state.info['command'], last_target)
        ramp = jp.where(resampled, new_ramp, last_ramp)
        limits = self.dt * jp.asarray(self._config.command_slew)
        command = applied_command(last_command, target, limits, ramp)
        info = {**state.info, 'rng': next_key, 'command_target': target, 'command_ramp_active': ramp, 'command': command}
        return state.replace(info=info, obs=observation_for_command(state.obs, command))
