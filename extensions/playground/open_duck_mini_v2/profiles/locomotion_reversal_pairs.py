"""Practice the undersampled transitions between forward turns and reverse."""
import jax
import jax.numpy as jp

from playground.open_duck_mini_v2 import locomotion_combined_balance
from playground.open_duck_mini_v2.locomotion_combined_slew import applied_command
from playground.open_duck_mini_v2.locomotion_gait import coherent_task_info


def default_config():
    config = locomotion_combined_balance.default_config()
    config.reverse_pair_probability = .75
    config.stand_reverse_probability = .45
    return config


def choose_successor(previous, sampled, backward, arc, draw, pair_probability, stand_probability):
    vx, vy, yaw = previous[:3]
    stationary = (jp.abs(vx) < .01) & (jp.abs(vy) < .01) & (jp.abs(yaw) < .01)
    forward_arc = (vx > .03) & (jp.abs(yaw) > .05)
    reverse = vx < -.01
    into_reverse = (forward_arc & (draw < pair_probability)) | (stationary & (draw < stand_probability))
    out_of_reverse = reverse & (draw < pair_probability)
    return jp.where(into_reverse, backward, jp.where(out_of_reverse, arc, sampled))


class ReversalPairs(locomotion_combined_balance.BalanceLocomotion):
    def __init__(self, task='flat_terrain', config=None, config_overrides=None):
        super().__init__(task=task, config=default_config() if config is None else config,
                         config_overrides=config_overrides)

    def step(self, state, action):
        info = coherent_task_info(state.info, state.obs['state'][-2:])
        due = info['step'] >= self._config.turn_command_hold_steps - 1
        prior_target = info['command_target']
        prior_applied = info['command']
        state = super().step(state.replace(info=info), action)
        next_key, selector, reverse_key, vx_key, yaw_key, sign_key = jax.random.split(state.info['rng'], 6)
        reverse_speed = jax.random.uniform(reverse_key, minval=self._config.reverse_speed[0],
                                          maxval=self._config.reverse_speed[1])
        arc_x = jax.random.uniform(vx_key, minval=self._config.arc_forward_speed[0],
                                   maxval=self._config.arc_forward_speed[1])
        arc_yaw = jax.random.uniform(yaw_key, minval=self._config.turn_yaw_speed[0],
                                     maxval=self._config.turn_yaw_speed[1])
        arc_yaw *= jp.where(jax.random.bernoulli(sign_key), 1., -1.)
        empty = jp.zeros_like(prior_target)
        alternative = choose_successor(prior_target, state.info['command_target'], empty.at[0].set(-reverse_speed),
            empty.at[0].set(arc_x).at[2].set(arc_yaw), jax.random.uniform(selector),
            self._config.reverse_pair_probability, self._config.stand_reverse_probability)
        retarget = due & (state.done == 0.)
        target = jp.where(retarget, alternative, state.info['command_target'])
        limited = applied_command(prior_applied, target, self.dt * jp.asarray(self._config.command_slew),
                                  state.info['command_ramp_active'])
        command = jp.where(retarget, limited, state.info['command'])
        info = {**state.info, 'rng': next_key, 'command_target': target, 'command': command}
        return self._observe_balance(self.align_reference(state.replace(info=info)))
