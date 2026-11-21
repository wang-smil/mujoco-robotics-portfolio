"""Align the Critic's reference with the applied command and penalize reverse overspeed."""
import jax.numpy as jp

from playground.open_duck_mini_v2 import locomotion, locomotion_combined_slew


def default_config():
    config = locomotion_combined_slew.default_config()
    config.reverse_imitation_weight = .25
    config.reverse_speed_tolerance = .035
    config.reverse_overspeed_weight = 2.
    return config


def reverse_terms(command_x, measured_x, imitation, imitation_weight, tolerance):
    reverse_intensity = jp.clip((-command_x - .01) / .035, 0., 1.)
    imitation_scale = 1. + reverse_intensity * (imitation_weight - 1.)
    overspeed = jp.clip((command_x - measured_x - tolerance) / .1, 0., 3.)
    return imitation * imitation_scale, reverse_intensity * jp.square(overspeed)


class StableReverseLocomotion(locomotion_combined_slew.SlewedLocomotion):
    def __init__(self, task='flat_terrain', config=None, config_overrides=None):
        super().__init__(task=task, config=default_config() if config is None else config,
                         config_overrides=config_overrides)

    def align_reference(self, state):
        observation = state.obs
        command = state.info['command']
        reference = self.PRM.get_reference_motion(*command[:3], state.info['imitation_i'])
        info = {**state.info, 'current_reference_motion': reference}
        return state.replace(info=info, obs=locomotion.patch_command_observation(observation, command, reference))

    def reset(self, rng):
        state = self.align_reference(super().reset(rng))
        snapshot = {**state.info['_episode_reset'], 'current_reference_motion': state.info['current_reference_motion']}
        metrics = {**state.metrics, 'cost/reverse_overspeed': jp.asarray(0.)}
        return state.replace(info={**state.info, '_episode_reset': snapshot}, metrics=metrics)

    def step(self, state, action):
        return self.align_reference(super().step(state, action))

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        terms = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
        terms['imitation'], overspeed = reverse_terms(
            info['command'][0], info['filtered_twist'][0], terms['imitation'],
            self._config.reverse_imitation_weight, self._config.reverse_speed_tolerance)
        terms, invalid = locomotion.sanitize_terms(terms, info['invalid_transition'])
        info['invalid_transition'] = invalid
        cost = jp.where(invalid, 0., self._config.reverse_overspeed_weight * overspeed)
        metrics['cost/reverse_overspeed'] = cost
        info['signed_reward'] = locomotion.signed_reward(terms, self._config.reward_config.scales, self.dt) - self.dt * cost
        return terms
