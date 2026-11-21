"""Pay for actual backward progress up to the requested speed."""
import jax.numpy as jp

from playground.open_duck_mini_v2 import locomotion, locomotion_reverse_tracking


def default_config():
    config = locomotion_reverse_tracking.default_config()
    config.reverse_progress_scale = .55
    return config


def reverse_progress_reward(command, filtered_velocity, scale):
    requested_speed = jp.maximum(-command[0], .01)
    progress = jp.clip(-filtered_velocity / requested_speed, -1., 1.)
    return jp.where(command[0] < -.01, scale * progress, 0.)


class ReverseProgress(locomotion_reverse_tracking.ReverseTracking):
    def __init__(self, task='flat_terrain', config=None, config_overrides=None):
        super().__init__(task=task, config=default_config() if config is None else config,
                         config_overrides=config_overrides)

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        terms = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
        progress = reverse_progress_reward(info['command'], info['filtered_twist'][0], self._config.reverse_progress_scale)
        terms['tracking_lin_vel'] += jp.where(info['invalid_transition'], 0., progress)
        info['signed_reward'] = locomotion.signed_reward(terms, self._config.reward_config.scales, self.dt)
        return terms
