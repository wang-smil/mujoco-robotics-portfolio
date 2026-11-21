"""Penalize excess backward speed while retaining the walking curriculum."""
import jax.numpy as jp

from playground.open_duck_mini_v2 import locomotion, locomotion_reverse


def default_config():
    config = locomotion_reverse.default_config()
    config.reverse_overspeed_scale = .5
    config.reverse_overspeed_sigma = .045
    return config


def reverse_tracking_reward(command, filtered_velocity, reward, scale, sigma):
    is_reverse = command[0] < -.01
    overspeed = jp.maximum(command[0] - filtered_velocity, 0.)
    penalty = scale * (1. - jp.exp(-jp.square(overspeed / sigma)))
    return jp.where(is_reverse, reward - penalty, reward)


class ReverseTracking(locomotion_reverse.ReverseCurriculum):
    def __init__(self, task='flat_terrain', config=None, config_overrides=None):
        super().__init__(task=task, config=default_config() if config is None else config,
                         config_overrides=config_overrides)

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        terms = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
        tracked = reverse_tracking_reward(info['command'], info['filtered_twist'][0], terms['tracking_lin_vel'],
            self._config.reverse_overspeed_scale, self._config.reverse_overspeed_sigma)
        terms['tracking_lin_vel'] = jp.where(info['invalid_transition'], 0., tracked)
        info['signed_reward'] = locomotion.signed_reward(terms, self._config.reward_config.scales, self.dt)
        return terms
