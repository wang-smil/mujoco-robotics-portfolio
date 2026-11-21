"""Increase the cost of yaw tracking error during pure turns."""
import jax.numpy as jp

from playground.open_duck_mini_v2 import locomotion, locomotion_turn


def default_config():
    config = locomotion_turn.default_config()
    config.turn_yaw_error_multiplier = 2.0
    return config


def turn_tracking_reward(command, reward, multiplier):
    pure_turn = (jp.linalg.norm(command[:2]) < .01) & (jp.abs(command[2]) >= .01)
    return jp.where(pure_turn, 1. - multiplier * (1. - reward), reward)


class TurnTracking(locomotion_turn.ConservativeTurn):
    def __init__(self, task='flat_terrain', config=None, config_overrides=None):
        super().__init__(task=task, config=default_config() if config is None else config,
                         config_overrides=config_overrides)

    def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
        terms = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
        tracked = turn_tracking_reward(info['command'], terms['tracking_ang_vel'],
                                       self._config.turn_yaw_error_multiplier)
        terms['tracking_ang_vel'] = jp.where(info['invalid_transition'], 0., tracked)
        info['signed_reward'] = locomotion.signed_reward(terms, self._config.reward_config.scales, self.dt)
        return terms
