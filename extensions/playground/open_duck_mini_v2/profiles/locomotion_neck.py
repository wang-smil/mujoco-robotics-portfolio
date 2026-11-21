"""Minimal head/neck target slew limit; measured velocity is not hard-clamped."""
import copy
import jax.numpy as jp
import numpy as np
from playground.open_duck_mini_v2 import locomotion_gait, joystick


def default_config():
    cfg=locomotion_gait.default_config()
    cfg.head_max_motor_velocity=.5
    return cfg


def motor_velocity_limits(base_limits,head_limit):
    values=np.asarray(base_limits,dtype=float)
    if values.ndim==0:values=np.full(14,float(values))
    if values.shape!=(14,) or not np.isfinite(values).all() or np.any(values<=0):
        raise ValueError('Motor speed limits must be a positive finite scalar or14-element vector')
    if not np.isfinite(head_limit) or head_limit<=0:
        raise ValueError('Head target speed limit must be positive and finite')
    return jp.asarray(values).at[5:9].set(float(head_limit))


def configured_limits(config=None,config_overrides=None):
        cfg=copy.deepcopy(default_config() if config is None else config)
        with cfg.ignore_type():cfg.update_from_flattened_dict(config_overrides or {})
        limits=motor_velocity_limits(cfg.max_motor_velocity,cfg.head_max_motor_velocity)
        with cfg.ignore_type():cfg.max_motor_velocity=limits
        return cfg


class NeckLimitedGait(locomotion_gait.GaitLocomotion):
    def __init__(self,task='flat_terrain',config=None,config_overrides=None):
        if not joystick.USE_MOTOR_SPEED_LIMITS:
            raise RuntimeError('Neck profile requires the inherited target slew limiter')
        super().__init__(task=task,config=configured_limits(config,config_overrides),config_overrides=None)
