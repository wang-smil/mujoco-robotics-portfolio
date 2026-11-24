"""三维机身速度指令以及 A/B 共用的指令和动作接口。"""


COMMAND_SCALE = (0.034, 0.012, 0.25)
COMMAND_SLEW = (0.08, 0.06, 1.2)
LINEAR_TRACK_WIDTH = (0.025, 0.012)
YAW_TRACK_WIDTH = 0.22


def advance_command(applied, requested, dt, xp):
    limit = xp.asarray(COMMAND_SLEW) * dt
    return applied + xp.clip(requested - applied, -limit, limit)


def body_velocity(delta_xy, heading, dt, xp):
    x, y = delta_xy / dt
    c, s = xp.cos(heading), xp.sin(heading)
    return xp.stack((c * x + s * y, -s * x + c * y))


def heading_error(yaw, reference, xp):
    error = yaw - reference
    return xp.arctan2(xp.sin(error), xp.cos(error))


def command_authority(applied, xp):
    magnitude = xp.max(xp.abs(applied) / xp.asarray(COMMAND_SCALE))
    return xp.minimum(1.0, 2.0 * magnitude)


def observation(base, yaw, requested, applied, reference_yaw, xp):
    w, x, y, z = base[:4]
    c, s = xp.cos(yaw / 2), xp.sin(yaw / 2)
    level_quat = xp.stack((c * w + s * z, c * x + s * y, c * y - s * x, c * z - s * w))
    level_quat = xp.where(level_quat[0] < 0, -level_quat, level_quat)
    local_xy = body_velocity(base[4:6], yaw, 1.0, xp)
    requested = requested / xp.asarray(COMMAND_SCALE)
    applied = applied / xp.asarray(COMMAND_SCALE)
    error = heading_error(yaw, reference_yaw, xp)
    # 第 53 个特征曾是世界系 y 偏移；侧移任务保留其输入位置并置零。
    commands = xp.stack((xp.zeros_like(error), requested[0], applied[0], requested[1], requested[2],
                         applied[1], applied[2], xp.sin(error), xp.cos(error) - 1))
    return xp.concatenate((level_quat, local_xy, base[6:], commands))
