"""小蜘蛛前后速度指令和机身前向速度的公共计算。"""

from spider_walk import TARGET_SPEED


COMMAND_SLEW_M_S2 = 0.08
TRACKING_WIDTH_M_S = 0.025


def slew_speed(previous, requested, dt, xp):
    limit = COMMAND_SLEW_M_S2 * dt
    return previous + xp.clip(requested - previous, -limit, limit)


def scale_action(action, hips, speed, xp):
    # 反向时保留抬腿相位，只反转前后摆腿；零速回到 home。
    ratio = speed / TARGET_SPEED
    scale = xp.where(hips, xp.minimum(xp.abs(ratio), 1.0), ratio)
    return xp.clip(action * scale, -1.0, 1.0)


def forward_speed(delta_xy, torso_forward_xy, dt, xp):
    unit = torso_forward_xy / xp.maximum(xp.linalg.norm(torso_forward_xy), 1e-6)
    return xp.dot(delta_xy, unit) / dt
