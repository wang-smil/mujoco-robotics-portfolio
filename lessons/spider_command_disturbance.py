"""传感器读数噪声和控制周期级传输延时，供原生 MuJoCo 和 MJX 共用。"""

from math import radians


MAX_DELAY_TICKS = 2
HISTORY_LENGTH = MAX_DELAY_TICKS + 1
NOISE_STD = {
    "attitude_roll_pitch_deg": .35,
    "heading_deg": .50,
    "linear_velocity_m_s": .002,
    "gyro_rad_s": .018,
    "joint_position_rad": .006,
    "joint_velocity_rad_s": .12,
}


def push_history(history, newest, delay, xp):
    updated = xp.concatenate((newest[None, ...], history[:-1]), axis=0)
    return updated, updated[delay]


def sensor_observation(current, delayed, standard_normal, xp):
    """只扰动传感器数据，保留最新步态相位和控制端生成的指令。"""
    quaternion_delta = xp.asarray((0., .5, .5, 0.)) * radians(NOISE_STD["attitude_roll_pitch_deg"])
    attitude = delayed[:4] + standard_normal[:4] * quaternion_delta
    attitude = attitude / xp.maximum(xp.linalg.norm(attitude), 1e-8)
    heading = xp.arctan2(delayed[59], delayed[60] + 1.)
    heading += radians(NOISE_STD["heading_deg"]) * standard_normal[59]
    return xp.concatenate((
        attitude,
        delayed[4:7] + NOISE_STD["linear_velocity_m_s"] * standard_normal[4:7],
        delayed[7:10] + NOISE_STD["gyro_rad_s"] * standard_normal[7:10],
        delayed[10:26] + NOISE_STD["joint_position_rad"] * standard_normal[10:26],
        delayed[26:42] + NOISE_STD["joint_velocity_rad_s"] * standard_normal[26:42],
        delayed[42:50], current[50:59], xp.stack((xp.sin(heading), xp.cos(heading) - 1.0)),
    ))
