"""保持七条腿的 home 目标，抬起并放回左前腿。"""

from math import radians

import mujoco
import numpy as np

from spider_standing import build_model, floor_contact_geoms, torso_tilt_degrees


LIFT_ANGLE = radians(6)
START_LIFT, END_LIFT = 0.6, 1.0
START_LOWER, END_LOWER = 1.6, 2.0
DURATION = 2.6


def main():
    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    hip = data.actuator("left_1_hip_position")
    home_target = float(hip.ctrl[0])
    torso = data.body("torso")
    floor = model.geom("floor").id
    shell = model.geom("torso_shell").id
    foot = model.geom("left_1_foot")
    feet = {index for index in range(model.ngeom) if model.geom(index).name.endswith("_foot")}
    other_feet = feet - {foot.id}

    min_clearance, max_tilt = float("inf"), 0.0
    min_supports = len(other_feet)
    initial_contact = False
    swing_contact = False
    for _ in range(round(DURATION / model.opt.timestep)):
        # 逐渐加大髋角，保持一段，再逐渐回到 home。
        blend = np.interp(data.time, [START_LIFT, END_LIFT, START_LOWER, END_LOWER], [0, 1, 1, 0])
        hip.ctrl[0] = home_target + LIFT_ANGLE * blend
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        contacts = floor_contact_geoms(data, floor)
        if shell in contacts:
            raise RuntimeError("Torso touched the floor")
        max_tilt = max(max_tilt, torso_tilt_degrees(torso))
        if 0.5 <= data.time <= START_LIFT:
            initial_contact |= feet <= contacts
        if END_LIFT + 0.1 <= data.time <= START_LOWER:
            clearance = data.geom(foot.id).xpos[2] - foot.size[0]
            min_clearance = min(min_clearance, clearance)
            min_supports = min(min_supports, len(contacts & other_feet))
            swing_contact |= foot.id in contacts

    touchdown = feet <= contacts
    print(f"min_hold_clearance={min_clearance:.4f} m  other_supports={min_supports}")
    print(f"max_tilt={max_tilt:.2f} deg  touchdown={'yes' if touchdown else 'no'}")
    if (
        not initial_contact
        or swing_contact
        or not touchdown
        or not np.isfinite(min_clearance)
        or min_clearance < 0.01
        or min_supports != len(other_feet)
        or max_tilt > 5
        or not np.isclose(data.time, DURATION)
        or any(warning.number for warning in data.warning)
    ):
        raise RuntimeError("Single-leg lift check failed")
    print("lift: PASS")


if __name__ == "__main__":
    main()
