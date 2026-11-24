"""从静止的膝关节读出位置伺服力矩，再推进一个物理步。"""

from pathlib import Path

import mujoco


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "spider_single_leg.xml"


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    knee = data.joint("knee")
    servo = data.actuator("knee_position")

    knee.qpos[0] = 0.3
    servo.ctrl[0] = 0.4

    mujoco.mj_forward(model, data)
    print(f"mj_forward  t={data.time:.3f} s  q={knee.qpos[0]:.6f} rad  "
          f"torque={servo.force[0]:+.3f} N*m")

    mujoco.mj_step(model, data)
    print(f"mj_step     t={data.time:.3f} s  q={knee.qpos[0]:.6f} rad  "
          f"qvel={knee.qvel[0]:+.6f} rad/s")


if __name__ == "__main__":
    main()
