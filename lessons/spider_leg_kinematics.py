"""小蜘蛛单腿实验：固定髋角，测量脚尖在世界坐标系中的位置。"""

from pathlib import Path

import mujoco


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "spider_single_leg.xml"
HIP_ANGLE = 0.0
KNEE_ANGLES = (-0.2, 0.0, 0.2)


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    hip = data.joint("hip")
    knee = data.joint("knee")
    foot = data.site("foot_tip")

    hip.qpos[0] = HIP_ANGLE
    print("knee(rad)        x(m)        y(m)        z(m)")
    for angle in KNEE_ANGLES:
        knee.qpos[0] = angle
        mujoco.mj_forward(model, data)
        x, y, z = foot.xpos
        print(f"{angle:+9.3f}  {x:+10.6f}  {y:+10.6f}  {z:+10.6f}")


if __name__ == "__main__":
    main()
