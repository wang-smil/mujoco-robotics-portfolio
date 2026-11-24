"""从单腿 MJCF 生成八足模型，从 home 重置并验证平地站立。"""

from copy import deepcopy
from math import acos, degrees, radians, sin
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np


LEG_TEMPLATE = Path(__file__).resolve().parents[1] / "models" / "spider_single_leg.xml"
HOME_ANGLES = {"hip": radians(-30), "knee": 0.0}
FOOT_RADIUS = 0.012
STAND_SECONDS = 3.0
TORSO_GYRO = "torso_gyro"


def values(numbers):
    return " ".join(str(number) for number in numbers)


def build_model():
    root = ET.parse(LEG_TEMPLATE).getroot()
    root.set("model", "spider_standing")
    world = root.find("worldbody")
    torso = world.find("body")
    leg_template = torso.find("body")
    torso.remove(leg_template)
    actuators = root.find("actuator")
    actuator_templates = list(actuators)
    actuators.clear()

    # 两段连杆在模板的零位沿局部 +Y，站姿由负髋角放低足端。
    lower_leg = leg_template.find("body")
    toe_site = lower_leg.find("site")
    leg_length = float(lower_leg.attrib["pos"].split()[1]) + float(toe_site.attrib["pos"].split()[1])
    home_height = -leg_length * sin(HOME_ANGLES["hip"]) + FOOT_RADIUS + 0.001
    torso.set("pos", values((0, 0, home_height)))
    torso_geom = torso.find("geom")
    torso_geom.attrib.update(name="torso_shell", size="0.11 0.05 0.022", mass="0.35")
    torso.insert(0, ET.Element("freejoint", name="root"))
    ET.SubElement(world, "geom", name="floor", type="plane", size="1 1 0.1")
    ET.SubElement(torso, "site", name="torso_imu", size="0.002", rgba="0 0 0 0")
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "gyro", name=TORSO_GYRO, site="torso_imu")

    qpos = [0, 0, home_height, 1, 0, 0, 0]
    ctrl = []
    mounts = ((0.081, -30), (0.027, -10), (-0.027, 10), (-0.081, 30))
    for side, sign in (("left", 1), ("right", -1)):
        for index, (x, yaw) in enumerate(mounts, start=1):
            prefix = f"{side}_{index}_"
            leg = deepcopy(leg_template)
            leg.set("pos", values((x, sign * 0.05, 0)))
            leg.set("euler", values((0, 0, radians(yaw if sign == 1 else 180 - yaw))))
            qpos.extend(HOME_ANGLES[joint.attrib["name"]] for joint in leg.iter("joint"))
            for element in leg.iter():
                if "name" in element.attrib:
                    element.set("name", prefix + element.attrib["name"])
            # 连杆提供质量和惯量，球形足端负责地面接触。
            for geom in leg.iter("geom"):
                geom.attrib.update(contype="0", conaffinity="0")
            ET.SubElement(
                leg.find("body"),
                "geom",
                name=prefix + "foot",
                type="sphere",
                size=str(FOOT_RADIUS),
                pos=toe_site.attrib["pos"],
                mass="0.005",
            )
            torso.append(leg)
            for template in actuator_templates:
                actuator = deepcopy(template)
                joint = template.attrib["joint"]
                actuator.attrib.update(name=prefix + template.attrib["name"], joint=prefix + joint)
                actuators.append(actuator)
                ctrl.append(HOME_ANGLES[joint])

    keys = ET.SubElement(root, "keyframe")
    ET.SubElement(keys, "key", name="home", qpos=values(qpos), ctrl=values(ctrl))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def floor_contact_geoms(data, floor):
    contacts = set()
    for contact in data.contact:
        if contact.geom1 == floor:
            contacts.add(contact.geom2)
        if contact.geom2 == floor:
            contacts.add(contact.geom1)
    return contacts


def torso_tilt_degrees(torso):
    return degrees(acos(float(np.clip(torso.xmat[8], -1, 1))))


def main():
    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)

    torso = data.body("torso")
    floor = model.geom("floor").id
    shell = model.geom("torso_shell").id
    feet = {
        index for index in range(model.ngeom) if (model.geom(index).name or "").endswith("_foot")
    }
    min_height, max_tilt, min_support = float(torso.xpos[2]), 0.0, len(feet)
    for _ in range(round(STAND_SECONDS / model.opt.timestep)):
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        floor_contacts = floor_contact_geoms(data, floor)
        if shell in floor_contacts:
            raise RuntimeError("Torso touched the floor")
        min_height = min(min_height, torso.xpos[2])
        max_tilt = max(max_tilt, torso_tilt_degrees(torso))
        if data.time >= 0.5:
            min_support = min(min_support, len(floor_contacts & feet))

    drift = float(np.linalg.norm(torso.xpos[:2]))
    print(f"nq={model.nq} nv={model.nv} nu={model.nu}")
    print(f"t={data.time:.2f} s  min_height={min_height:.3f} m  max_tilt={max_tilt:.2f} deg")
    print(f"min_support_after_0.5s={min_support}  torso_drift={drift:.4f} m")
    if (
        (model.nq, model.nv, model.nu) != (23, 22, 16)
        or any(warning.number for warning in data.warning)
        or not np.isclose(data.time, STAND_SECONDS)
        or min_height < 0.07
        or max_tilt > 5
        or min_support < 4
        or drift > 0.02
    ):
        raise RuntimeError("Standing check failed")
    print("stand: PASS")


if __name__ == "__main__":
    main()
