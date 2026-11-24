"""Command-conditioned reference gait for the two-joint spider legs."""

from dataclasses import dataclass

import mujoco
import numpy as np

from spider_walk import ACTION_LIMIT, GROUPS, LIFT_ANGLE, PERIOD, STANCE_FRACTION, TARGET_SPEED


@dataclass(frozen=True)
class CommandGait:
    sweep_matrix: np.ndarray
    actuator_phase: np.ndarray
    actuator_lift: np.ndarray
    foot_phase: np.ndarray

    def offsets(self, command, phase, xp):
        """Return target joint offsets in radians for body-frame vx, vy, and yaw rate."""
        command = xp.asarray(command)
        phase = xp.mod(phase + xp.asarray(self.actuator_phase), 1.0)
        stance = phase < STANCE_FRACTION
        progress = xp.where(stance, phase / STANCE_FRACTION, (phase - STANCE_FRACTION) / (1 - STANCE_FRACTION))
        sweep = xp.where(stance, 1, -1) * xp.cos(xp.pi * progress)
        lift_envelope = xp.where(stance, 0.0, xp.square(xp.sin(xp.pi * progress)))
        nominal_command = xp.asarray((TARGET_SPEED, 0.014, 0.22))
        intensity = xp.minimum(1.0, xp.linalg.norm(command / nominal_command))
        swing = xp.asarray(self.sweep_matrix) @ command
        return xp.clip(sweep * swing + xp.asarray(self.actuator_lift) * lift_envelope * intensity,
                       -ACTION_LIMIT, ACTION_LIMIT)

    def normalized_action(self, command, phase, xp):
        return self.offsets(command, phase, xp) / ACTION_LIMIT

    def stance_mask(self, phase, xp):
        """Reference stance flags in the same foot order as spider_mjx_env."""
        return xp.mod(phase + xp.asarray(self.foot_phase), 1.0) < STANCE_FRACTION


def build_command_gait(model):
    """Build the joint-space gait map with MuJoCo foot Jacobians at home."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    torso = data.body("torso")
    torso_from_world = torso.xmat.reshape(3, 3).T
    sweep = np.zeros((model.nu, 3), dtype=np.float32)
    actuator_phase = np.zeros(model.nu, dtype=np.float32)
    actuator_lift = np.zeros(model.nu, dtype=np.float32)
    foot_phase = []
    jacobian = np.zeros((3, model.nv))
    half_stance = 0.5 * PERIOD * STANCE_FRACTION

    for group_index, group in enumerate(GROUPS):
        for leg in group:
            hip = model.actuator(leg + "_hip_position").id
            knee = model.actuator(leg + "_knee_position").id
            actuators = [hip, knee]
            foot = model.geom(leg + "_foot").id
            dofs = model.jnt_dofadr[model.actuator_trnid[actuators, 0]]
            mujoco.mj_jacGeom(model, data, jacobian, None, foot)
            foot_x, foot_y = (torso_from_world @ (data.geom_xpos[foot] - torso.xpos))[:2]
            planar_twist = np.array(((1.0, 0.0, -foot_y), (0.0, 1.0, foot_x)))
            local_jacobian = (torso_from_world @ jacobian)[:2, dofs]
            joint_sweep = half_stance * np.linalg.solve(local_jacobian, planar_twist)

            # Lateral hip sweep changes foot height as well as reach. Native replay
            # favors a larger tangential knee sweep with hip used for swing clearance.
            joint_sweep[:, 0] *= 1.06
            joint_sweep[0, 1] = 0.0
            joint_sweep[1, 1] *= 10.5
            sweep[actuators] = joint_sweep
            actuator_phase[actuators] = 0.5 * group_index
            actuator_lift[hip] = LIFT_ANGLE
            foot_phase.append(0.5 * group_index)

    return CommandGait(sweep, actuator_phase, actuator_lift, np.asarray(foot_phase, dtype=np.float32))
