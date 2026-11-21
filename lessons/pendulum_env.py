"""第 4 课：MuJoCo 单摆的 Gymnasium 环境；动作直接控制电机力矩。"""

from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np


MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts/lesson01/pendulum_motor.xml"


class PendulumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None, max_episode_steps=400,
                 model_name="Lesson 04 - Reinforcement learning environment"):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or human")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        xml = ET.parse(MODEL_PATH).getroot()
        xml.set("model", model_name)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        self.model.opt.gravity[:] = [0, 0, -9.81]
        self.data = mujoco.MjData(self.model)
        self.target = 0.5
        self.frame_skip = 10
        self.control_dt = self.frame_skip * float(self.model.opt.timestep)  # 0.02 秒
        self.max_episode_steps = max_episode_steps
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        self.render_mode = render_mode
        self.viewer = None
        self.key_callback = None  # 仅供演示程序接收查看器按键。
        self.viewer_label = "Pendulum environment"
        self.viewer_controls = "1: random  2: PD+gravity  R: restart  Space: pause"
        self.elapsed_steps = 0
        self.last_action, self.last_torque, self.last_reward = 0.0, 0.0, 0.0
        self._needs_reset = True

    def _observation(self):
        # 目标固定为 0.5，故 error 也能唯一确定 q；没有向策略提供 bias。
        return np.array([self.target - self.data.qpos[0], self.data.qvel[0]], dtype=np.float32)

    def _info(self):
        return {"q_rad": float(self.data.qpos[0]), "dq_rad_s": float(self.data.qvel[0]),
                "error_rad": self.target - float(self.data.qpos[0]),
                "torque_Nm": self.last_torque, "episode_step": self.elapsed_steps,
                "simulated_seconds": float(self.data.time)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # Gymnasium 提供可复现的随机数发生器。
        options = {} if options is None else options
        mujoco.mj_resetData(self.model, self.data)
        # 训练时轻微随机初态；演示/边界检查可通过 options 指定。
        q = float(options.get("q", self.np_random.uniform(-0.05, 0.05)))
        dq = float(options.get("dq", self.np_random.uniform(-0.05, 0.05)))
        if not np.isfinite([q, dq]).all():
            raise ValueError("Initial q and dq must be finite")
        self.data.qpos[0], self.data.qvel[0] = q, dq
        mujoco.mj_forward(self.model, self.data)
        self.elapsed_steps = 0
        self.last_action, self.last_torque, self.last_reward = 0.0, 0.0, 0.0
        self._needs_reset = False
        return self._observation(), self._info()

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Call reset() before step(), including after an episode ends")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError("Action must be one finite number in an array of shape (1,)")
        self.last_action = float(np.clip(action[0], -1.0, 1.0))
        self.last_torque = 2.0 * self.last_action
        # 环境不替策略做 PD 或重力补偿；同一动作保持 10 个物理步。
        for _ in range(self.frame_skip):
            self.data.ctrl[0] = self.last_torque
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.elapsed_steps += 1
        obs = self._observation()
        if not np.isfinite(obs).all():
            raise RuntimeError("Non-finite simulation state; inspect the model and action")
        error, dq = float(obs[0]), float(obs[1])
        # 这是本任务的失败边界；到达目标后仍需继续保持，不立即结束。
        terminated = bool(abs(error) > 1.2 or abs(dq) > 8.0)
        truncated = bool(self.elapsed_steps >= self.max_episode_steps and not terminated)
        error_cost, speed_cost = 4.0 * error**2, 0.05 * dq**2
        torque_cost = 0.001 * self.last_torque**2
        failure_cost = 10.0 if terminated else 0.0
        self.last_reward = float(1.0 - error_cost - speed_cost - torque_cost - failure_cost)
        info = self._info()
        info.update(error_cost=error_cost, speed_cost=speed_cost,
                    torque_cost=torque_cost, failure_cost=failure_cost)
        self._needs_reset = terminated or truncated
        return obs, self.last_reward, terminated, truncated, info

    def render(self):
        """显式调用 render() 显示；训练时 render_mode=None，无需画图。"""
        if self.render_mode != "human":
            return None
        if self.viewer is None:
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(
                self.model, self.data, key_callback=self.key_callback,
                show_left_ui=False, show_right_ui=False)
            target_data = mujoco.MjData(self.model)
            target_data.qpos[0] = self.target
            mujoco.mj_forward(self.model, target_data)
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = [0.25, 0, 1.15]
                self.viewer.cam.distance, self.viewer.cam.azimuth, self.viewer.cam.elevation = 2.7, 90, -10
                sphere = self.viewer.user_scn.geoms[0]
                mujoco.mjv_initGeom(
                    sphere, mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.15, 0, 0]), target_data.geom("bob").xpos,
                    np.eye(3).ravel(), np.array([0.2, 0.9, 0.35, 0.25], dtype=np.float32))
                self.viewer.user_scn.ngeom = 1
        info = self._info()
        text = (f"{self.viewer_label}\n"
                f"Environment step={self.elapsed_steps}/{self.max_episode_steps}  t={self.data.time:.2f}s\n"
                "Physics dt=0.002s  action dt=0.020s\n"
                f"obs=[error {info['error_rad']:+.3f}, speed {info['dq_rad_s']:+.3f}]\n"
                f"action={self.last_action:+.3f}  torque={self.last_torque:+.3f} Nm\n"
                f"reward={self.last_reward:+.4f} (computed after action)\n"
                "Green=target; yellow=actual bob\n"
                f"{self.viewer_controls}")
        self.viewer.set_texts((mujoco.mjtFont.mjFONT_BIG,
                               mujoco.mjtGridPos.mjGRID_TOPLEFT, text, ""))
        self.viewer.sync()
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
