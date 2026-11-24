"""用 Windows 手柄在原生 MuJoCo Viewer 驾驶已训练的小蜘蛛。"""

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import time

import mujoco
import numpy as np

from spider_command_curriculum import ARC_LEFT, ARC_RIGHT, BACKWARD, FORWARD, LEFT, RIGHT, TURN_LEFT, TURN_RIGHT
from spider_command_disturbance import HISTORY_LENGTH, push_history, sensor_observation
from spider_command_task import advance_command, body_velocity, command_authority, heading_error, observation
from spider_standing import build_model, floor_contact_geoms, torso_tilt_degrees
from spider_walk import ACTION_LIMIT, CONTROL_SUBSTEPS, FALL_TILT_DEGREES, GROUPS, PERIOD, read_observation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "spider-command-a-524288" / "checkpoint.npz"
DEADZONE = .18
GEARS = ("low", "medium", "high")
PAD_RESET, PAD_LOW, PAD_HIGH = 0x1000, 0x0100, 0x0200


def stick_command(forward, right, turn_right, gear):
    forward, right, turn_right = np.clip([forward, right, turn_right], -1., 1.)
    forward_cap = FORWARD[gear][0]
    turn_cap = TURN_LEFT[2]
    if forward and turn_right:
        forward_cap = min(forward_cap, ARC_LEFT[0])
        turn_cap = ARC_LEFT[2]
    return np.array([forward * forward_cap, -right * LEFT[gear][1], -turn_right * turn_cap])


def deadzone(value):
    value = np.clip(value, -1., 1.)
    return float(np.sign(value) * max(0., abs(value) - DEADZONE) / (1. - DEADZONE))


class XInputGamepad(ctypes.Structure):
    _fields_ = [("buttons", wintypes.WORD), ("left_trigger", ctypes.c_ubyte), ("right_trigger", ctypes.c_ubyte),
                ("left_x", ctypes.c_short), ("left_y", ctypes.c_short),
                ("right_x", ctypes.c_short), ("right_y", ctypes.c_short)]


class XInputState(ctypes.Structure):
    _fields_ = [("packet", wintypes.DWORD), ("gamepad", XInputGamepad)]


class JoyInfo(ctypes.Structure):
    _fields_ = [(name, wintypes.DWORD) for name in
                ("size", "flags", "x", "y", "z", "r", "u", "v", "buttons", "button", "pov", "reserved1", "reserved2")]


class JoyCaps(ctypes.Structure):
    _fields_ = (
        [("manufacturer", wintypes.WORD), ("product", wintypes.WORD), ("name", ctypes.c_wchar * 32)]
        + [(name, wintypes.UINT) for name in (
            "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "num_buttons", "period_min", "period_max",
            "r_min", "r_max", "u_min", "u_max", "v_min", "v_max", "caps", "max_axes", "num_axes", "max_buttons",
        )]
        + [("registry_key", ctypes.c_wchar * 32), ("oem_driver", ctypes.c_wchar * 260)]
    )


class WindowsControls:
    def __init__(self, generic_yaw_axis):
        self.generic_yaw_axis = generic_yaw_axis
        self._get_xinput = None
        self._winmm = self._user32 = self._ole32 = None
        self._com_initialized = False

    def __enter__(self):
        if os.name != "nt":
            return self
        self._ole32 = ctypes.WinDLL("ole32")
        initialize = self._ole32.CoInitializeEx
        initialize.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        initialize.restype = ctypes.c_long
        self._com_initialized = initialize(None, 0) in (0, 1)
        for dll in ("xinput1_4", "xinput9_1_0", "xinput1_3"):
            try:
                self._get_xinput = ctypes.WinDLL(dll).XInputGetState
            except (OSError, AttributeError):
                continue
            self._get_xinput.argtypes = [wintypes.DWORD, ctypes.POINTER(XInputState)]
            self._get_xinput.restype = wintypes.DWORD
            break
        self._winmm = ctypes.WinDLL("winmm")
        self._winmm.joyGetNumDevs.restype = wintypes.UINT
        self._winmm.joyGetPosEx.argtypes = [wintypes.UINT, ctypes.POINTER(JoyInfo)]
        self._winmm.joyGetPosEx.restype = wintypes.UINT
        self._winmm.joyGetDevCapsW.argtypes = [ctypes.c_size_t, ctypes.POINTER(JoyCaps), wintypes.UINT]
        self._winmm.joyGetDevCapsW.restype = wintypes.UINT
        self._user32 = ctypes.WinDLL("user32")
        self._user32.GetForegroundWindow.restype = wintypes.HWND
        self._user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self._user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self._user32.GetAsyncKeyState.restype = ctypes.c_short
        return self

    def __exit__(self, *exc):
        if self._com_initialized:
            self._ole32.CoUninitialize()

    def gamepad(self):
        if self._get_xinput is not None:
            state = XInputState()
            for index in range(4):
                if self._get_xinput(index, ctypes.byref(state)) == 0:
                    pad = state.gamepad
                    axes = tuple(map(deadzone, (pad.left_y / 32767., pad.left_x / 32767., pad.right_x / 32767.)))
                    return axes, pad.buttons, f"XInput controller {index + 1}"
        if self._winmm is not None:
            for index in range(self._winmm.joyGetNumDevs()):
                info = JoyInfo(size=ctypes.sizeof(JoyInfo), flags=0xff)
                caps = JoyCaps()
                if self._winmm.joyGetPosEx(index, ctypes.byref(info)) or self._winmm.joyGetDevCapsW(
                        index, ctypes.byref(caps), ctypes.sizeof(caps)):
                    continue
                def axis(name):
                    low, high = getattr(caps, f"{name}_min"), getattr(caps, f"{name}_max")
                    return deadzone(2. * (getattr(info, name) - low) / (high - low) - 1.) if high > low else 0.
                buttons = ((PAD_RESET if info.buttons & 1 else 0) | (PAD_LOW if info.buttons & 16 else 0)
                           | (PAD_HIGH if info.buttons & 32 else 0))
                return (-axis("y"), axis("x"), axis(self.generic_yaw_axis)), buttons, f"Windows joystick {index + 1}: {caps.name}"
        return (0., 0., 0.), 0, None

    def keyboard(self):
        if self._user32 is None:
            return (0., 0., 0.), 0
        foreground = self._user32.GetForegroundWindow()
        owner = wintypes.DWORD()
        self._user32.GetWindowThreadProcessId(foreground, ctypes.byref(owner))
        if owner.value != os.getpid():
            return (0., 0., 0.), 0

        def down(key):
            return int(bool(self._user32.GetAsyncKeyState(ord(key)) & 0x8000))

        axes = (down("W") - down("S"), down("D") - down("A"), down("E") - down("Q"))
        buttons = (PAD_RESET * down("R")) | (PAD_LOW * down("Z")) | (PAD_HIGH * down("X"))
        return axes, buttons


class Playback:
    def __init__(self, actor, *, action_delay=0, sensor_delay=0, noise_seed=None):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self.torso = self.data.body("torso")
        self.actor = actor
        self.dt = self.model.opt.timestep * CONTROL_SUBSTEPS
        self.home = self.model.key("home")
        self.floor = self.model.geom("floor").id
        self.shell = self.model.geom("torso_shell").id
        self.feet = [self.model.geom(name + "_foot").id for group in GROUPS for name in group]
        joints = self.model.actuator_trnid[:, 0]
        self.qpos_ids = self.model.jnt_qposadr[joints]
        self.qvel_ids = self.model.jnt_dofadr[joints]
        self.action_delay, self.sensor_delay = action_delay, sensor_delay
        self.noise_seed = noise_seed
        self.reset()

    @property
    def yaw(self):
        return np.arctan2(self.torso.xmat[3], self.torso.xmat[0])

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home.id)
        mujoco.mj_forward(self.model, self.data)
        self.applied = np.zeros(3)
        self.goal_yaw = self.yaw
        self.failure = None
        self.rng = np.random.default_rng(self.noise_seed) if self.noise_seed is not None else None
        self.target_history = np.repeat(self.data.ctrl[None, :], HISTORY_LENGTH, axis=0) if self.action_delay else None
        self.sensor_history = None
        if self.sensor_delay or self.rng is not None:
            initial = self._observation(self.applied)
            self.sensor_history = np.repeat(initial[None, :], HISTORY_LENGTH, axis=0)

    def _observation(self, command):
        phase = (round(self.data.time / self.dt) * self.dt / PERIOD) % 1.0
        contacts = floor_contact_geoms(self.data, self.floor)
        base = read_observation(self.data, self.qpos_ids, self.qvel_ids, self.feet, contacts, phase)
        return observation(base, self.yaw, command, self.applied, self.goal_yaw, np).astype(np.float32)

    def step(self, command):
        obs = self._observation(command)
        if self.sensor_history is not None:
            self.sensor_history, delayed = push_history(self.sensor_history, obs, self.sensor_delay, np)
            noise = self.rng.standard_normal(obs.shape, dtype=obs.dtype) if self.rng is not None else np.zeros_like(obs)
            obs = sensor_observation(obs, delayed, noise, np)
        action = self.actor(obs)
        if not np.isfinite(action).all():
            self.failure = "策略输出无效"
            return np.zeros(3)
        self.applied = advance_command(self.applied, command, self.dt, np)
        target = self.home.ctrl + ACTION_LIMIT * np.clip(action, -1., 1.) * command_authority(self.applied, np)
        if self.target_history is not None:
            self.target_history, target = push_history(self.target_history, target, self.action_delay, np)
        self.data.ctrl[:] = target
        previous_xy, previous_yaw = self.data.qpos[:2].copy(), self.yaw
        shell_contact = False
        for _ in range(CONTROL_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            shell_contact |= self.shell in floor_contact_geoms(self.data, self.floor)
        mujoco.mj_forward(self.model, self.data)
        self.goal_yaw += self.applied[2] * self.dt
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            self.failure = "仿真状态无效"
            return np.zeros(3)
        if (shell_contact or self.shell in floor_contact_geoms(self.data, self.floor)
                or torso_tilt_degrees(self.torso) > FALL_TILT_DEGREES):
            self.failure = "摔倒或机身触地"
        dyaw = heading_error(self.yaw, previous_yaw, np)
        xy = body_velocity(self.data.qpos[:2] - previous_xy, previous_yaw + dyaw / 2, self.dt, np)
        return np.append(xy, dyaw / self.dt)


def draw_grid(scene):
    if scene is None:
        return
    positions = np.arange(-1., 1.01, .1)
    for index, offset in enumerate(positions):
        for axis in range(2):
            pos = np.zeros(3)
            size = np.array([1., .0007, .0003]) if axis else np.array([.0007, 1., .0003])
            pos[axis] = offset
            geom = scene.geoms[scene.ngeom]
            color = (0.18, 0.44, 0.58, .75) if abs(offset) < 1e-6 else (.28, .33, .39, .45)
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX, size, pos, np.eye(3).ravel(), np.array(color, np.float32))
            scene.ngeom = 2 * index + axis + 1


def run_check(playback):
    commands = {"forward": FORWARD[1], "backward": BACKWARD[1], "left": LEFT[1], "right": RIGHT[1],
                "turn_left": TURN_LEFT, "turn_right": TURN_RIGHT, "arc_left": ARC_LEFT, "arc_right": ARC_RIGHT}
    failed = []
    for name, goal in commands.items():
        playback.reset()
        target = np.asarray(goal)
        measured = []
        for _ in range(round(3.6 / playback.dt)):
            measured.append(playback.step(target))
            if playback.failure:
                break
        velocity = np.mean(measured[-round(PERIOD / playback.dt):], axis=0)
        active = target != 0.
        if playback.failure or np.any(velocity[active] * np.sign(target[active]) < .5 * np.abs(target[active])):
            failed.append(f"{name}: {playback.failure or 'command tracking failed'}")
        print(f"{name:12s}  vx={velocity[0]:+.4f}  vy={velocity[1]:+.4f} m/s  yaw={velocity[2]:+.3f} rad/s")
    if failed:
        raise RuntimeError("Native command playback: " + ", ".join(failed))
    print("Native command playback: PASS", flush=True)


def run_viewer(playback, controls, status_file, seconds):
    import mujoco.viewer

    def publish(ready, device, gear, steps):
        if status_file is not None:
            status_file.parent.mkdir(parents=True, exist_ok=True)
            temporary = status_file.with_suffix(status_file.suffix + ".tmp")
            value = dict(ready=ready, pid=os.getpid(), gamepad=device, gear=GEARS[gear], control_steps=steps,
                         failure=playback.failure)
            temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
            try:
                temporary.replace(status_file)
            except PermissionError:
                # Windows readers can briefly prevent replacing the status file.
                # The next refresh can publish without interrupting the simulation.
                pass

    device, gear, steps = None, 1, 0
    previous_buttons = 0
    with mujoco.viewer.launch_passive(playback.model, playback.data, show_left_ui=False, show_right_ui=False) as view:
        with view.lock():
            view.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            view.cam.trackbodyid = playback.torso.id
            view.cam.distance, view.cam.azimuth, view.cam.elevation = .82, 135, -56
            if playback.model.geom_type[playback.floor] == mujoco.mjtGeom.mjGEOM_PLANE:
                draw_grid(view.user_scn)
        view.sync()
        started = last_publish = time.monotonic()
        _, _, device = controls.gamepad()
        publish(True, device, gear, steps)
        print(f"VIEWER_READY. Gamepad: {device or 'not detected; connect a pad or use keyboard'}", flush=True)
        try:
            while view.is_running():
                tick = time.monotonic()
                if seconds is not None and tick - started >= seconds:
                    break
                axes, buttons, current_device = controls.gamepad()
                if device != current_device:
                    device = current_device
                    print(f"Gamepad: {device or 'disconnected'}", flush=True)
                key_axes, key_buttons = controls.keyboard()
                if any(key_axes):
                    axes = key_axes
                buttons |= key_buttons
                changed = buttons & ~previous_buttons
                previous_buttons = buttons
                if changed & (PAD_LOW | PAD_HIGH):
                    gear = min(2, max(0, gear + int(bool(changed & PAD_HIGH)) - int(bool(changed & PAD_LOW))))
                    print(f"Speed gear: {GEARS[gear]}", flush=True)
                with view.lock():
                    if changed & PAD_RESET:
                        playback.reset()
                    if playback.failure is None:
                        playback.step(stick_command(*axes, gear))
                        steps += 1
                        if playback.failure:
                            print(f"{playback.failure}，按手柄 A 或键盘 R 重置。", flush=True)
                view.sync()
                if tick - last_publish >= .5:
                    publish(True, device, gear, steps)
                    last_publish = tick
                time.sleep(max(0., playback.dt - (time.monotonic() - tick)))
        finally:
            publish(False, device, gear, steps)


def main():
    from spider_command_policy import load_actor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--check", action="store_true", help="Headless native replay and gamepad detection")
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--seconds", type=float, help="Optional bounded viewer session")
    parser.add_argument("--action-delay", type=int, choices=range(HISTORY_LENGTH), default=0,
                        help="Delay issued joint targets by this many control cycles")
    parser.add_argument("--sensor-delay", type=int, choices=range(HISTORY_LENGTH), default=0,
                        help="Delay observations from the sensors by this many control cycles")
    parser.add_argument("--noise-seed", type=int, help="Enable repeatable sensor noise with this seed")
    parser.add_argument("--winmm-yaw-axis", choices=("z", "r", "u", "v"), default="z", help="Right-stick axis for generic joysticks")
    args = parser.parse_args()
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be positive")
    if args.noise_seed is not None and args.noise_seed < 0:
        parser.error("--noise-seed must be nonnegative")
    playback = Playback(load_actor(args.checkpoint), action_delay=args.action_delay,
                        sensor_delay=args.sensor_delay, noise_seed=args.noise_seed)
    print("左摇杆前后／横移，右摇杆左右转向，LB/RB 换档，A 重置。键盘 W/S、A/D、Q/E，Z/X 换档，R 重置。", flush=True)
    if args.action_delay or args.sensor_delay or args.noise_seed is not None:
        print(f"动作延时 {args.action_delay} 周期，传感器延时 {args.sensor_delay} 周期，观测噪声种子 {args.noise_seed}",
              flush=True)
    with WindowsControls(args.winmm_yaw_axis) as controls:
        if args.check:
            print(f"Gamepad: {controls.gamepad()[2] or 'not detected'}", flush=True)
            run_check(playback)
        else:
            run_viewer(playback, controls, args.status_file, args.seconds)


if __name__ == "__main__":
    main()
