"""Run the validated native MuJoCo assisted policy through the existing local gamepad."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import queue
import webbrowser
import time

import mujoco
import numpy as np

from benchmark_author import (
    STAGE, TARGETS, apply_native_control, load_module, make_local_env, motion_failed, reset, rollout,
)
from probe_adaptive_reverse import assisted_policy
from probe_direction_experts import bridge_reverse
from probe_moving_reverse import ROUTES, moving_policy, prepare_reference
from run_combined_refine import tracks_command


class Driver:
    def __init__(self, env, ramp, phase, pause, moving_entry=None):
        self.env, self.ramp, self.phase, self.pause = env, ramp, phase, pause
        self.dt = env.sim_dt * env.decimation
        self.speed = int(env.model.sensor('local_linvel').adr[0])
        self.up_z = int(env.model.sensor('upvector').adr[0]) + 2
        self.moving_entry = moving_entry

    def reset(self, seed):
        reset(self.env, seed)
        self.applied = np.zeros(3)
        self.request = np.zeros(3)
        self.remaining = 0
        self.reversing = False
        self.origin = self.env.data.qpos[:2].copy()
        self.yaw = None
        self.heading = 0.
        self.frames = 0
        if self.moving_entry:
            self.moving_entry.reset()
            self.filtered_source = np.zeros(3)
            self.filter_previous = np.zeros(3)
            self.command_tick = -1

    def tick(self, request):
        request = np.asarray(request)
        if not np.array_equal(request, self.request):
            crosses_zero = (self.request[0] < -.01) != (request[0] < -.01)
            self.remaining = self.pause if crosses_zero and np.any(self.request) and np.any(request) else 0
            self.request = request
        target = np.zeros(3) if self.remaining else request
        self.remaining = max(self.remaining - 1, 0)
        data = self.env.data
        mujoco.mj_step(self.env.model, data, nstep=self.env.decimation)
        self.frames += 1
        w, x, y, z = data.qpos[3:7]
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        if self.yaw is not None:
            self.heading += np.arctan2(np.sin(yaw - self.yaw), np.cos(yaw - self.yaw))
        self.yaw = yaw
        if motion_failed(data, self.up_z):
            return False
        if self.moving_entry:
            if np.array_equal(target, self.filtered_source):
                self.command_tick += 1
            else:
                self.command_tick = 0
                self.filter_previous = self.filtered_source
                self.filtered_source = target
            target = self.moving_entry.filter(self.env, target, self.filter_previous, self.command_tick)
        self.applied = self.ramp(self.applied, target, self.dt)
        self.reversing = apply_native_control(self.env, self.applied, self.dt, self.reversing, self.phase)
        return True

    def telemetry(self):
        data = self.env.data
        position = data.qpos[:2] - self.origin
        return dict(applied=self.applied.tolist(), actual_velocity=data.sensordata[self.speed:self.speed + 3].tolist(),
            actual_yaw_rate=float(self.env.get_gyro(data)[2]), position=position.tolist(),
            distance=float(np.linalg.norm(position)), heading_deg=float(np.degrees(self.heading)),
            sim_seconds=float(data.time), frames=self.frames)


def verify(driver, seed, sequence):
    order = ROUTES[sequence]
    original = [(name, 200) for name in order]
    schedule = bridge_reverse(original, driver.pause)
    trace = {name: [] for name in ('qpos', 'qvel', 'ctrl', 'done')}
    driver.reset(seed)
    command_filter = driver.moving_entry.filter if driver.moving_entry else None
    outcome = rollout(driver.env, seed, schedule, ramp=True, reverse_entry_phase=driver.phase,
                      trace=trace, command_filter=command_filter)
    assert len(outcome) == len(schedule) and all(tracks_command(row) for row in outcome if row['steps'] >= 100)
    driver.reset(seed)
    offset = 0
    lengths = [steps for _, steps in original]
    for i in range(1, len(order)):
        old, new = np.asarray(TARGETS[order[i - 1]]), np.asarray(TARGETS[order[i]])
        if (old[0] < -.01) != (new[0] < -.01) and np.any(old) and np.any(new):
            lengths[i] += driver.pause
    for name, steps in zip(order, lengths):
        for _ in range(steps):
            ctrl = driver.env.data.ctrl.copy()
            assert driver.tick(TARGETS[name])
            np.testing.assert_array_equal(driver.env.data.qpos, trace['qpos'][offset])
            np.testing.assert_array_equal(driver.env.data.qvel, trace['qvel'][offset])
            np.testing.assert_array_equal(ctrl, trace['ctrl'][offset])
            offset += 1
    assert offset == len(trace['qpos'])
    print('Live controller exactly reproduced the verified sequence:', offset, 'steps, seed', seed, flush=True)


def serve(project, game, driver, args, run, checkpoint):
    class AssistedControl(game.ControlState):
        def apply(self, action, speed=.1, now=None):
            with self.lock:
                if action in ('slide_left', 'slide_right'):
                    raise ValueError('此模式支持站立、前进、后退和左右转向')
                super().apply(action, speed, now)
                if not self.failed and action in ('backward', 'left', 'right', 'forward_left', 'forward_right'):
                    name = {'backward': 'reverse', 'left': 'left', 'right': 'right',
                            'forward_left': 'walk_left', 'forward_right': 'walk_right'}[action]
                    self.target = list(TARGETS[name])

    control, events = AssistedControl(), queue.SimpleQueue()
    html = (project / 'lessons/duck_gamepad.html').read_text()
    html = '\n'.join(line for line in html.splitlines() if 'data-action="slide' not in line)
    html = html.replace('小鸭子 · 电脑手柄', '小鸭子 · 姿态辅助手柄').replace(
        '一个策略，实时接收你的速度命令。', '前进速度可调。后退为 0.05 m/s，转向为 0.20 rad/s。')
    html = html.replace(' · 未训练方向', '').replace(' · 已训练范围', '').replace(
        '本模型正式训练了前进和停稳；虚线按钮可探索，但后退、横移、转弯可能跌倒，按 R 重置。',
        '平地仿真。前进与后退互换时有短暂的平衡过渡。按 R 重置。')
    server, url = game.start_server(control, html.encode(), args.port)
    env = driver.env
    driver.reset(args.seed)
    status = dict(url=url, run=str(run), checkpoint=checkpoint, seed=args.seed, ready=True, pid=os.getpid())
    args.status_file.parent.mkdir(parents=True, exist_ok=True)

    def publish():
        readings = driver.telemetry()
        if all(np.isfinite(value).all() for value in readings.values()):
            control.update(**readings)
        temporary = args.status_file.with_suffix('.tmp')
        temporary.write_text(json.dumps({**status, **control.snapshot()}, ensure_ascii=False, allow_nan=False))
        temporary.replace(args.status_file)

    keys = {87: 'forward', 83: 'backward', 65: 'left', 68: 'right', 81: 'forward_left',
            69: 'forward_right', 32: 'stop', 80: 'pause', 82: 'reset'}
    started = time.monotonic()
    last_reset, last_publish = 0, 0.
    try:
        context = nullcontext(None)
        if not args.headless:
            from mujoco import viewer
            context = viewer.launch_passive(env.model, env.data, key_callback=events.put,
                                             show_left_ui=False, show_right_ui=False)
        with context as handle:
            if handle:
                with handle.lock():
                    handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    handle.cam.trackbodyid = env.model.body('trunk_assembly').id
                    handle.cam.distance, handle.cam.azimuth, handle.cam.elevation = .85, 40, -15
                    handle.opt.geomgroup[3] = 0
            publish()
            print(f'手柄地址 {url}', flush=True)
            if args.open_panel:
                webbrowser.open(url)
            print(f'手柄已就绪，状态文件 {args.status_file}', flush=True)
            while not control.snapshot()['closing'] and (handle is None or handle.is_running()):
                now = time.monotonic()
                if args.seconds and now - started >= args.seconds:
                    break
                while not events.empty():
                    action = keys.get(events.get())
                    if action:
                        control.apply(action)
                current = control.snapshot()
                with handle.lock() if handle else nullcontext():
                    if current['reset_id'] != last_reset:
                        driver.reset(args.seed)
                        last_reset = current['reset_id']
                    if not current['paused'] and not driver.tick(current['target']):
                        control.fail('触发物理失败', reset_id=current['reset_id'])
                if handle:
                    handle.sync()
                if now - last_publish >= .1:
                    publish()
                    last_publish = now
                time.sleep(max(0., driver.dt - (time.monotonic() - now)))
    finally:
        control.apply('close')
        status['ready'] = False
        publish()
        server.shutdown()
        server.server_close()


def main():
    project = Path.cwd()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', type=Path, default=STAGE / 'reverse_adaptive_refine.json')
    parser.add_argument('--seed', type=int, default=5701)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--sequence', choices=tuple(ROUTES), default='standard')
    parser.add_argument('--moving-entry', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--open-panel', action='store_true')
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--status-file', type=Path, default=project / 'artifacts/gamepad/assisted_status.json')
    args = parser.parse_args()
    if args.seconds is not None and (not np.isfinite(args.seconds) or args.seconds <= 0):
        parser.error('--seconds must be positive')
    args.status_file = args.status_file.resolve()
    selected = json.loads(args.configuration.read_text())
    run, composition = Path(selected['source']), selected['best']
    env, identity = make_local_env(run, 'initial', cpu_restore=True)
    entry = None
    if args.moving_entry:
        verified = json.loads((STAGE / 'reverse_moving_entry_matched.json').read_text())
        if Path(verified['source']) != run:
            raise ValueError('The moving-entry settings were validated with a different actor')
        reference, params = prepare_reference(env)
        entry, phase = moving_policy(env, env.policy, composition, verified['selected'], reference, params)
        env.policy = entry
    else:
        env.policy, phase = assisted_policy(env, env.policy, composition)
    game = load_module('lesson_gamepad', project / 'lessons/08_duck_gamepad.py')
    driver = Driver(env, game.ramp_command, phase, composition[3], entry)
    if args.verify:
        verify(driver, args.seed, args.sequence)
    else:
        serve(project, game, driver, args, run, identity['model'])


if __name__ == '__main__':
    main()
