"""Measure the published ONNX with the author's native MuJoCo controller."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np


STAGE = Path(__file__).resolve().parent
PROJECT = STAGE.parent
TARGETS = {
    'stop': (0., 0., 0.), 'forward_005': (.05, 0., 0.),
    'forward_010': (.1, 0., 0.), 'forward_015': (.15, 0., 0.),
    'left': (0., 0., .2), 'right': (0., 0., -.2), 'reverse': (-.05, 0., 0.),
    'walk_left': (.10, 0., .20), 'walk_right': (.10, 0., -.20),
}


def reset(env, seed):
    model, data = env.model, env.data
    home = model.keyframe('home')
    mujoco.mj_resetDataKeyframe(model, data, home.id)
    rng = np.random.default_rng(seed)
    data.qpos[7:] += rng.uniform(-.01, .01, size=model.nq - 7)
    data.qvel[:6] += rng.uniform(-.01, .01, size=6)
    env.imitation_i = 0
    env.imitation_phase = np.zeros(2)
    env.last_action = np.zeros(model.nu)
    env.last_last_action = np.zeros(model.nu)
    env.last_last_last_action = np.zeros(model.nu)
    env.motor_targets = home.ctrl.copy()
    env.prev_motor_targets = home.ctrl.copy()
    mujoco.mj_forward(model, data)


def add_balance_observations(observation, model, data):
    up = int(model.sensor('upvector').adr[0])
    velocity = int(model.sensor('local_linvel').adr[0])
    observation[9:11] = data.sensordata[up:up + 2]
    observation[11:13] = data.sensordata[velocity:velocity + 2]
    return observation


def motion_failed(data, up_z):
    return (data.qpos[2] < .10 or data.sensordata[up_z] < .5
            or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all())


def apply_native_control(env, applied, control_dt, was_reversing, reverse_entry_phase):
    reversing = applied[0] < -.01
    if reverse_entry_phase is not None and reversing and not was_reversing:
        phase = reverse_entry_phase(env) if callable(reverse_entry_phase) else reverse_entry_phase
        if phase is not None:
            env.imitation_i = phase * env.PRM.nb_steps_in_period
    cadence = getattr(env, 'reverse_cadence', env.phase_frequency_factor) if reversing else env.phase_frequency_factor
    env.imitation_i = (env.imitation_i + cadence) % env.PRM.nb_steps_in_period
    phase = env.imitation_i / env.PRM.nb_steps_in_period * 2 * np.pi
    env.imitation_phase = np.array([np.cos(phase), np.sin(phase)])
    action = env.policy.infer(env.get_obs(env.data, [*applied, 0., 0., 0., 0.]))
    env.last_last_last_action = env.last_last_action
    env.last_last_action = env.last_action
    env.last_action = action
    targets = env.default_actuator + action * env.action_scale
    delta = env.max_motor_velocity * control_dt
    env.motor_targets = np.clip(targets, env.prev_motor_targets - delta, env.prev_motor_targets + delta)
    env.prev_motor_targets = env.motor_targets.copy()
    env.data.ctrl[:] = env.motor_targets
    return reversing


def rollout(env, seed, schedule, ramp=False, brake_direction=False, reverse_entry_phase=None, trace=None,
            reverse_yaw_gate=None, command_filter=None):
    reset(env, seed)
    model, data = env.model, env.data
    origin = data.qpos[:2].copy()
    local_speed = int(model.sensor('local_linvel').adr[0])
    up_z = int(model.sensor('upvector').adr[0]) + 2
    control_dt = env.sim_dt * env.decimation
    applied = np.zeros(3)
    rate_limit = np.array([.3, .3, 1.]) * control_dt
    previous_target = np.zeros(3)
    was_reversing = False
    records = []
    for name, steps in schedule:
        command = TARGETS[name]
        target = np.asarray(command)
        entering_reverse = target[0] < -.01 and previous_target[0] >= -.01
        leaving_reverse = previous_target[0] < -.01 and target[0] >= -.01
        brake = brake_direction and bool(np.any(previous_target)) and (entering_reverse or leaving_reverse)
        twist = []
        farthest = 0.
        failed = False
        for tick in range(steps):
            mujoco.mj_step(model, data, nstep=env.decimation)
            twist.append([*data.sensordata[local_speed:local_speed + 2], env.get_gyro(data)[2]])
            farthest = max(farthest, float(np.linalg.norm(data.qpos[:2] - origin)))
            failed = motion_failed(data, up_z)
            if trace is not None:
                trace['qpos'].append(data.qpos.copy())
                trace['qvel'].append(data.qvel.copy())
                trace['ctrl'].append(data.ctrl.copy())
                trace['done'].append(bool(failed))
                if 'target_command' in trace:
                    trace['target_command'].append(target.copy())
            if failed:
                break
            filtered_target = (command_filter(env, target, previous_target, tick)
                               if command_filter is not None else target)
            if brake and tick < 20:
                applied = np.zeros(3)
            elif ramp and not (brake_direction and target[0] < -.01):
                applied = applied + np.clip(filtered_target - applied, -rate_limit, rate_limit)
            else:
                applied = filtered_target
            if ramp and reverse_yaw_gate is not None and target[0] < -.01 and abs(applied[2]) > reverse_yaw_gate:
                applied[0] = max(applied[0], 0.)
            was_reversing = apply_native_control(env, applied, control_dt, was_reversing, reverse_entry_phase)
        settled = np.asarray(twist[50:] if len(twist) > 50 else twist)
        error = settled - np.asarray(command)
        records.append(dict(segment=name, command=command, steps=len(twist), terminated=bool(failed),
            mean_twist=settled.mean(axis=0).tolist(), yaw_abs_mean_error=abs(float(error[:, 2].mean())),
            planar_mean_error=float(np.linalg.norm(error[:, :2].mean(axis=0))), farthest_from_start_m=farthest))
        if failed:
            break
        previous_target = target
    return records


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_author_env():
    import onnxruntime as ort

    sys.path.insert(0, str(PROJECT))
    from bootstrap_duck import AUTHOR_SHA

    vendor = PROJECT / 'third_party/Open_Duck_Playground'
    model_path = PROJECT / 'references/author_walk_policy.onnx'
    expected = AUTHOR_SHA
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != expected:
        raise ValueError('Author ONNX checksum mismatch')
    sys.path.insert(0, str(vendor))
    from playground.open_duck_mini_v2 import mujoco_infer

    class OneThreadPolicy:
        def __init__(self, path, awd=False):
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(path, sess_options=options, providers=['CPUExecutionProvider'])

        def infer(self, obs):
            return self.session.run(None, {'obs': np.asarray(obs, dtype=np.float32)[None]})[0][0]

    mujoco_infer.OnnxInfer = OneThreadPolicy
    model = vendor / 'playground/open_duck_mini_v2'
    env = mujoco_infer.MjInfer(str(model / 'xmls/scene_flat_terrain.xml'),
                              str(model / 'data/polynomial_coefficients.pkl'), str(model_path), False)
    return env, dict(model='BEST_WALK_ONNX_2.onnx', policy_sha256=expected, accelerometer_bias=1.3)


def make_local_env(run, checkpoint_name, cpu_restore=False):
    project = Path.cwd()
    lesson = load_module('duck_lesson', project / 'lessons/07_duck_ppo.py')
    config = json.loads((run / 'config.json').read_text())
    lesson.validate_sources(config)
    selected = f'checkpoints/{checkpoint_name}' if checkpoint_name else config['final_checkpoint']
    checkpoint = lesson.latest_checkpoint(run, selected)
    if checkpoint_name and checkpoint.name != checkpoint_name:
        raise FileNotFoundError(checkpoint_name)
    if cpu_restore:
        import jax
        from brax.training.agents.ppo.networks import make_inference_fn
        from pretrain_reverse_actor import restore_actor
        ppo, normalizer, weights, _, _ = restore_actor(lesson, run, checkpoint)
        make_policy = make_inference_fn(ppo)
        params = normalizer, weights.policy, weights.value
        critic_shape = (212,)
        sys.path.insert(0, str(project / 'duck_training'))
        os.chdir(project / 'duck_training')
    else:
        lesson.load_runtime()
        training_env = lesson.make_environment(config)
        make_policy, params = lesson.restore(training_env, config['ppo'], checkpoint, lesson.network_from_config(config))
        jax = lesson.jax
        critic_shape = training_env.observation_size['privileged_state']
    cpu = jax.devices('cpu')[0]
    params = jax.device_put(params, cpu)
    policy = jax.jit(make_policy(params, deterministic=True))

    class LocalActor:
        def __init__(self):
            self.key = jax.device_put(jax.random.PRNGKey(0), cpu)
            self.critic = jax.device_put(np.zeros(critic_shape, np.float32), cpu)

        def infer(self, obs):
            actor_obs = jax.device_put(np.asarray(obs, dtype=np.float32), cpu)
            action, _ = policy({'state': actor_obs, 'privileged_state': self.critic}, self.key)
            return np.asarray(action)

    # Native upstream imports ONNX Runtime even when its policy is replaced.
    sys.path.extend(str(p) for p in (project / '.venv-author-audit/lib').glob('python*/site-packages'))
    from playground.open_duck_mini_v2 import mujoco_infer
    mujoco_infer.OnnxInfer = lambda *_args, **_kwargs: LocalActor()
    xml = project / 'duck_training/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml'
    official_xml = project / 'third_party/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml'
    assert xml.read_bytes() == official_xml.read_bytes()
    env = mujoco_infer.MjInfer(str(xml), str(project / 'duck_training/playground/open_duck_mini_v2/data/polynomial_coefficients.pkl'), '', False)
    author_observation = env.get_obs

    def get_obs(data, command):
        obs = author_observation(data, command)
        # Our MJX lesson's accelerometer bias update has no assignment.
        obs[3] -= 1.3
        if config['environment'].get('balance_observations', False):
            obs = add_balance_observations(obs, env.model, data)
        return obs

    env.get_obs = get_obs
    env.max_motor_velocity = np.asarray(config['environment']['max_motor_velocity'])
    env.reverse_cadence = config['environment'].get('reverse_cadence', 1.0)
    return env, dict(model=str(checkpoint), accelerometer_bias=0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', type=Path, default=STAGE / 'author_benchmark.json')
    parser.add_argument('--cpu-restore', action='store_true', help='Restore a local Actor without initializing MJX')
    parser.add_argument('--reverse-switches', action='store_true', help='Exercise only the trained direction and reverse transitions')
    parser.add_argument('--arc-switches', action='store_true', help='Exercise switches among forward walking, arcs and in-place turns')
    parser.add_argument('--extended-switches', action='store_true', help='Also test repeated arcs, straight reversals and turn-to-reverse')
    parser.add_argument('--reverse-sweep', action='store_true', help='Test several negative speeds and a forward-to-reverse sequence')
    parser.add_argument('--ramp', action='store_true', help='Apply the gamepad slew limits to command transitions')
    parser.add_argument('--paired-ramp', action='store_true', help='Evaluate abrupt and slewed commands using the same loaded policy')
    parser.add_argument('--brake-direction', action='store_true', help='Insert 0.4 s of stand before changes into or out of reverse')
    parser.add_argument('--reverse-entry-phase', type=float, help='Probe an explicit gait phase at reverse entry, from 0 to less than 1')
    parser.add_argument('--seed-start', type=int, default=1101)
    parser.add_argument('--seed-count', type=int)
    args = parser.parse_args()
    if args.reverse_entry_phase is not None and not 0. <= args.reverse_entry_phase < 1.:
        parser.error('--reverse-entry-phase must be in [0, 1)')
    args.output = args.output.resolve()
    env, model_metadata = make_local_env(args.run.resolve(), args.checkpoint, args.cpu_restore) if args.run else make_author_env()
    schedules = {key: [(key, 1000)] for key in TARGETS}
    schedules['switches'] = [(key, 200) for key in ('stop', 'forward_010', 'left', 'forward_010', 'right',
                                                     'reverse', 'walk_left', 'walk_right', 'stop')]
    if args.extended_switches:
        extra_routes = {
            'switches_repeated_arcs': ('stop', 'walk_right', 'reverse', 'walk_left', 'reverse', 'walk_right', 'stop'),
            'switches_straight_reverse': ('stop', 'forward_010', 'reverse', 'stop', 'reverse', 'forward_005', 'stop'),
            'switches_turn_reverse': ('stop', 'left', 'reverse', 'right', 'reverse', 'left', 'stop'),
        }
        schedules.update({key: [(move, 200) for move in route] for key, route in extra_routes.items()})
    if args.reverse_switches:
        schedules = {key: schedules[key] for key in ('stop', 'forward_010', 'left', 'right', 'reverse')}
        schedules['switches'] = [(key, 200) for key in ('stop', 'forward_010', 'reverse', 'stop',
                                                        'left', 'right', 'forward_010', 'reverse', 'stop')]
    if args.arc_switches:
        schedules.pop('reverse')
        schedules['switches'] = [(key, 200) for key in ('stop', 'forward_010', 'walk_left', 'forward_010',
                                                        'walk_right', 'left', 'right', 'stop')]
    if args.reverse_sweep:
        for name, speed in [('reverse_0015', .015), ('reverse_0025', .025), ('reverse_0035', .035),
                            ('reverse_0065', .065), ('reverse_0074', .074),
                            ('reverse_0090', .09)]:
            TARGETS[name] = (-speed, 0., 0.)
        schedules = {key: [(key, 500)] for key in TARGETS if key.startswith('reverse')}
        schedules['switches'] = [(key, 200) for key in ('stop', 'forward_010', 'reverse', 'stop')]
    seed_count = args.seed_count if args.seed_count is not None else (4 if args.reverse_sweep else 8)
    if seed_count <= 0:
        parser.error('--seed-count must be positive')
    seeds = list(range(args.seed_start, args.seed_start + seed_count))
    for ramp in (False, True) if args.paired_ramp else (args.ramp,):
        report = dict(**model_metadata,
            protocol='Native CPU MuJoCo, 50 Hz, nominal flat terrain, small uniform seed perturbations, model-specific target slew and accelerometer bias, no simulated delays or external pushes',
            seeds=seeds, ramp=ramp, brake_direction=args.brake_direction, reverse_entry_phase=args.reverse_entry_phase,
            episodes=[], summary={})
        for name, schedule in schedules.items():
            rows = [dict(seed=seed, case=name, segments=rollout(
                env, seed, schedule, ramp=ramp, brake_direction=args.brake_direction,
                reverse_entry_phase=args.reverse_entry_phase)) for seed in seeds]
            report['episodes'].extend(rows)
            survived = sum(len(row['segments']) == len(schedule) and not row['segments'][-1]['terminated'] for row in rows)
            report['summary'][name] = dict(survived=survived, cases=len(rows))
            if len(schedule) == 1:
                report['summary'][name]['mean_twist'] = np.mean([row['segments'][0]['mean_twist'] for row in rows], axis=0).tolist()
            print('ramp' if ramp else 'direct', name, report['summary'][name], flush=True)
        output = args.output.with_stem(f'{args.output.stem}_ramp') if args.paired_ramp and ramp else args.output
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
