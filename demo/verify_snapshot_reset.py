"""Check saved contact poses, action history and command at the MJX reset boundary."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from benchmark_author import load_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    lesson = load_module('duck_lesson', Path.cwd().parent / 'lessons/07_duck_ppo.py')
    lesson.jp = jp
    cfg = json.loads((args.run / 'config.json').read_text())
    cfg['task_profile'] = 'neck_combined_snapshot_reset'
    fixed = lesson.make_environment(cfg, command=(-.05, 0., 0.))
    assert fixed._config.reverse_snapshot_probability == 0.
    env = lesson.make_environment(cfg)
    env._config.reverse_snapshot_probability = 1.
    env._config.noise_config.level = 0.
    env._config.push_config.enable = False
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    verified = 0
    for seed in range(5101, 5109):
        rng = jax.random.PRNGKey(seed)
        row_key = jax.random.split(rng, 3)[2]
        i = int(jax.random.randint(row_key, (), 0, len(env.reverse_snapshots['qpos'])))
        expected = {key: values[i] for key, values in env.reverse_snapshots.items()}
        initial = reset(rng)
        seen_phase = float(initial.info['imitation_i'])
        if not np.allclose(initial.data.qpos[2:], expected['qpos'][2:], atol=1e-7):
            continue
        np.testing.assert_array_equal(initial.data.qvel, expected['qvel'])
        np.testing.assert_array_equal(initial.data.ctrl, expected['ctrl'])
        np.testing.assert_array_equal(initial.info['command'], expected['command'])
        np.testing.assert_array_equal(initial.info['last_last_act'], expected['action_history'][1])
        assert np.isclose(seen_phase, float(expected['phase']))
        np.testing.assert_allclose(initial.obs['state'][6:9], expected['command'][:3], atol=1e-7)
        features = jp.concatenate((env.get_gravity(initial.data)[:2], env.get_local_linvel(initial.data)[:2]))
        np.testing.assert_array_equal(initial.obs['state'][9:13], features)
        np.testing.assert_array_equal(initial.obs['state'][-2:], initial.info['_episode_reset']['imitation_phase'])
        assert not bool(env._get_termination(initial.data))
        next_state = step(initial, expected['action_history'][0])
        assert np.isfinite(np.asarray(next_state.reward))
        verified += 1
    assert verified >= 6, verified
    lesson.disable_assisted_resets(env)
    command = jp.array([.10, 0., -.2])
    live = jax.jit(lambda key: lesson.patch_evaluation_command(env, env.reset(key), command))(jax.random.PRNGKey(5512))
    np.testing.assert_array_equal(live.info['command_target'][:3], command)
    np.testing.assert_array_equal(live.obs['state'][6:9], command)
    np.testing.assert_array_equal(live.obs['state'][9:13], jp.concatenate((
        env.get_gravity(live.data)[:2], env.get_local_linvel(live.data)[:2])))
    assert not bool(live.info['command_ramp_active'])
    assert float(live.info['imitation_i']) == 0.
    print(f'Physical reset validation passed for {verified}/8 sampled snapshots.', flush=True)


if __name__ == '__main__':
    main()
