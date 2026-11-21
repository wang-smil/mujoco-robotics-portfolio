"""Imitate validated backward steps with replay of the existing actor's actions."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid

import jax
import jax.numpy as jp
import numpy as np
import optax
from brax.training import types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import losses, networks
from flax.training import orbax_utils
from orbax import checkpoint as ocp

from benchmark_author import STAGE, load_module, make_author_env, rollout


def restore_actor(lesson, run, checkpoint):
    config = json.loads((run / 'config.json').read_text())
    lesson.validate_sources(config)
    net = lesson.network_from_config(config)
    observation_shape = {'state': (101,), 'privileged_state': (212,)}
    preprocess = (running_statistics.normalize if config['ppo']['normalize_observations']
                  else types.identity_observation_preprocessor)
    ppo = networks.make_ppo_networks(observation_shape, 14, preprocess_observations_fn=preprocess,
        policy_hidden_layer_sizes=tuple(net['policy_hidden']), value_hidden_layer_sizes=tuple(net['value_hidden']),
        policy_obs_key=net['policy_obs_key'], value_obs_key=net['value_obs_key'])
    obs_specs = {name: specs.Array(shape, jp.dtype('float32')) for name, shape in observation_shape.items()}
    target = (running_statistics.init_state(obs_specs), losses.PPONetworkParams(
        policy=ppo.policy_network.init(jax.random.PRNGKey(14)), value=ppo.value_network.init(jax.random.PRNGKey(27))))
    checkpointer = ocp.PyTreeCheckpointer()
    normalizer, params = checkpointer.restore(str(checkpoint), item=target,
        restore_args=orbax_utils.restore_args_from_target(target))
    return ppo, normalizer, params, config, checkpointer


def replay_baseline(predict, actor, config, source):
    output = STAGE / 'nonreverse_actor_replay.npz'
    if output.is_file():
        data = np.load(output)
        if str(data['source']) == str(source):
            return data['obs'], data['action']
    project = source.parents[4]
    sys.path.extend(str(path) for path in (project / '.venv-author-audit/lib').glob('python*/site-packages'))
    env, _ = make_author_env()
    env.max_motor_velocity = np.asarray(config['environment']['max_motor_velocity'])

    class Recorder:
        def __init__(self):
            self.obs = []
            self.actions = []

        def infer(self, value):
            obs = value.astype(np.float32, copy=True)
            obs[3] -= 1.3
            action = np.asarray(predict(actor, obs))
            self.obs.append(obs)
            self.actions.append(action)
            return action

    observations, actions = [], []
    modes = ('stop', 'forward_005', 'forward_010', 'forward_015', 'left', 'right', 'walk_left', 'walk_right')
    for name in modes:
        for seed in (1401, 1402, 1403):
            recorder = Recorder()
            env.policy = recorder
            trial = rollout(env, seed, [(name, 500)])[0]
            if not trial['terminated']:
                observations.extend(recorder.obs)
                actions.extend(recorder.actions)
    obs, act = np.stack(observations), np.stack(actions)
    np.savez_compressed(output, obs=obs, action=act, source=str(source))
    print('Completed baseline replay samples:', len(obs), flush=True)
    return obs, act


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--transitions', type=Path)
    parser.add_argument('--entry-focus', type=Path)
    parser.add_argument('--replay', type=Path, help='Use pre-recorded positive-command observations')
    parser.add_argument('--run-pointer', default='reverse_imitation_run.txt')
    parser.add_argument('--updates', type=int, default=2200)
    args = parser.parse_args()
    if args.entry_focus and not args.transitions:
        parser.error('--entry-focus requires --transitions')
    root = Path.cwd()
    source = args.source.resolve()
    source_run = source.parent.parent
    lesson = load_module('duck_lesson', root / 'lessons/07_duck_ppo.py')
    ppo, normalizer, original, config, checkpointer = restore_actor(lesson, source_run, source)
    teacher = np.load(args.teacher)
    assert teacher['obs'].shape[1:] == (101,) and teacher['action'].shape[1:] == (14,)
    cadence = float(teacher['cadence'])
    if not 1.0 <= cadence <= 1.25:
        raise ValueError(f'Teacher cadence {cadence} lies outside the supported range')

    @jax.jit
    def predict(weights, obs):
        full = {'state': obs, 'privileged_state': jp.zeros(obs.shape[:-1] + (212,), dtype=obs.dtype)}
        logits = ppo.policy_network.apply(normalizer, weights, full)
        return ppo.parametric_action_distribution.mode(logits)

    if args.replay:
        with np.load(args.replay, allow_pickle=False) as replay:
            replay_obs, replay_targets = replay['obs'], replay['action']
        assert replay_obs.shape[1:] == (101,) and replay_targets.shape == (len(replay_obs), 14)
        assert np.isfinite(replay_obs).all() and np.isfinite(replay_targets).all()
    else:
        replay_obs, replay_targets = replay_baseline(predict, original.policy, config, source)
    train_obs, targets = teacher['obs'], teacher['action']
    teacher_count = len(train_obs)
    focus_rows = None
    if args.transitions:
        transitions = np.load(args.transitions)
        assert float(transitions['cadence']) == cadence
        if args.entry_focus:
            with np.load(args.entry_focus, allow_pickle=False) as focus:
                indices = focus['transition_row']
                np.testing.assert_array_equal(transitions['obs'][indices], focus['obs'])
                np.testing.assert_array_equal(transitions['action'][indices], focus['action'])
                focus_rows = indices + teacher_count
        train_obs = np.concatenate([train_obs, transitions['obs']])
        targets = np.concatenate([targets, transitions['action']])
    targets = np.clip(targets, -.99, .99)
    rng = np.random.default_rng(615)
    train_index = rng.permutation(len(train_obs))
    first = int(len(train_index) * .85)
    train_rows, eval_rows = train_index[:first], train_index[first:]
    reverse_eval_rows = eval_rows[eval_rows < teacher_count]
    transition_eval_rows = eval_rows[eval_rows >= teacher_count]
    if args.transitions:
        reverse_train_rows = train_rows[train_rows < teacher_count]
        transition_train_rows = train_rows[train_rows >= teacher_count]
    if focus_rows is not None:
        focus_train_rows = np.intersect1d(focus_rows, transition_train_rows, assume_unique=True)
        assert len(focus_train_rows) >= 128
    optimizer = optax.adam(2e-4)
    optim_state = optimizer.init(original.policy)

    def loss(weights, reverse_observation, reverse_target, other_observation, other_target):
        reverse_mse = jp.mean(jp.square(predict(weights, reverse_observation) - reverse_target))
        replay_mse = jp.mean(jp.square(predict(weights, other_observation) - other_target))
        return reverse_mse + 1.5 * replay_mse

    @jax.jit
    def improve(weights, state, rev_obs, rev_target, old_obs, old_target):
        gradients = jax.grad(loss)(weights, rev_obs, rev_target, old_obs, old_target)
        delta, state = optimizer.update(gradients, state, weights)
        return optax.apply_updates(weights, delta), state

    def measure(weights):
        def mse(rows):
            return float(np.mean((np.asarray(predict(weights, train_obs[rows])) - targets[rows])**2))
        result = dict(reverse_mse=mse(reverse_eval_rows),
                      replay_mse=float(np.mean((np.asarray(predict(weights, replay_obs[::3])) - replay_targets[::3])**2)))
        if len(transition_eval_rows):
            result['transition_mse'] = mse(transition_eval_rows)
        return result

    weights = original.policy
    before = measure(weights)
    for tick in range(args.updates):
        if focus_rows is not None:
            reverse_rows = np.concatenate([rng.choice(reverse_train_rows, 128),
                rng.choice(transition_train_rows, 64), rng.choice(focus_train_rows, 64)])
        elif args.transitions:
            reverse_rows = np.concatenate([rng.choice(reverse_train_rows, 128), rng.choice(transition_train_rows, 128)])
        else:
            reverse_rows = rng.choice(train_rows, 256)
        replay_rows = rng.integers(len(replay_obs), size=256)
        weights, optim_state = improve(weights, optim_state, train_obs[reverse_rows], targets[reverse_rows],
                                      replay_obs[replay_rows], replay_targets[replay_rows])
        if (tick + 1) % 400 == 0:
            print('Actor imitation update', tick + 1, measure(weights), flush=True)
    after = measure(weights)
    assert after['reverse_mse'] < before['reverse_mse'] and after['replay_mse'] < .02, after
    unique = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6]
    run = root / 'artifacts/lesson07' / f'run_imitation_{unique}'
    checkpoint = run / 'checkpoints/initial'
    checkpoint.parent.mkdir(parents=True)
    result = copy.deepcopy(config)
    if not result['environment'].get('balance_observations', False):
        result['task_profile'] = 'neck_combined'
        result['source_sha256'] = lesson.source_manifest('neck_combined')
        result['environment'].update(turn_command_hold_steps=200, reverse_probability=.15,
                                     reverse_speed=[.045, .055], reverse_cadence=cadence)
    else:
        assert abs(float(result['environment']['reverse_cadence']) - cadence) < 1e-6
    source_steps = 0 if source.name == 'initial' else int(source.name.removeprefix('step_'))
    result.update(final_checkpoint='checkpoints/initial', prior_environment_steps=config.get('prior_environment_steps', 0) + source_steps,
        supervised_warm_start=dict(source=str(source), dataset=str(args.teacher.resolve()), optimization_steps=args.updates,
                                  teacher_examples=teacher_count, replay_examples=len(replay_obs), before=before, after=after,
                                  transition_examples=len(train_obs) - teacher_count))
    if focus_rows is not None:
        result['supervised_warm_start']['entry_focus'] = dict(path=str(args.entry_focus.resolve()), examples=len(focus_rows))
    if args.replay:
        result['supervised_warm_start']['replay_source'] = str(args.replay.resolve())
    result['cumulative_environment_steps'] = result['prior_environment_steps']
    lesson.write_json(run / 'config.json', result)
    checkpointer.save(str(checkpoint), (normalizer, original.replace(policy=weights)))
    (STAGE / Path(args.run_pointer).name).write_text(str(run) + '\n')
    print('Saved imitation initializer:', checkpoint, after, flush=True)


if __name__ == '__main__':
    main()
