"""Keep a learned gait represented in the Actor during PPO refinement."""
from contextlib import contextmanager
import hashlib
import math

import jax
import jax.numpy as jp
import numpy as np


def load_demonstrations(paths, cadence):
    data, metadata = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as dataset:
            obs = np.asarray(dataset['obs'], dtype=np.float32)
            action = np.asarray(dataset['action'], dtype=np.float32)
            if 'cadence' in dataset and not math.isclose(float(dataset['cadence']), cadence, abs_tol=1e-5):
                raise ValueError(f'Demonstration phase frequency differs from environment: {path}')
            has_cadence = 'cadence' in dataset
        if (obs.ndim != 2 or obs.shape[1] != 101 or action.shape != (len(obs), 14) or len(obs) < 128
                or not np.isfinite(obs).all() or not np.isfinite(action).all() or np.max(np.abs(action)) > 1.):
            raise ValueError(f'Invalid demonstration arrays: {path}')
        if not has_cadence and np.any(obs[:, 6] < -.01):
            raise ValueError(f'Reverse demonstrations must identify their phase frequency: {path}')
        data.append((jp.asarray(obs), jp.asarray(np.clip(action, -.99, .99))))
        metadata.append(dict(path=str(path.resolve()), observations=len(obs), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return data, metadata


@contextmanager
def anchor_loss(loss_module, datasets, weight, *, format_observation=None):
    original = loss_module.compute_ppo_loss

    def compute(params, normalizer_params, data, rng, ppo_network, **kwargs):
        total, metrics = original(params, normalizer_params, data, rng, ppo_network, **kwargs)
        demo_losses = []
        for i, (observations, actions) in enumerate(datasets):
            indices = jax.random.randint(jax.random.fold_in(rng, 8191 + i), (128,), 0, len(observations))
            sample = observations[indices]
            if format_observation is None:
                obs = {'state': sample, 'privileged_state': jp.zeros((128, 212), dtype=jp.float32)}
            else:
                obs = format_observation(sample)
            logits = ppo_network.policy_network.apply(normalizer_params, params.policy, obs)
            prediction = ppo_network.parametric_action_distribution.mode(logits)
            demo_losses.append(jp.mean(jp.square(prediction - actions[indices])))
        imitation = jp.mean(jp.stack(demo_losses))
        total += weight * imitation
        return total, {**metrics, 'total_loss': total, 'demonstration_mse': imitation}

    loss_module.compute_ppo_loss = compute
    try:
        yield
    finally:
        loss_module.compute_ppo_loss = original
