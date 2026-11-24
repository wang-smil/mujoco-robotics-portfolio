"""Fit a shared Brax PPO actor to bounded demonstration actions."""

import jax
import jax.numpy as jp
import numpy as np
import optax


def fit_warm_actor(initial_params, make_policy, network_factory, observations, actions, seed, steps=2500):
    """Return the warm parameter tuple and errors on disjoint demonstration rows."""
    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if (
        observations.ndim != 2
        or actions.ndim != 2
        or observations.shape[0] != actions.shape[0]
        or observations.shape[0] < 8
        or observations.shape[1] == 0
        or actions.shape[1] == 0
        or not np.isfinite(observations).all()
        or not np.isfinite(actions).all()
        or np.any(np.abs(actions) > 1.0)
        or steps < 1
    ):
        raise ValueError("Warm-start fitting needs finite observation rows, matching actions in [-1, 1], and positive steps")

    held_out = np.arange(len(observations)) % 8 == 0
    train_obs, val_obs = jp.asarray(observations[~held_out]), jp.asarray(observations[held_out])
    train_actions, val_actions = jp.asarray(actions[~held_out]), jp.asarray(actions[held_out])
    batch_size = min(256, len(train_obs))
    normalizer, initial_actor, value = initial_params
    networks = network_factory(observations.shape[1], actions.shape[1])
    distribution = networks.parametric_action_distribution
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(0.0015))
    key = jax.random.PRNGKey(seed)

    def loss(actor, obs, target):
        logits = networks.policy_network.apply(normalizer, actor, obs)
        prediction = distribution.mode(logits)
        std = distribution.create_dist(logits).scale
        # A modest exploration spread avoids erasing a warm actor with large PPO noise.
        return jp.mean(jp.square(prediction - target)) + 0.05 * jp.mean(jp.square(std - 0.10))

    @jax.jit
    def fit(actor, obs, target):
        def update(_, carry):
            actor, opt_state, rng = carry
            rng, batch_key = jax.random.split(rng)
            batch = jax.random.randint(batch_key, (batch_size,), 0, obs.shape[0])
            gradient = jax.grad(loss)(actor, obs[batch], target[batch])
            updates, opt_state = optimizer.update(gradient, opt_state, actor)
            return optax.apply_updates(actor, updates), opt_state, rng

        return jax.lax.fori_loop(0, steps, update, (actor, optimizer.init(actor), key))[0]

    @jax.jit
    def rmse(actor, obs, target):
        policy = make_policy((normalizer, actor, value), deterministic=True)
        prediction, _ = policy(obs, key)
        return jp.sqrt(jp.mean(jp.square(prediction - target)))

    fitted = fit(initial_actor, train_obs, train_actions)
    initial_error = float(rmse(initial_actor, val_obs, val_actions))
    fitted_error = float(rmse(fitted, val_obs, val_actions))
    if not np.isfinite([initial_error, fitted_error]).all():
        raise RuntimeError("Non-finite behavior-cloning validation error")
    selected = fitted_error <= initial_error
    warm_actor = fitted if selected else initial_actor
    train_error = float(rmse(warm_actor, train_obs, train_actions))
    if not np.isfinite(train_error):
        raise RuntimeError("Non-finite behavior-cloning training error")

    report = {
        "optimizer_steps": int(steps),
        "train_samples": len(train_obs),
        "validation_samples": len(val_obs),
        "train_rmse": train_error,
        "initial_validation_rmse": initial_error,
        "fitted_validation_rmse": fitted_error,
        "validation_rmse": min(initial_error, fitted_error),
        "accepted_fitted_actor": bool(selected),
    }
    return (normalizer, warm_actor, value), report
