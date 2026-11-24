"""NumPy inference for the saved spider command-conditioned Actor."""

import numpy as np


OBSERVATION_SIZE = 61
ACTION_SIZE = 16
LAYER_SHAPES = ((OBSERVATION_SIZE, 32), (32, 32), (32, 2 * ACTION_SIZE))


def load_actor(path):
    with np.load(path, allow_pickle=False) as archive:
        layers = [(np.array(archive[f"w{i}"], dtype=np.float32), np.array(archive[f"b{i}"], dtype=np.float32))
                  for i in range(len(LAYER_SHAPES))]
    for (weights, bias), shape in zip(layers, LAYER_SHAPES):
        if weights.shape != shape or bias.shape != (shape[-1],):
            raise ValueError(f"Incompatible Actor layer shape in {path}")
        if not np.isfinite(weights).all() or not np.isfinite(bias).all():
            raise ValueError(f"Non-finite Actor weights in {path}")

    def actor(observation):
        current = np.asarray(observation, dtype=np.float32)
        if current.ndim < 1 or current.shape[-1] != OBSERVATION_SIZE:
            raise ValueError(f"Expected {OBSERVATION_SIZE} observation features")
        for weights, bias in layers[:-1]:
            logits = current @ weights + bias
            current = logits / (1.0 + np.exp(-np.clip(logits, -80., 80.)))
        weights, bias = layers[-1]
        return np.tanh((current @ weights + bias)[..., :ACTION_SIZE])

    return actor
