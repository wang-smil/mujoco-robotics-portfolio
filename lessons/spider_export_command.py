"""Save the selected Brax checkpoint and a verified NumPy Actor for Windows."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from brax.io import model as brax_model
from brax.training.agents.ppo import networks
import jax
import numpy as np

from spider_command_policy import ACTION_SIZE, OBSERVATION_SIZE, load_actor
from spider_command_task import COMMAND_SCALE, COMMAND_SLEW
from spider_train_commands import capped_exploration_network


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("demonstrations", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    source = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Output directory already exists: {output}")

    params = brax_model.load_params(str(source))
    layers = params[1]["params"]
    arrays = {}
    for index in range(3):
        layer = layers[f"hidden_{index}"]
        weights = np.asarray(layer["kernel"], dtype=np.float32)
        bias = np.asarray(layer["bias"], dtype=np.float32)
        if index == 2:
            # Flax emits means followed by scales. The stored Actor still has all training weights.
            assert weights.shape[-1] == ACTION_SIZE * 2
        arrays.update({f"w{index}": weights, f"b{index}": bias})

    with np.load(args.demonstrations, allow_pickle=False) as demo:
        observations = demo["observations"]
        observed = observations[np.linspace(0, len(observations) - 1, 512, dtype=int)]
    random = np.random.default_rng(4).normal(0., .7, (512, OBSERVATION_SIZE)).astype(np.float32)
    samples = np.concatenate((observed, random))
    make_policy = networks.make_inference_fn(capped_exploration_network(OBSERVATION_SIZE, ACTION_SIZE))
    expected = np.asarray(make_policy(params, deterministic=True)(jax.device_put(samples), jax.random.PRNGKey(0))[0])

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".spider-export-") as temporary:
        stage = Path(temporary) / "checkpoint"
        stage.mkdir()
        raw, portable = stage / "checkpoint.params", stage / "checkpoint.npz"
        shutil.copyfile(source, raw)
        np.savez_compressed(portable, **arrays)
        actual = load_actor(portable)(samples)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        error = float(np.max(np.abs(actual - expected)))
        metadata = dict(
            source_path=str(args.checkpoint), source_sha256=sha256(source), numpy_sha256=sha256(portable),
            ppo_control_steps=args.step, observation_size=OBSERVATION_SIZE, action_size=ACTION_SIZE,
            controller_dt_seconds=.02, physics_dt_seconds=.002, hidden_activation="swish",
            output_activation="tanh", observation_normalization=False, verification_observations=len(samples),
            max_absolute_action_error=error, command_scale=COMMAND_SCALE, slew_limits=COMMAND_SLEW,
        )
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        stage.rename(output)
    print(f"Saved {output}; maximum NumPy/Brax action error {error:.3g}", flush=True)


if __name__ == "__main__":
    main()
