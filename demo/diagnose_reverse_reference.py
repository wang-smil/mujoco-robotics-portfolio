"""Check whether the existing reference alone launches a physical backward gait."""
import json

import numpy as np

from benchmark_author import STAGE, TARGETS, make_author_env, rollout


def axis_mix(nodes, value):
    value = np.clip(value, nodes[0], nodes[-1])
    right = np.clip(np.searchsorted(nodes, value, side='right'), 1, len(nodes) - 1)
    left = right - 1
    ratio = float((value - nodes[left]) / (nodes[right] - nodes[left]))
    return ((left, 1. - ratio), (right, ratio))


class ReferenceController:
    def __init__(self, env):
        self.env = env
        self.cache = {}

    def infer(self, obs):
        vx, vy, yaw = map(float, obs[6:9])
        phase = int(self.env.imitation_i)
        key = (vx, vy, yaw, phase)
        if key not in self.cache:
            data = self.env.PRM
            mixes = [axis_mix(nodes, v) for nodes, v in zip((data.dxs, data.dys, data.dthetas), key[:3])]
            t = phase / data.nb_steps_in_period
            frame = sum(wx * wy * wz * np.asarray(data.sample_polynomial(t, data.data_array[ix][iy][iz]))
                        for ix, wx in mixes[0] for iy, wy in mixes[1] for iz, wz in mixes[2])
            targets = np.concatenate([frame[:9], frame[11:16]])
            targets[5:9] = self.env.default_actuator[5:9]
            self.cache[key] = (targets - self.env.default_actuator) / self.env.action_scale
        return self.cache[key]


def main():
    env, _ = make_author_env()
    env.policy = ReferenceController(env)
    TARGETS['reverse_0074'] = (-.074, 0., 0.)
    samples = {}
    for target in ('forward_005', 'forward_010', 'reverse', 'reverse_0074'):
        for ramp in (False, True):
            rows = [rollout(env, seed, [(target, 500)], ramp=ramp)[0] for seed in range(1101, 1105)]
            key = f'{target}_ramp_{ramp}'
            samples[key] = rows
            print(key, sum(not row['terminated'] for row in rows),
                  np.mean([row['mean_twist'] for row in rows], axis=0).tolist(), flush=True)
    path = STAGE / 'reverse_reference_diagnostic.json'
    path.write_text(json.dumps(samples, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
