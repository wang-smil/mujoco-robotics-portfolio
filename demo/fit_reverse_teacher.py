"""Search small reference-gait corrections before using a trajectory as a teacher."""
import argparse
import json

import numpy as np

from benchmark_author import STAGE, make_author_env, rollout
from diagnose_reverse_reference import ReferenceController


class Candidate:
    def __init__(self, env, poses, params):
        self.env = env
        self.poses = poses
        self.mean = poses.mean(axis=0)
        self.amplitude, self.posture, self.cadence, self.pitch_bias, self.pitch_gain, self.roll_gain = params[:6]
        self.yaw_bias, self.yaw_gain = params[6:8] if len(params) == 8 else (0., 0.)
        self.env.phase_frequency_factor = self.cadence

    def infer(self, obs):
        env = self.env
        index = int(env.imitation_i)
        targets = (env.default_actuator + self.posture * (self.mean - env.default_actuator)
                   + self.amplitude * (self.poses[index] - self.mean))
        w, x, y, z = env.data.qpos[3:7]
        pitch = np.arcsin(np.clip(2 * (w * y - x * z), -1., 1.))
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch_adjust = np.clip(self.pitch_bias + self.pitch_gain * (pitch + .08 * obs[1]), -.25, .25)
        roll_adjust = np.clip(self.roll_gain * (roll + .08 * obs[0]), -.20, .20)
        targets[[2, 11]] += pitch_adjust
        targets[[4, 13]] -= .5 * pitch_adjust
        targets[[1, 10]] += roll_adjust
        yaw_adjust = np.clip(self.yaw_bias + self.yaw_gain * obs[2], -.15, .15)
        targets[[0, 9]] += yaw_adjust
        return (targets - env.default_actuator) / env.action_scale


class ActionBoundedCandidate(Candidate):
    def infer(self, obs):
        return np.clip(super().infer(obs), -1. + 1e-5, 1. - 1e-5)


def score(rows, heading=False, refine=False):
    def cost(row):
        vx, vy, yaw = row['mean_twist']
        if refine:
            speed = abs(vx + .05)
            return (3. * row['terminated'] + 4. * max(speed - .012, 0.) + 2. * max(abs(yaw) - .055, 0.)
                    + .8 * max(abs(vy) - .012, 0.) + .5 * speed + .15 * abs(yaw))
        return 2. * row['terminated'] + abs(vx + .05) + .5 * abs(vy) + (.6 if heading else .12) * abs(yaw)
    return float(np.mean([cost(row) for row in rows]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bounded', action='store_true')
    parser.add_argument('--fixed-cadence', action='store_true')
    parser.add_argument('--heading', action='store_true')
    parser.add_argument('--refine-heading', action='store_true')
    args = parser.parse_args()
    if args.refine_heading and not (args.heading and args.bounded):
        parser.error('--refine-heading requires --heading and --bounded')
    if (args.fixed_cadence or args.heading) and not args.bounded:
        parser.error('Fixed cadence and heading objectives require --bounded')
    env, _ = make_author_env()
    ref = ReferenceController(env)
    command = np.zeros(101)
    command[6] = -.05
    poses = []
    for index in range(env.PRM.nb_steps_in_period):
        env.imitation_i = index
        poses.append(env.default_actuator + env.action_scale * ref.infer(command))
    poses = np.stack(poses)

    def evaluate(params, seeds, ticks):
        cls = ActionBoundedCandidate if args.bounded else Candidate
        env.policy = cls(env, poses, params)
        return [rollout(env, seed, [('reverse', ticks)])[0] for seed in seeds]

    low = np.array([.55, .35, .45, -.15, -1.25, -.85])
    high = np.array([1.9, 1.0, 1.15, .15, 1.25, .85])
    if args.heading:
        low = np.concatenate([low, [-.09, -.9]])
        high = np.concatenate([high, [.09, .9]])
    rng = np.random.default_rng(603)
    candidates = [np.array([1., 1., 1., 0., 0., 0.])]
    if args.bounded:
        replay = 'reverse_teacher_bounded_diagnostic.json' if args.heading else 'reverse_teacher_diagnostic.json'
        earlier = json.loads((STAGE / replay).read_text())
        candidates.extend(np.asarray(row['parameters']) for row in earlier)
    if args.heading:
        candidates = [np.concatenate([row, [0., 0.]]) for row in candidates]
        candidates.extend(np.clip(rng.normal(candidates[1], .12 * (high - low), (80, len(low))), low, high))
    candidates.extend(rng.uniform(low, high, (120, len(low))))
    if args.refine_heading:
        observed = json.loads((STAGE / 'reverse_teacher_heading_validated.json').read_text())
        center = min(observed, key=lambda row: (sum(trial['terminated'] for trial in row['trials']), -row['passed']))
        center = np.asarray(center['parameters'])
        spread = np.array([.075, .08, .014, .014, .12, .10, .028, .13])
        candidates = [center] + [np.asarray(row['parameters']) for row in observed]
        candidates.extend(np.clip(rng.normal(center, spread, (155, len(center))), low, high))
    ranked = []
    for generation in range(3):
        if args.fixed_cadence:
            candidates = [np.asarray(row).copy() for row in candidates]
            for row in candidates:
                row[2] = 1.
        seeds = [2311, 2312, 2313] if args.refine_heading else ([1741, 1742] if args.heading else
                ([1731, 1732] if args.bounded else [1701]))
        ranked.extend((score(evaluate(row, seeds, 480 if args.refine_heading else (400 if args.bounded else 250)),
                             heading=args.heading, refine=args.refine_heading), row) for row in candidates)
        ranked.sort(key=lambda result: result[0])
        elite = np.stack([row for _, row in ranked[:12]])
        spread = np.maximum(elite.std(axis=0), .06 * (high - low))
        candidates = np.clip(rng.normal(elite.mean(axis=0), spread, (80, len(low))), low, high)
        print('search round', generation, 'best screening loss', ranked[0][0], flush=True)
    results = []
    for _, params in ranked[:12]:
        selection = range(2321, 2325) if args.refine_heading else (range(1733, 1737) if args.bounded else range(1702, 1706))
        rows = evaluate(params, selection, 550 if args.refine_heading else 500)
        results.append(dict(parameters=params.tolist(), score=score(rows, args.heading, args.refine_heading), cases=rows))
    results.sort(key=lambda row: row['score'])
    best = results[0]
    print('Held-out average vx', np.mean([row['mean_twist'][0] for row in best['cases']]),
          'completed', sum(not row['terminated'] for row in best['cases']), flush=True)
    prefix = ('reverse_teacher_refined' if args.refine_heading else 'reverse_teacher_heading' if args.heading else 'reverse_teacher_fixed' if args.fixed_cadence
              else 'reverse_teacher_bounded' if args.bounded else 'reverse_teacher')
    output = f'{prefix}_diagnostic.json'
    (STAGE / output).write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
