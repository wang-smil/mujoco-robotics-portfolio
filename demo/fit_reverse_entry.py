"""Fit and validate the reverse reference at locomotion transitions."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_author import STAGE, make_local_env, rollout
from diagnose_reverse_reference import ReferenceController
from fit_reverse_teacher import ActionBoundedCandidate
from probe_direction_experts import bridge_reverse
from run_combined_refine import tracks_command


def make_reverse_reference(env, parameters):
    reference = ReferenceController(env)
    observation = np.zeros(101)
    observation[6] = -.05
    poses = []
    for phase in range(env.PRM.nb_steps_in_period):
        env.imitation_i = phase
        poses.append(env.default_actuator + env.action_scale * reference.infer(observation))
    reverse = ActionBoundedCandidate(env, np.stack(poses), parameters)
    env.reverse_cadence = reverse.cadence
    env.phase_frequency_factor = 1.
    return reverse


class ReverseEntry:
    def __init__(self, env, positive, reverse, velocity_gain, entry_ticks, trigger, record=True):
        self.positive, self.reverse = positive, reverse
        self.env = env
        self.velocity_gain = velocity_gain
        self.entry_ticks = entry_ticks
        self.trigger = trigger
        self.record = record
        self.speed_address = int(env.model.sensor('local_linvel').adr[0])
        self.count, self.filtered_x = 0, 0.
        self.observations, self.actions = [], []

    def infer(self, obs):
        if obs[6] > -self.trigger:
            self.count = 0
            self.filtered_x = self.env.data.sensordata[self.speed_address]
            action = self.positive.infer(obs)
        else:
            self.count += 1
            vx = self.env.data.sensordata[self.speed_address]
            self.filtered_x += .12 * (vx - self.filtered_x)
            correction = np.clip(self.velocity_gain * (self.filtered_x - obs[6]), -.075, .075)
            reverse = self.reverse.infer(obs)
            reverse[[2, 11]] += correction / self.env.action_scale
            reverse[[4, 13]] -= .5 * correction / self.env.action_scale
            mix = min(self.count / self.entry_ticks, 1.)
            action = reverse
            if mix < 1.:
                base = self.positive.infer(obs)
                action = base + mix * (reverse - base)
        action = np.clip(action, -.99, .99)
        if self.record:
            self.observations.append(np.asarray(obs, dtype=np.float32))
            self.actions.append(np.asarray(action, dtype=np.float32))
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refine', action='store_true')
    args = parser.parse_args()
    env, _ = make_local_env(Path((STAGE / 'round11_run.txt').read_text().strip()), 'step_5079040', cpu_restore=True)
    baseline = env.policy
    teacher = json.loads((STAGE / 'reverse_teacher_refined_validated.json').read_text())[0]
    reverse = make_reverse_reference(env, teacher['parameters'])
    routes = {
        'forward': [('stop', 100), ('forward_010', 180), ('reverse', 220), ('forward_010', 180), ('stop', 120)],
        'turn': [('stop', 100), ('right', 180), ('reverse', 220), ('left', 180), ('stop', 120)],
        'arcs': [('stop', 100), ('walk_left', 180), ('reverse', 220), ('walk_right', 180), ('stop', 120)],
        'stand': [('stop', 160), ('reverse', 240), ('stop', 120)],
    }
    if args.refine:
        routes.update(
            home=[('reverse', 1000)],
            arcs_right=[('stop', 120), ('walk_right', 180), ('reverse', 240), ('walk_left', 180), ('stop', 120)],
            switches=[(name, 200) for name in ('stop', 'forward_010', 'left', 'forward_010', 'right',
                                             'reverse', 'walk_left', 'walk_right', 'stop')],
        )

    def trial(settings, key, seed, ramp, gather=False):
        gain, ticks, trigger, pause, entry_phase = settings
        controller = ReverseEntry(env, baseline, reverse, gain, ticks, trigger)
        env.policy = controller
        schedule = bridge_reverse(routes[key], pause)
        rows = rollout(env, seed, schedule, ramp=ramp, reverse_entry_phase=entry_phase)
        judged = [row for row in rows if row['steps'] >= 100]
        full = len(rows) == len(schedule) and not rows[-1]['terminated']
        n = sum(tracks_command(row) for row in judged)
        reverse_result = next((row for row in rows if row['segment'] == 'reverse'), None)
        if reverse_result:
            reverse_loss = 2. * abs(reverse_result['mean_twist'][0] + .05) + .3 * abs(reverse_result['mean_twist'][2])
        else:
            reverse_loss = 1.
        accepted = full and n == len(judged)
        if gather and accepted:
            outputs[0].extend(controller.observations)
            outputs[1].extend(controller.actions)
        if gather:
            return dict(seed=seed, route=key, ramp=ramp, passed=accepted, segments=rows)
        if args.refine:
            return (2. * (not full) + .35 * (len(judged) - n)
                    + .2 * (len(routes[key]) - len(judged)) + reverse_loss)
        return 2. * (not full) + (5 - n) * .2 + reverse_loss

    rng = np.random.default_rng(1984)
    if args.refine:
        previous = json.loads((STAGE / 'reverse_entry_fit.json').read_text())
        gain, ticks, threshold, _, phase = previous['selection']
        settings = [tuple(previous['selection'])]
        settings += [(gain, ticks, threshold, 0, float(value)) for value in np.linspace(.34, .66, 9)]
        settings += [(float(rng.normal(gain, .14)), int(rng.integers(2, 12)), float(rng.choice([.006, .012, .018, .024])),
                      int(rng.choice([0, 0, 0, 12])), float(rng.choice([0., phase]) + rng.normal(0., .055)) % 1.)
                     for _ in range(54)]
        screening = [(key, 3441 + index, bool(index % 2)) for index, key in enumerate(routes)]
        finalist_seeds, verified_seeds = (3451, 3452), range(3501, 3509)
    else:
        settings = [(0., 1, .04, pause, phase) for pause in (0, 25, 50) for phase in (None, 0., .5)]
        settings += [(float(rng.uniform(-.8, .8)), int(rng.choice([1, 5, 12])), float(rng.choice([.012, .025, .04])),
                      int(rng.choice([0, 25, 50])), [None, 0., .25, .5, .75][rng.integers(5)]) for _ in range(42)]
        screening = list(zip(('forward', 'turn', 'arcs', 'stand'), (3001, 3002, 3003, 3004), (True, False, True, True)))
        finalist_seeds, verified_seeds = (3031, 3032), range(3071, 3079)
    ranked = []
    for index, candidate in enumerate(settings):
        cost = float(np.mean([trial(candidate, key, seed, ramp) for key, seed, ramp in screening]))
        ranked.append((cost, candidate))
        if index % 10 == 0:
            print('Entry fits tested:', index + 1, 'lowest screening cost:', min(row[0] for row in ranked), flush=True)
    finalists = []
    for _, candidate in sorted(ranked, key=lambda row: row[0])[:8]:
        score = np.mean([trial(candidate, key, seed, ramp)
                         for key in routes for seed in finalist_seeds for ramp in (False, True)])
        finalists.append((float(score), candidate))
    selected = min(finalists, key=lambda row: row[0])[1]
    outputs = ([], [])
    verified = [trial(selected, key, seed, ramp, True)
                for key in routes for ramp in (False, True) for seed in verified_seeds]
    counts = {f"{key}:{ramp}": sum(row['passed'] for row in verified if row['route'] == key and row['ramp'] == ramp)
              for key in routes for ramp in (False, True)}
    print('Selected:', selected, 'independent passes by route:', counts, flush=True)
    report = dict(selection=selected, screening=ranked, finalists=finalists, verified=verified, counts=counts, samples=len(outputs[0]))
    stem = 'reverse_entry_refined' if args.refine else 'reverse_entry'
    (STAGE / f'{stem}_fit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if sum(counts.values()) >= .75 * len(verified) and min(counts.values()) >= 3:
        np.savez_compressed(STAGE / f'{stem}_demo_data.npz', obs=np.stack(outputs[0]),
                            action=np.stack(outputs[1]), cadence=np.asarray(reverse.cadence, dtype=np.float32))


if __name__ == '__main__':
    main()
