"""Compare the published walking policy on the matched handpad routes."""
import json

from benchmark_author import STAGE, make_author_env, rollout
from run_combined_refine import tracks_command


def main():
    matched = json.loads((STAGE / 'reverse_moving_entry_matched.json').read_text())
    env, identity = make_author_env()
    assert abs(env.sim_dt * env.decimation - .02) < 1e-9
    results = {}
    for route, versions in matched['matched'].items():
        reference = versions['original']
        assert all(trial['complete'] and trial['ramp'] for trial in reference)
        schedule = [(segment['segment'], segment['steps']) for segment in reference[0]['segments']]
        assert all([(segment['segment'], segment['steps']) for segment in row['segments']] == schedule
                   for row in reference)
        episodes = []
        for trial in reference:
            segments = rollout(env, trial['seed'], schedule, ramp=True)
            complete = len(segments) == len(schedule) and not segments[-1]['terminated']
            passed = complete and all(tracks_command(segment) for segment in segments if segment['steps'] >= 100)
            episodes.append(dict(seed=trial['seed'], complete=complete, passed=passed, segments=segments))
        results[route] = episodes
        print(route, sum(row['complete'] for row in episodes), sum(row['passed'] for row in episodes),
              '/', len(episodes), flush=True)
    output = STAGE / 'author_moving_entry_matched.json'
    output.write_text(json.dumps(dict(identity=identity, matches=results), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
