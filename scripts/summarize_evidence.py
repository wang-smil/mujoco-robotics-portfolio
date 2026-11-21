"""Recompute published headline results using only the Python standard library."""
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / 'evidence' / relative).read_text(encoding='utf-8'))


def tally(rows):
    return {'passed': sum(row['passed'] for row in rows),
            'completed': sum(row['complete'] for row in rows), 'count': len(rows)}


def main():
    gravity = read('lesson03/checks.json')['cases']
    pendulum = read('lesson05/evaluation.json')
    paired = read('duck/reverse_moving_entry_matched.json')['matched']
    author = read('duck/author_moving_entry_matched.json')['matches']
    result = {
        'pendulum': {
            'absolute_final_error_rad': {case['case']: abs(case['final_error_rad']) for case in gravity},
            'ppo_steps': read('lesson05/checks.json')['actual_environment_steps'],
            'mean_test_return': {name: row['mean_return'] for name, row in pendulum.items()},
        },
        'duck': {},
    }
    for scenario, versions in paired.items():
        compared = {'before': versions['original'], 'after': versions['candidate'], 'author': author[scenario]}
        seed_sets = [tuple(row['seed'] for row in rows) for rows in compared.values()]
        assert seed_sets[0] == seed_sets[1] == seed_sets[2] and len(set(seed_sets[0])) == 20
        result['duck'][scenario] = {name: tally(rows) for name, rows in compared.items()}
    result['duck']['standard_reverse_vx_m_s'] = {
        name: mean(segment['mean_twist'][0] for trial in rows for segment in trial['segments']
                   if segment['segment'] == 'reverse' and segment['steps'] >= 100)
        for name, rows in {'after': paired['standard']['candidate'], 'author': author['standard']}.items()
    }
    result['duck']['round22_reverse'] = {
        'step': read('duck/round22_native_final.json')['summary']['reverse'],
        'ramp': read('duck/round22_native_final_ramp.json')['summary']['reverse'],
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
