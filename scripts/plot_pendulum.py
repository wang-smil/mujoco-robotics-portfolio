"""Render the measured P, PD, and gravity-compensated pendulum traces."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def trace(relative):
    with (ROOT / 'evidence' / relative).open(encoding='utf-8', newline='') as csv_file:
        rows = list(csv.DictReader(csv_file))
    return [float(row['time_s']) for row in rows], [float(row['q_rad']) for row in rows]


def main():
    panels = [('Zero gravity', ('lesson02/p_only.csv', 'P'), ('lesson02/pd.csv', 'PD')),
              ('With gravity', ('lesson03/pd_only.csv', 'PD'), ('lesson03/pd_gravity.csv', 'PD + feedforward'))]
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.1), constrained_layout=True, sharex=True)
    for ax, (title, first, second) in zip(axes, panels):
        for (path, label), color in zip((first, second), ('#b94e39', '#164b82')):
            ax.plot(*trace(path), label=label, color=color, linewidth=1.5)
        ax.axhline(.5, color='#333333', linestyle='--', linewidth=1, label='Target')
        ax.set(title=title, xlabel='Time (s)')
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel('Joint angle (rad)')
    figure.savefig(ROOT / 'docs/media/pendulum-controls.png', dpi=180)


if __name__ == '__main__':
    main()
