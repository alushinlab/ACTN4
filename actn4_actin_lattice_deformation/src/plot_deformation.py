"""Stage 4b - ACTN4 occupancy versus curvature deformation score.

One figure per 3DVA component, with two panels: the score computed over the
whole filament and over the ACTN4-centred window.  Axis ranges are identical
in every panel of both components so the two components can be compared
directly by eye.

Usage:  python -m src.plot_deformation
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.stats import pearsonr, spearmanr

from . import config as C

FRAME_TICKS = [1, 5, 10, 15, 20]
PANELS = [('D_curv_global_invA', 'global filament'),
          ('D_curv_local_invA', 'local ACTN4 centered window')]
COLUMNS = ['frame', 'occupancy_pct', 'D_curv_global_invA', 'D_curv_local_invA']


def load_scores(component):
    path = C.paths(component)['measurements'].parent / 'curvature_deformation_scores.csv'
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    return {name: data[:, i] for i, name in enumerate(COLUMNS)}


def shared_xlim(scores):
    vals = np.concatenate([s[key] for s in scores.values() for key, _ in PANELS])
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def make_figure(component, s, xlim):
    out = C.paths(component)
    out['fig_deformation'].mkdir(parents=True, exist_ok=True)
    frames = s['frame'] + 1                      # display frames as 1..20
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)

    for a, (key, title) in zip(ax, PANELS):
        sc = a.scatter(s[key], s['occupancy_pct'], c=frames, cmap='viridis',
                       vmin=1, vmax=C.N_FRAMES, s=80, edgecolors='k', lw=0.5)
        r, _ = pearsonr(s[key], s['occupancy_pct'])
        rho, _ = spearmanr(s[key], s['occupancy_pct'])
        a.set_title(title, fontsize=12)
        a.set_xlabel('mean curvature  ($\\times 10^{-3}$ Å$^{-1}$)')
        a.set_xlim(*xlim)
        a.set_ylim(0, 100)
        a.xaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
        a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v * 1e3:g}'))
        a.set_yticks(range(0, 101, 20))
        a.tick_params(axis='both', labelsize=10)
        a.grid(alpha=0.25)
        a.text(0.03, 0.04, f'Pearson r = {r:+.2f}\nSpearman ρ = {rho:+.2f}',
               transform=a.transAxes, fontsize=9, va='bottom',
               bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'))
    ax[0].set_ylabel('ACTN4 occupancy %')

    cbar = fig.colorbar(sc, ax=ax, shrink=0.9, ticks=FRAME_TICKS)
    cbar.set_label('frame')
    cbar.ax.set_yticklabels([str(t) for t in FRAME_TICKS])

    fig.suptitle(C.COMPONENTS[component]['label'], fontsize=15, y=1.02)
    path = out['fig_deformation'] / 'occupancy_vs_curvature_deformation.png'
    fig.savefig(path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    return path


def run():
    scores = {c: load_scores(c) for c in C.COMPONENTS}
    xlim = shared_xlim(scores)
    print('shared curvature-deformation x-range: %.6f .. %.6f 1/A' % xlim)
    for component, s in scores.items():
        path = make_figure(component, s, xlim)
        r_g = pearsonr(s['D_curv_global_invA'], s['occupancy_pct'])[0]
        r_l = pearsonr(s['D_curv_local_invA'], s['occupancy_pct'])[0]
        print(f'{component}: occupancy {s["occupancy_pct"].min():.1f}-'
              f'{s["occupancy_pct"].max():.1f}% | r global {r_g:+.2f}, '
              f'local {r_l:+.2f} -> {path.name}')


if __name__ == '__main__':
    run()
