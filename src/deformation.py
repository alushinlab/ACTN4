"""Stage 3 - curvature deformation scores.

The deformation score is the mean of the central-axis curvature trace
kappa(t) of each frame:

    D_curv = mean( kappa(t) )        [1/A]

It is reported over two regions:
    global filament            - the whole stitched filament
    local ACTN4 centred window - +/- LOCAL_HALF_WIDTH subunits around the
                                 middle ACTN4-adjacent subunit

Usage:  python -m src.deformation <component>
"""
import json
import sys
import numpy as np

from . import config as C
from .central_axis import axis_spline

def _moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def curvature_trace(knots):
    """Rolling-mean curvature of the axis spline and its subunit-x coordinate."""
    d1 = axis_spline(knots, 0.001, order=1)
    d2 = axis_spline(knots, 0.001, order=2)
    kappa = (np.linalg.norm(np.cross(d1, d2), axis=-1) /
             np.linalg.norm(d1, axis=-1) ** 3)
    smooth = _moving_average(kappa, C.ROLL_WINDOW)[:-1]
    t = np.arange(len(smooth)) / 1000.0 + C.SUBUNIT_X_OFFSET
    return t, smooth


def run(component):
    out = C.paths(component)
    rows = []
    for frame in range(C.N_FRAMES):
        meta = json.load(open(out['metadata'] / f'frame_{frame:03d}.json'))
        knots = np.loadtxt(out['measurements'] / f'frame_{frame:03d}_axis_knots.txt',
                           delimiter=',')
        t, kappa = curvature_trace(knots)
        centre = meta['abp']['middle_rank'] + C.SUBUNIT_X_OFFSET
        window = (t >= centre - C.LOCAL_HALF_WIDTH) & (t <= centre + C.LOCAL_HALF_WIDTH)
        rows.append([frame, meta['occupancy_pct'],
                     float(kappa.mean()), float(kappa[window].mean())])
        print(f'{component} frame {frame:03d}: occupancy {rows[-1][1]:.1f}%, '
              f'D_curv global {rows[-1][2]:.6f}, local {rows[-1][3]:.6f} 1/A', flush=True)

    header = 'frame,occupancy_pct,D_curv_global_invA,D_curv_local_invA' 
    np.savetxt(out['measurements'].parent / 'curvature_deformation_scores.csv',
               np.array(rows), delimiter=',', fmt='%.8f',
               header=header, comments='')


if __name__ == '__main__':
    run(sys.argv[1])
