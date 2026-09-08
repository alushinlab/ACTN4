"""Stage 2 - central-axis refinement.

The curvature deformation score is measured on the filament's central axis, so
this stage fits that axis and nothing else.

The axis is a natural cubic spline through knots that are refined iteratively
against the subunit centroids at a fixed helical radius: each centroid is
projected onto the current axis, displaced towards it by the helical radius,
and every knot is replaced by the mean of its two adjacent displaced points and
its own previous value.

Frames are processed as a Markov chain: frame 0 starts from the standard cold
initialisation and every later frame initialises its refinement from the
previous frame's converged axis, rotated into the current frame.  The update
rule and iteration count are unchanged, so each axis remains a fixed point of
the standard refinement on its own frame's data; the warm start only selects
among near-degenerate solutions using the continuity of the 3DVA trajectory.

Usage:  python -m src.central_axis <component>
"""
import sys
import numpy as np
from scipy.interpolate import CubicSpline

from . import config as C
from .geometry import kabsch

R_HELIX = C.HELICAL_RADIUS


def load_centroids(path):
    """Per-subunit CA centroids from a measurement PDB (ATOM records only),
    sorted along z and centred.  Non-actin chains are written as HETATM by
    stage 1 and are therefore excluded here."""
    groups, block = {}, 0
    for line in open(path):
        if line.startswith('TER'):
            block += 1
        if line.startswith('ATOM') and line[13:15] == 'CA':
            groups.setdefault((line[21], block), []).append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54])))
    centroids = np.array([np.mean(v, axis=0) for v in groups.values()])
    centroids = centroids[np.argsort(centroids[:, 2])]
    return centroids - centroids.mean(axis=0)


def axis_spline(knots, res=0.1, order=0):
    """Natural cubic spline through the axis knots (or its `order`-th derivative)."""
    t = np.arange(len(knots))
    grid = np.arange(-1, len(knots), res)
    cols = [CubicSpline(t, knots[:, i], bc_type='natural')(grid, order) if order
            else CubicSpline(t, knots[:, i], bc_type='natural')(grid)
            for i in range(3)]
    return np.stack(cols, axis=-1)


def _pointers(centroids, axis):
    """Each centroid displaced towards its nearest axis point by the helical radius."""
    out = np.zeros(centroids.shape)
    for j, pt in enumerate(centroids):
        v = axis[np.argmin(np.linalg.norm(axis - pt, axis=1))] - pt
        out[j] = v / np.linalg.norm(v) * R_HELIX
    return out


def cold_initial_axis(centroids):
    avgs = (centroids[:-1] + centroids[1:]) / 2.0
    cp = _pointers(centroids, axis_spline(avgs, 0.1))
    return ((cp + centroids)[:-1] + avgs + (cp + centroids)[1:]) / 3.0


def refine_axis(centroids, knots, iterations=C.N_REFINE_ITERS):
    prev = knots.copy()
    for _ in range(iterations):
        cp = _pointers(centroids, axis_spline(prev, 0.01))
        prev = ((cp + centroids)[:-1] + prev + (cp + centroids)[1:]) / 3.0
    return prev


def run(component):
    out = C.paths(component)
    out['measurements'].mkdir(parents=True, exist_ok=True)
    prev_axis = prev_centroids = None

    for frame in range(C.N_FRAMES):
        path = out['structures_meas'] / f'frame_{frame:03d}_measure.pdb'
        centroids = load_centroids(path)

        if prev_axis is None:
            start, mode = cold_initial_axis(centroids), 'cold start'
        else:                                   # Markov-chain warm start
            R, t, _ = kabsch(prev_centroids, centroids)
            start, mode = prev_axis @ R + t, 'warm start from previous frame'

        knots = refine_axis(centroids, start)
        base = out['measurements'] / f'frame_{frame:03d}'
        np.savetxt(f'{base}_axis_knots.txt', knots, delimiter=',')
        np.savetxt(f'{base}_subunit_centroids.txt', centroids, delimiter=',')
        prev_axis, prev_centroids = knots, centroids
        print(f'{component} frame {frame:03d}: {len(centroids)} subunits, '
              f'{len(knots)} axis knots, {mode}', flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
