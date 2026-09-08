"""Stage 1 - fixed-register stitching, ACTN4 docking and density occupancy.

For every 3DVA frame:

  1. The 12 fitted actin subunits are ordered along the filament.
  2. ONE rigid transform T is derived per frame by superposing a BLOCK of
     STITCH_BLOCK consecutive subunits of a copy onto the terminal block of the
     reference filament, so that copy subunit j maps onto position
     j - STITCH_SHIFT.  Both constants follow from the terminal-overlap
     construction (see src/config.py), so the stitching operation is identical
     for every frame of every component.
  3. One copy of the whole filament is attached at each end (transforms T and
     T^-1); subunits that would duplicate the fitted model are discarded.  With
     block 3 and shift 9 the result is a 30-subunit filament whose fitted
     (in-map) subunits occupy ranks 9..20 and whose coordinates are untouched,
     so the structure stays aligned with the map.
  4. The ACTN4 model is docked by superposing its actin chain C onto
     fitted subunit 7; ACTN4 chain D is kept and added as chain 'z'.
  5. The fraction of ACTN4 CA atoms inside the corresponding frame map is
     measured by trilinear interpolation at a fixed threshold.
  6. Two structures are written: the map-frame all-atom filament, and a
     measurement input in which the filament long axis is rotated onto z
     (rigid; helical parameters are invariant) and every non-actin chain is
     written as HETATM so it is excluded from the centroid mathematics.

Usage:  python -m src.stitch <component>
"""
import json
import sys
import numpy as np

from . import config as C
from .geometry import (CHAIN_ALPHABET, chain_ca, chain_ca_by_resseq, chain_com,
                       classify_chains, invert_transform, kabsch, load_structure,
                       order_along_filament, rotation_to_z, transform_chain,
                       transformed_chain, write_pdb)

def _subunits_from_frame(path):
    """Ordered actin chains plus the accessory chains bound to each of them."""
    model = load_structure(path)[0]
    actins, others = classify_chains(model)
    actins = [actins[i] for i in order_along_filament(actins)]
    coms = np.array([chain_com(c) for c in actins])
    attached = {i: [] for i in range(len(actins))}
    for chain in others:
        attached[int(np.argmin(np.linalg.norm(coms - chain_com(chain), axis=1)))].append(chain)
    return actins, attached


def stitch_frame(frame_path):
    """Build the extended filament (n_fitted + 2*shift subunits). Returns (subunits, info).

    """
    head_actins, head_extras = _subunits_from_frame(frame_path)
    actins, extras = _subunits_from_frame(frame_path)          # fitted model, untouched
    tail_actins, tail_extras = _subunits_from_frame(frame_path)

    n = len(actins)
    shift, blk = C.STITCH_SHIFT, C.STITCH_BLOCK

    mobile = np.vstack([chain_ca(actins[j]) for j in range(shift, shift + blk)])
    target = np.vstack([chain_ca(actins[j]) for j in range(blk)])
    R, t, rmsd = kabsch(mobile, target)
    R_inv, t_inv = invert_transform(R, t)

    for chains, (Rk, tk) in (
            (head_actins + [c for v in head_extras.values() for c in v], (R, t)),
            (tail_actins + [c for v in tail_extras.values() for c in v], (R_inv, t_inv))):
        for chain in chains:
            transform_chain(chain, Rk, tk)

    def block(src_actins, src_extras, ranks, fitted):
        return [dict(actin=src_actins[i], extras=src_extras[i],
                     source_rank=i, is_fitted=fitted) for i in ranks]

    subunits = (block(head_actins, head_extras, range(shift), False) +
                block(actins, extras, range(n), True) +
                block(tail_actins, tail_extras, range(n - shift, n), False))

    info = dict(n_fitted=n, n_subunits=len(subunits),
                stitch_block_rmsd_A=round(rmsd, 3),
                stitch_block=blk, stitch_shift=shift, stitch_overlap=n - shift,
                fitted_ranks=[i for i, s in enumerate(subunits) if s['is_fitted']])
    return subunits, info


def dock_actn4(subunits, target_rank):
    """Superpose the ACTN4 model onto `target_rank` and return (chain, info)."""
    actn4 = load_structure(C.ACTN4_MODEL)[0]
    ref = chain_ca_by_resseq(actn4[C.ACTN4_REF_CHAIN])
    tgt = chain_ca_by_resseq(subunits[target_rank]['actin'])
    common = sorted(set(ref) & set(tgt))
    R, t, rmsd = kabsch(np.array([ref[r] for r in common]),
                        np.array([tgt[r] for r in common]))
    chain = transformed_chain(actn4[C.ACTN4_KEEP_CHAIN], R, t)
    return chain, dict(actn4_rmsd_A=round(rmsd, 3), actn4_ca_pairs=len(common))


def occupancy(actn4_chain, map_path, threshold):
    """Fraction of ACTN4 CA atoms whose trilinearly interpolated map value >= threshold."""
    import mrcfile
    from scipy.ndimage import map_coordinates
    cas = chain_ca(actn4_chain)
    with mrcfile.mmap(str(map_path), permissive=True) as m:
        h = m.header
        if (int(h.mapc), int(h.mapr), int(h.maps)) != (1, 2, 3):
            raise RuntimeError(f'unexpected axis order in {map_path}')
        apix = float(m.voxel_size.x)
        origin = np.array([float(h.origin.x), float(h.origin.y), float(h.origin.z)])
        nstart = np.array([int(h.nxstart), int(h.nystart), int(h.nzstart)])
        vox = (cas - origin) / apix - nstart
        vals = map_coordinates(m.data, np.vstack([vox[:, 2], vox[:, 1], vox[:, 0]]),
                               order=1, mode='constant', cval=0.0)
    return dict(occupancy_pct=round(100.0 * float((vals >= threshold).mean()), 2),
                density_threshold=threshold, n_actn4_ca=int(len(cas)),
                median_map_value=round(float(np.median(vals)), 4))


def abp_neighbours(subunits, actn4_chain):
    """Closest actin subunit, its same-strand (+/-2) partner and the one between."""
    coms = np.array([chain_com(s['actin']) for s in subunits])
    d = np.linalg.norm(coms - chain_com(actn4_chain), axis=1)
    closest = int(np.argmin(d))
    cands = [r for r in (closest - 2, closest + 2) if 0 <= r < len(subunits)]
    partner = int(min(cands, key=lambda r: d[r]))
    middle = (closest + partner) // 2
    ranks = sorted([closest, partner, middle])
    return dict(closest_rank=closest, partner_rank=partner, middle_rank=middle,
                ranks=ranks, x_positions=[r - 4 for r in ranks],
                distances_A={str(r): round(float(d[r]), 1) for r in ranks})


def orient_for_measurement(subunits, actn4_chain):
    """Rigidly rotate the filament IN PLACE so its long axis lies on z.

    Helical parameters are invariant under this rotation, but the measurement
    needs the filament on z because subunits are ordered by z.  The sign of
    the principal axis is anchored to the map frame (the subunit at low map-z
    stays at low z), which makes the orientation deterministic.

    Call this only after the map-frame structure has been written: the input
    chains are modified.
    """
    coms = np.array([chain_com(s['actin']) for s in subunits])
    centre = coms.mean(axis=0)
    _, _, Vt = np.linalg.svd(coms - centre)
    axis = Vt[0]
    span = coms[np.argmax(coms[:, 2])] - coms[np.argmin(coms[:, 2])]
    if np.dot(axis, span) < 0:
        axis = -axis
    M = rotation_to_z(axis)
    R_row, t_row = M.T, -centre @ M.T                 # v' = M (v - centre)

    for s in subunits:
        transform_chain(s['actin'], R_row, t_row)
        for extra in s['extras']:
            transform_chain(extra, R_row, t_row)
    transform_chain(actn4_chain, R_row, t_row)

    z = np.sort([chain_com(s['actin'])[2] for s in subunits])
    return float(np.diff(z).min())


def _blocks(subunits, actn4_chain, extras_as_het):
    blocks = []
    for i, s in enumerate(subunits):
        char = CHAIN_ALPHABET[i % len(CHAIN_ALPHABET)]
        blocks.append((char, s['actin'], False))
        for extra in s.get('extras', []):
            blocks.append((char, extra, extras_as_het))
    blocks.append((C.ACTN4_CHAIN_ID, actn4_chain, extras_as_het))
    return blocks


def run(component):
    cfg, out = C.COMPONENTS[component], C.paths(component)
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)

    for frame in range(C.N_FRAMES):
        frame_path = cfg['frames'] / (cfg['frame_pattern'] % frame)
        map_path = cfg['maps'] / (cfg['map_pattern'] % frame)

        subunits, info = stitch_frame(frame_path)
        target_rank = info['fitted_ranks'][C.ACTN4_TARGET_FITTED_SUBUNIT]
        actn4, dock_info = dock_actn4(subunits, target_rank)
        occ = occupancy(actn4, map_path, C.DENSITY_THRESHOLD)
        abp = abp_neighbours(subunits, actn4)

        # map-frame structure (fitted subunits keep their original coordinates)
        write_pdb(_blocks(subunits, actn4, False),
                  out['structures_map'] / f'frame_{frame:03d}_stitched_ACTN4.pdb')

        # verification against a fresh read of the source model, before the
        # measurement rotation modifies the chains
        reference_ca, _ = _subunits_from_frame(frame_path)
        dev = max(float(np.abs(chain_ca(subunits[r]['actin']) - chain_ca(reference_ca[k])).max())
                  for k, r in enumerate(info['fitted_ranks']))

        # measurement input: filament on z, everything non-actin as HETATM
        min_gap = orient_for_measurement(subunits, actn4)
        write_pdb(_blocks(subunits, actn4, True),
                  out['structures_meas'] / f'frame_{frame:03d}_measure.pdb')

        meta = dict(component=component, label=cfg['label'], frame=frame,
                    actn4_target_rank=target_rank,
                    min_subunit_z_gap_A=round(min_gap, 2),
                    fitted_coord_max_deviation_A=round(dev, 6),
                    **info, **dock_info, **occ, abp=abp)
        with open(out['metadata'] / f'frame_{frame:03d}.json', 'w') as fh:
            json.dump(meta, fh, indent=1)
        print(f'{component} frame {frame:03d}: {info["n_subunits"]} subunits, '
              f'block-fit RMSD {info["stitch_block_rmsd_A"]} A, ACTN4 RMSD '
              f'{dock_info["actn4_rmsd_A"]} A, occupancy {occ["occupancy_pct"]}%, '
              f'fitted-coord deviation {dev:.4f} A', flush=True)


if __name__ == '__main__':
    run(sys.argv[1])
