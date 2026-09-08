"""Shared geometry utilities: structure parsing, superposition, PDB writing."""
import copy
import numpy as np
from Bio.PDB import PDBParser

# Single-character chain ids for the actin subunits; 'z' is reserved for ACTN4.
CHAIN_ALPHABET = ('ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                  'abcdefghijklmnopqrstuvwxy')
ACTIN_MIN_RES, ACTIN_MAX_RES = 300, 450


# ---------------------------------------------------------------- parsing --
def load_structure(path):
    return PDBParser(QUIET=True).get_structure('model', str(path))


def chain_ca(chain):
    """Protein CA coordinates (element check excludes ions named CA)."""
    return np.array([a.get_coord() for a in chain.get_atoms()
                     if a.get_name() == 'CA'
                     and (a.element or '').strip().upper() == 'C'])


def chain_ca_by_resseq(chain):
    out = {}
    for a in chain.get_atoms():
        if a.get_name() == 'CA' and (a.element or '').strip().upper() == 'C':
            out[a.get_parent().id[1]] = a.get_coord()
    return out


def chain_com(chain):
    return np.array([a.get_coord() for a in chain.get_atoms()]).mean(axis=0)


def classify_chains(model):
    """Split chains into actin subunits and accessory chains (e.g. phalloidin)."""
    actins, others = [], []
    for chain in model:
        n_res = len(list(chain.get_residues()))
        (actins if ACTIN_MIN_RES < n_res < ACTIN_MAX_RES else others).append(chain)
    return actins, others


def order_along_filament(chains):
    """Indices ordering subunits by a 3D nearest-neighbour walk from a tip."""
    coms = np.array([chain_com(c) for c in chains])
    d = np.linalg.norm(coms[:, None, :] - coms[None, :, :], axis=-1)
    start = int(np.argmin((d < 65.0).sum(axis=1)))
    order, remaining, cur = [start], set(range(len(chains))) - {start}, start
    while remaining:
        nxt = min(remaining, key=lambda j: d[cur, j])
        order.append(nxt)
        remaining.discard(nxt)
        cur = nxt
    return order


# ---------------------------------------------------------- superposition --
def kabsch(mobile, target):
    """Rigid transform with  target ~= mobile @ R + t  (row-vector convention).

    Returns (R, t, rmsd).
    """
    cm, ct = mobile.mean(axis=0), target.mean(axis=0)
    U, _, Vt = np.linalg.svd((mobile - cm).T @ (target - ct))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    t = ct - cm @ R
    rmsd = float(np.sqrt(np.mean(np.sum((target - (mobile @ R + t)) ** 2, axis=1))))
    return R, t, rmsd


def invert_transform(R, t):
    """Inverse of  y = x @ R + t."""
    return R.T, -(t @ R.T)


def transform_chain(chain, R, t):
    """Apply  x -> x @ R + t  to every atom of a chain, in place.
    """
    for a in chain.get_atoms():
        a.set_coord(a.get_coord() @ R + t)
    return chain


def transformed_chain(chain, R, t):
    """Transform a deep copy of a chain (kept for callers that need the original)."""
    return transform_chain(copy.deepcopy(chain), R, t)


def rotation_to_z(axis):
    """Rotation matrix (column convention: v' = M v) sending `axis` onto +z."""
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(axis, z), float(np.dot(axis, z))
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    S = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + S + S @ S / (1.0 + c)


# ------------------------------------------------------------ PDB writing --
def _safe_coord(val):
    for fmt in ('%8.3f', '%8.2f', '%8.1f', '%8.0f'):
        s = fmt % val
        if len(s) <= 8:
            return s
    return '%8.0f' % val


def _atom_name_field(name, element):
    name = name.strip()
    if len(name) >= 4:
        return name[:4]
    return f'{name:<4s}' if len(element) == 2 else f' {name:<3s}'


def write_pdb(blocks, path):
    """Write chains as TER-separated blocks with single-character chain ids.

    blocks : list of (chain_char, Bio.PDB chain, force_hetatm)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    serial = 1
    with open(path, 'w') as fh:
        for chain_char, chain, force_het in blocks:
            r_seq, last = 0, ('ALA', 1)
            for res in chain.get_residues():
                r_seq += 1
                seq = (r_seq - 1) % 9999 + 1
                record = 'HETATM' if (force_het or res.id[0] != ' ') else 'ATOM  '
                for atom in res.get_atoms():
                    elem = (atom.element or '').strip().upper()
                    x, y, z = atom.get_coord()
                    fh.write(f'{record}{(serial - 1) % 99999 + 1:5d} '
                             f'{_atom_name_field(atom.get_name(), elem)} '
                             f'{res.get_resname():>3s} {chain_char}{seq:4d}    '
                             f'{_safe_coord(x)}{_safe_coord(y)}{_safe_coord(z)}'
                             f'  1.00  0.00          {elem:>2s}\n')
                    serial += 1
                last = (res.get_resname(), seq)
            fh.write(f'TER   {(serial - 1) % 99999 + 1:5d}      '
                     f'{last[0]:>3s} {chain_char}{last[1]:4d}\n')
            serial += 1
        fh.write('END\n')
