"""Central configuration for the ACTN4 / actin lattice-deformation analysis.

External input locations can be overridden with environment variables so the
pipeline runs on any machine (see data/README.md).
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# Inputs:
# The fitted models and the ACTN4 model are shipped with the repository; the
# 3DVA volumes (512 MB each) are too large to distribute here.
# --------------------------------------------------------------------------
DATA = Path(os.environ.get('ACTN4_DATA_ROOT', str(REPO / 'data')))

ACTN4_MODEL = Path(os.environ.get('ACTN4_MODEL', str(
    DATA / 'actn4_model' /
    'k255e_hardcodedbonds_phalloidinfrom7PLV_real_space_refined_021-coot-0'
    '_addchainnamesinheader.pdb')))

COMPONENTS = {
    'component_1': dict(
        label='Component 1',
        internal='J62_component_000',
        frames=DATA / 'fitted_models' / 'component_1',
        frame_pattern='frame_%03d.pdb',
        maps=DATA / 'maps' / 'component_1',
        map_pattern='J62_component_000_frame_%03d.mrc'),
    'component_2': dict(
        label='Component 2',
        internal='J62_component_001',
        frames=DATA / 'fitted_models' / 'component_2',
        frame_pattern='frame_%03d.pdb',
        maps=DATA / 'maps' / 'component_2',
        map_pattern='J62_component_001_frame_%03d.mrc'),
}
N_FRAMES = 20

# --------------------------------------------------------------------------
# Stitching (fixed register - identical operation for every frame)
# --------------------------------------------------------------------------
# BLOCK-ALIGNED STITCHING (sliding-block variant).
#
# Each attached copy is positioned by superposing a BLOCK of STITCH_BLOCK
# consecutive subunits of the copy onto the terminal block of the reference
# filament (the terminal subunits of a copy's barbed end are aligned with the 
# terminal subunits of the original's pointed end, and vice versa; overlapping 
# subunits of the copies are then discarded).
#
# STITCH_BLOCK = 3 is the terminal
# three-subunit overlap and the shift then
# follows from the geometry, because a copy overlaps the reference in
# (n_fitted - shift) subunits:
#
# STITCH_SHIFT = n_fitted - STITCH_BLOCK = 12 - 3 = 9.
#
# Fitting the placement to a block rather than to a single subunit makes each
# copy depend on the whole overlap region rather than on one subunit's local
# conformation.
STITCH_BLOCK = 3      # subunits used to fit each copy's placement
STITCH_SHIFT = 9      # lattice shift; overlap = n_fitted - STITCH_SHIFT

ACTN4_REF_CHAIN = 'C'                 # actin chain of the ACTN4 model, used for superposition
ACTN4_KEEP_CHAIN = 'D'                # ACTN4 chain carried into the stitched filament
ACTN4_TARGET_FITTED_SUBUNIT = 7       # index within the 12 fitted subunits (0-based)
ACTN4_CHAIN_ID = 'z'                  # chain id of ACTN4 in all written PDBs
DENSITY_THRESHOLD = 0.2               # map value defining "inside density"

# --------------------------------------------------------------------------
# Central axis and curvature
# --------------------------------------------------------------------------
HELICAL_RADIUS = 15.76719916835962     # subunit centroid distance from the axis (A) from Reynolds et. al 2022
N_REFINE_ITERS = 500                   # axis-refinement iterations per frame
ROLL_WINDOW = 2000                     # rolling mean applied to the curvature trace

SUBUNIT_X_OFFSET = -4.0                # subunit rank r sits at filament coordinate r - 4

# ABP-centred local window: +- this many subunits around the middle
# ACTN4-adjacent subunit (11 subunits wide).
LOCAL_HALF_WIDTH = 5.5

# --------------------------------------------------------------------------
# Output layout
# --------------------------------------------------------------------------
RESULTS = REPO / 'results'

def component_dir(component):
    return RESULTS / component

def paths(component):
    d = component_dir(component)
    return dict(
        structures_map=d / 'structures' / 'stitched_map_frame',
        structures_meas=d / 'structures' / 'measurement_input',
        measurements=d / 'measurements',
        metadata=d / 'metadata',
        fig_deformation=d / 'figures' / 'deformation',
        logs=d / 'logs',
    )
