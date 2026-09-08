# Input data

Everything the pipeline reads lives in this directory. Two of the three inputs
are shipped with the repository; the 3DVA volumes are too large to distribute
here and are linked in locally.

Either location can be overridden with an environment variable
(`ACTN4_DATA_ROOT`, `ACTN4_MODEL`); see `src/config.py`.

```
data/
  actn4_model/          ACTN4 (K255E) model                    0.8 MB   included
  fitted_models/        40 fitted filament models              111 MB   included
    component_1/        frame_000.pdb … frame_019.pdb
    component_2/        frame_000.pdb … frame_019.pdb
  maps/                 40 3DVA volumes                         22 GB   NOT included
    component_1/        J62_component_000_frame_%03d.mrc
    component_2/        J62_component_001_frame_%03d.mrc
```

## 1. Fitted filament models — included

All-atom models of a 12-subunit actin filament refined into each 3DVA frame
volume (ISOLDE / real-space refinement). Chains of 300–450 residues are treated
as actin subunits; bound ADP, Mg²⁺ and phosphate — and, in Component 2,
phalloidin peptides — are present and are bound to the nearest actin subunit so
they travel with it during stitching. Non-actin chains are excluded from all
centroid mathematics.

## 2. ACTN4 (K255E) model — included

Chain C is an actin subunit used only for superposition onto the filament;
chain D is the ACTN4 chain transplanted onto the stitched structure. Other
chains are not used.

## 3. 3DVA volumes — not included

512³ voxels, 1.09 Å per voxel, origin at zero, axis order (x, y, z);
**512 MB per map, 22 GB for all 40**. GitHub rejects any file above 100 MB, so
these are excluded from version control (see `.gitignore`) and should be
obtained from the the authors.

Place them — or symbolic links to them — as:

```
data/maps/component_1/J62_component_000_frame_%03d.mrc
data/maps/component_2/J62_component_001_frame_%03d.mrc
```

The maps are used **only** for the ACTN4 occupancy measurement
(`src/stitch.py`). Every other stage — stitching, helical parameters,
curvature and the deformation scores — runs from the fitted models alone, so
the full analysis except occupancy is reproducible from what is included here.

## Reproducing

The curvature baseline is a published constant (1.14 um^-1 for ADP actin;
Reynolds et al., 2022) defined in `src/config.py`, so no reference dataset is
needed. With the fitted models and the ACTN4 model in place (both included):

```bash
./run_all.sh
```

Occupancy values additionally require the maps in `data/maps/`.
