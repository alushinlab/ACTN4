# Actin lattice deformation and ACTN4 density occupancy along 3DVA trajectories

Code and derived results for measuring the central-axis curvature of actin
filaments along two cryoSPARC 3D variability analysis (3DVA) components, and
relating the resulting lattice curvature deformation to the cryo-EM density
occupancy of a docked ACTN4 (K255E) model.

Each 3DVA component is represented by 20 frames. A frame consists of a
12-subunit actin filament model fitted into the corresponding 3DVA volume,
and the volume itself.

## What the pipeline does

| Stage | Module | Output |
|-------|--------|--------|
| 1 | `src/stitch.py` | block-aligned stitching to a 30-subunit filament, ACTN4 docking, map occupancy |
| 2 | `src/central_axis.py` | central-axis refinement (Markov chain across frames) |
| 3 | `src/deformation.py` | curvature deformation vs a micrograph-tracing baseline |
| 4 | `src/plot_deformation.py` | ACTN4 occupancy vs curvature deformation |

## Layout

```
data/                    inputs (see data/README.md)
src/                     analysis modules (see table above)
run_all.sh               end-to-end driver for both components
results/<component>/
    structures/
        stitched_map_frame/    30-subunit filament + ACTN4, in map coordinates
        measurement_input/     same filament rotated onto z for measurement
    measurements/              per-frame axis knots and subunit centroids
    curvature_deformation_scores.csv
    metadata/                  per-frame provenance and QC values
    figures/
        deformation/           occupancy vs curvature deformation score
    logs/
results/curvature_baseline.json    cached kappa_0
```

## Running

```bash
pip install -r requirements.txt
./run_all.sh          # inputs are read from data/ - see data/README.md
```

Individual stages can be run on their own, e.g.
`python3 -m src.stitch component_1`.

## Notes and caveats

* Component 1 and Component 2 correspond to internal dataset names
  `J62_component_000` and `J62_component_001`.
* Frames are numbered 0–19 in file names and 1–20 in figures.