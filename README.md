# RUWE Radial KS Clustering

Public-facing version of an open-cluster analysis workflow that combines:

- Monte Carlo + KMeans voting for membership inference
- Gaia color-magnitude diagrams
- RUWE-colored CMD inspection
- Radial Kolmogorov-Smirnov tests between RUWE-low and RUWE-high subsamples
- A multi-cluster summary grid for comparison across clusters

The repository keeps the cleaned notebooks for interactive use, but also adds a command-line script so the workflow can be run without editing notebook cells one by one.

## Included Data

- Raw VizieR query tables for `M45`, `M44`, `M67`, `NGC2516`, and `NGC3532`
- Final member catalogs for `M44`, `M45`, `M67`, `NGC188`, `NGC2516`, and `NGC3532`
- A sample final figure in [data/figures/figure1_evolution_grid.png](/home/ikbarfaiz/ruwe-radial-ks-clustering/data/figures/figure1_evolution_grid.png)
- Archived original notebooks in [archive/original-notebooks](/home/ikbarfaiz/ruwe-radial-ks-clustering/archive/original-notebooks)

`NGC188` currently includes only the final member catalog. The raw VizieR query table for that cluster was not present in the source files I found locally, so Step 1 is not fully reproducible for `NGC188` until that raw input is restored.

## Repository Layout

`notebooks/`
Interactive notebooks cleaned up for public sharing.

`scripts/run_analysis.py`
Single CLI entry point for the full pipeline.

`data/raw/`
Raw VizieR-style query tables.

`data/members/`
Final member catalogs used by the later analysis steps.

`configs/multicluster_grid.csv`
Config file used to build the combined summary figure.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

Run membership inference for `M67`:

```bash
python3 scripts/run_analysis.py membership \
  --input-path data/raw/m67.tsv \
  --output-members data/members/M67_members_recomputed.csv \
  --probability-output data/members/M67_probabilities.csv \
  --cluster-name "M67" \
  --target-pmra -10.9 \
  --target-pmdec -2.9 \
  --target-parallax 1.13 \
  --figure-dir data/figures/generated/m67
```

Plot a standard CMD from a member catalog:

```bash
python3 scripts/run_analysis.py cmd \
  --members-csv data/members/M67_Members_Final.csv \
  --cluster-name "M67" \
  --output-figure data/figures/generated/M67_cmd.png
```

Plot a RUWE-colored CMD:

```bash
python3 scripts/run_analysis.py ruwe-cmd \
  --members-csv data/members/M67_Members_Final.csv \
  --cluster-name "M67" \
  --output-figure data/figures/generated/M67_ruwe_cmd.png
```

Run the radial KS comparison:

```bash
python3 scripts/run_analysis.py radial-ks \
  --members-csv data/members/M67_Members_Final.csv \
  --cluster-name "M67" \
  --xlim-arcmin 60 \
  --summary-output data/figures/generated/M67_ks_summary.csv \
  --output-figure data/figures/generated/M67_radial_ks.png
```

Generate the multi-cluster summary grid:

```bash
python3 scripts/run_analysis.py grid \
  --config-csv configs/multicluster_grid.csv \
  --output-figure data/figures/generated/evolution_grid_generated.png
```

## Notes

- The public notebooks are based on the cleaned versions already present in the original project under `Jurnal/Codingan/GITHUB`.
- The archived `step1.ipynb` to `step5.ipynb` files are preserved for provenance, but the cleaned notebooks and CLI are the recommended public entry points.
- `phot_g_mean_mag` is plotted as apparent magnitude unless you pass a distance modulus in the CLI.

