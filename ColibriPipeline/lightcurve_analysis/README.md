# lightcurve_analysis

Night-long light-curve analysis and visualization for the Colibri telescope
array.

Each night, `colibri_main_py3` + `wcsmatching` write per-minute `.npy` products
into `ColibriArchive/<YYYY-MM-DD>/`:

- `<minute>_stars.npy` — `(n_frames≈2399, n_stars≈1287, 4)` float64,
  last axis `[x, y, flux, unix_time]` (~40 Hz over 60 s). Flux can be negative.
- `<minute>_<thresh>sig_pos.npy` — `(n_stars, 5)` `[x, y, half_light_radius,
  RA_deg, Dec_deg]` (3 cols before WCS), sharing star ordering with axis-1 of
  the stars file, so RA/Dec joins by index.

This package reads a whole night, assigns **stable global star IDs** across
minutes (a star's local index is consistent *within* a minute but not *across*
minutes), stitches the per-minute 40 Hz curves into night-long series, and
produces raw **and** differential light curves — as a library, a
figure-producing CLI, and an optional interactive widget notebook. It runs on a
telescope (`D:/`), on the sim tree, or on a dedicated analysis box pointed at an
arbitrary downloaded archive folder.

## Pointing at data

Path resolution precedence (first match wins):

1. Explicit `--archive-root PATH` / `--night-dir PATH` (CLI) or the
   `archive_root=` / `sim_root=` arguments (library).
2. `COLIBRI_ARCHIVE_ROOT` — the **analysis-box default**: a folder that
   directly contains `<YYYY-MM-DD>/` night dirs.
3. Sim layout: `COLIBRI_ENV=sim` → `COLIBRI_SIM_ROOT/<color>/ColibriArchive`
   (color from `TELESCOPE_COLORS = {REDBIRD:Red, GREENBIRD:Green, BLUEBIRD:Blue}`).
4. Telescope default `D:/ColibriArchive`.

```bash
# analysis box: point straight at a downloaded archive folder
export COLIBRI_ARCHIVE_ROOT=/data/colibri/green_archive   # contains 2025-08-30/

# sim tree
export COLIBRI_ENV=sim
export COLIBRI_SIM_ROOT=/home/agirmen/research_data/ColibriPipelineSimulatedDirs
```

## Quickstart (library)

```python
from lightcurve_analysis import build_night_catalog
from lightcurve_analysis import plotting as P

cat = build_night_catalog('GREENBIRD', '2025-08-30')   # builds + caches
print(cat.stars.head())                                # per-star metadata

# filter and extract
bright = cat.filter_stars(snr_min=5, gmag=(8, 14))
sid = int(bright.sort_values('snr', ascending=False).iloc[0]['star_id'])

lc = cat.get_raw_lightcurve(sid)            # raw, NaN at gaps
fig, ax = P.plot_single(lc)                 # returns (fig, ax)
fig = P.plot_raw_vs_diff(cat, sid)          # stacked raw + differential
```

## CLI

```bash
python -m lightcurve_analysis.cli \
  --telescope GREENBIRD --date 2025-08-30 \
  [--archive-root PATH | --night-dir PATH | --sim-root PATH] \
  --out OUT [--kind raw|diff|both] [--bin-seconds 1.0] \
  [--snr-min 3] [--gmag 8 14] [--star-ids ... | --max-stars 50] \
  [--diff-method ensemble|pca] [--comp-rad-pix 100] \
  [--attach-gaia] [--gaia-radius 0.5] [--cache-dir PATH] [--rebuild] [--grid]
```

Output tree:

```
OUT/<telescope>/<obsdate>/
  catalog_summary.csv          # cat.stars
  lightcurves/raw/star_<id>.png
  lightcurves/diff/star_<id>.png
  grids/raw_grid_<page>.png    # with --grid
  manifest.json                # params, counts, cache path, timestamp
```

Example:

```bash
COLIBRI_ENV=sim COLIBRI_SIM_ROOT=/path/to/sim \
python -m lightcurve_analysis.cli \
  --telescope GREENBIRD --date 2025-08-30 --max-stars 50 \
  --kind both --bin-seconds 1.0 --out /tmp/lc_out
```

## Interactive notebook

`notebooks/lightcurve_explorer.ipynb` builds a catalog and offers an
interactive explorer plus static-plot examples (useful even without
ipywidgets):

```python
from lightcurve_analysis.widgets import explore
explore(cat)          # widget UI if ipywidgets present; else a static plotter
```

The widget layer is optional. Install it with:

```bash
pip install -r requirements-notebook.txt   # ipywidgets, jupyterlab, notebook
```

When ipywidgets is absent, `explore(cat)` prints a notice and returns
`explore_static` bound to the catalog, so the same parameters drive plain
function calls:

```python
plot = explore(cat)
fig = plot(star_id, kind='both', bin_seconds=1.0)
```

## Module map

| module           | role |
|------------------|------|
| `paths.py`       | archive-root / night-dir resolution + config |
| `io.py`          | minute discovery + lazy/mmap loading |
| `matching.py`    | cross-minute star matching → stable global IDs |
| `stitch.py`      | night time axis, gap mask, binning, `LightCurve` |
| `catalog.py`     | `NightCatalog`, `build_night_catalog`, cache save/load, extraction API |
| `differential.py`| ensemble (default) + PCA differential light curves |
| `gaia.py`        | Gaia mag attach + per-night cache |
| `plotting.py`    | headless-safe figures (raw / diff / grid / zoom) |
| `cli.py`         | argparse batch script → PNGs + summary + manifest |
| `widgets.py`     | optional ipywidgets explorer with static fallback |

## Cache

`build_night_catalog` writes a per-night cache under
`<night_dir>/lightcurve_cache/` (or `--cache-dir`):

- `<telescope>_<obsdate>_lightcurves.npz` — stitched payload, **sparse-per-star
  ragged** (`flux_concat` float32, `time_concat`, `star_offsets`) so long
  nights avoid GB-scale NaN-padded dense matrices; loaded mmap-backed.
- `<telescope>_<obsdate>_stars.parquet` — the per-star metadata DataFrame.
- `<telescope>_<obsdate>_minutes.json` — `MinuteRef` list + per-minute
  local→global index maps.
- `<telescope>_<obsdate>_gaia.parquet` — Gaia match cache (`attach_gaia`).

A subsequent `build_night_catalog` (or `load_cache`) reuses the cache unless
`rebuild=True` / `--rebuild`.
