#!/usr/bin/env python3
"""
Filename : cli.py
Author   : ColibriPipeline / lightcurve_analysis (Wave 3)

Batch figure-producing command-line entry point for the night light-curve
analysis package. Builds (or loads cached) a NightCatalog for one
telescope/night, optionally attaches Gaia magnitudes, filters and selects
stars, and renders raw and/or differential light-curve PNGs (plus optional
binned variants and grid pages). A ``catalog_summary.csv`` and a
``manifest.json`` describing the run are also written.

Output tree::

    <out>/<telescope>/<obsdate>/
      catalog_summary.csv
      lightcurves/raw/star_<id>.png
      lightcurves/diff/star_<id>.png
      grids/raw_grid_<page>.png
      manifest.json

Example
-------
    COLIBRI_ENV=sim COLIBRI_SIM_ROOT=/path/to/sim \\
    python -m lightcurve_analysis.cli \\
      --telescope GREENBIRD --date 2025-08-30 \\
      --max-stars 50 --kind both --bin-seconds 1.0 --out /tmp/lc_out
"""

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

# --- Guarded intra-package imports -----------------------------------------
# When run as ``python -m lightcurve_analysis.cli`` (the documented form),
# __package__ is set and relative imports work. When run as a loose script
# from inside ColibriPipeline/, fall back to flat imports after ensuring the
# package directory's parent is importable.
if __package__:
    from . import plotting as P
    from .catalog import build_night_catalog, load_cache
    from . import paths as _paths
else:  # flat execution
    _here = Path(__file__).resolve().parent
    for _p in (str(_here), str(_here.parent)):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import plotting as P  # type: ignore
    from catalog import build_night_catalog, load_cache  # type: ignore
    import paths as _paths  # type: ignore

# Gaia is optional at runtime (only used with --attach-gaia); import lazily.

try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except ImportError:  # pragma: no cover - tqdm is a soft dependency
    _HAVE_TQDM = False

    def tqdm(iterable, **_kwargs):  # type: ignore
        return iterable


def build_parser():
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog='lightcurve_analysis.cli',
        description='Render night-long raw/differential light-curve figures '
                    'for one Colibri telescope/night.',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- where the data lives ---
    parser.add_argument('--telescope', default='GREENBIRD',
                        help='Telescope label (REDBIRD/GREENBIRD/BLUEBIRD or '
                             'a color). Default: GREENBIRD.')
    parser.add_argument('--date', required=True,
                        help='Observation date, YYYYMMDD or YYYY-MM-DD.')
    parser.add_argument('--archive-root', default=None,
                        help='Explicit archive root containing <YYYY-MM-DD>/ '
                             'night dirs. Overrides env/sim resolution.')
    parser.add_argument('--night-dir', default=None,
                        help='Explicit night directory; archive-root and date '
                             'are derived from it.')
    parser.add_argument('--sim-root', default=None,
                        help='Sim root (used with COLIBRI_ENV=sim).')

    # --- where the output goes ---
    parser.add_argument('--out', required=True,
                        help='Output base directory.')

    # --- what to render ---
    parser.add_argument('--kind', choices=['raw', 'diff', 'both'],
                        default='both',
                        help='Which curves to render per star. Default: both.')
    parser.add_argument('--bin-seconds', type=float, default=None,
                        help='If set, also render binned curves at this '
                             'bin width (seconds).')

    # --- star selection / filtering ---
    parser.add_argument('--snr-min', type=float, default=None,
                        help='Keep only stars with SNR >= this value.')
    parser.add_argument('--gmag', type=float, nargs=2, default=None,
                        metavar=('LO', 'HI'),
                        help='Keep only stars with LO <= gmag <= HI.')
    parser.add_argument('--star-ids', type=int, nargs='+', default=None,
                        help='Explicit list of global star ids to render '
                             '(overrides --max-stars).')
    parser.add_argument('--max-stars', type=int, default=50,
                        help='Max stars to render (top by SNR) when '
                             '--star-ids is not given. Default: 50.')

    # --- differential params ---
    parser.add_argument('--diff-method', choices=['ensemble', 'pca'],
                        default='ensemble',
                        help='Differential photometry method. Default: '
                             'ensemble.')
    parser.add_argument('--comp-rad-pix', type=float, default=100.0,
                        help='Comparison-star selection radius (pixels). '
                             'Default: 100.')

    # --- Gaia ---
    parser.add_argument('--attach-gaia', action='store_true',
                        help='Query Gaia and attach gmag/bp_rp (cache-gated).')
    parser.add_argument('--gaia-radius', type=float, default=0.5,
                        help='Gaia search radius (deg). Default: 0.5.')

    # --- cache control ---
    parser.add_argument('--cache-dir', default=None,
                        help='Catalog cache directory. Default: '
                             '<night_dir>/lightcurve_cache.')
    parser.add_argument('--rebuild', action='store_true',
                        help='Force a catalog rebuild even if a cache exists.')

    # --- extras ---
    parser.add_argument('--grid', action='store_true',
                        help='Also render grid pages of the selected stars.')

    return parser


def _derive_from_night_dir(night_dir):
    """Return (archive_root, obsdate) derived from an explicit night dir.

    The night directory name is taken as the obsdate and its parent as the
    archive root.
    """
    night_dir = Path(night_dir)
    return str(night_dir.parent), night_dir.name


def _select_star_ids(cat, args):
    """Resolve the list of star ids to render after filtering."""
    if args.star_ids:
        present = set(cat.stars['star_id'].tolist())
        ids = [int(s) for s in args.star_ids if int(s) in present]
        missing = [int(s) for s in args.star_ids if int(s) not in present]
        if missing:
            print(f"  [warn] requested star ids not in catalog: {missing}")
        return ids

    gmag = tuple(args.gmag) if args.gmag is not None else None
    filtered = cat.filter_stars(gmag=gmag, snr_min=args.snr_min)
    # Top by SNR, descending.
    filtered = filtered.sort_values('snr', ascending=False)
    if args.max_stars is not None and args.max_stars > 0:
        filtered = filtered.head(int(args.max_stars))
    return [int(s) for s in filtered['star_id'].tolist()]


def _render_star(cat, sid, out_dir, args):
    """Render the requested PNG(s) for one star. Returns list of written paths."""
    written = []
    kind = args.kind
    want_raw = kind in ('raw', 'both')
    want_diff = kind in ('diff', 'both')

    if want_raw:
        raw_dir = out_dir / 'lightcurves' / 'raw'
        lc = cat.get_raw_lightcurve(sid, with_gaps=True)
        fig, _ = P.plot_single(lc, title=f'Star #{sid} (raw)')
        path = P.save_figure(fig, raw_dir / f'star_{sid}.png')
        _close(fig)
        written.append(path)

        if args.bin_seconds:
            blc = cat.get_binned_lightcurve(sid, args.bin_seconds)
            fig, _ = P.plot_single(
                blc, title=f'Star #{sid} (binned {args.bin_seconds:g}s)')
            path = P.save_figure(
                fig, raw_dir / f'star_{sid}_bin{args.bin_seconds:g}s.png')
            _close(fig)
            written.append(path)

    if want_diff:
        diff_dir = out_dir / 'lightcurves' / 'diff'
        fig = P.plot_raw_vs_diff(
            cat, sid, bin_seconds=args.bin_seconds,
            diff_method=args.diff_method, comp_rad_pix=args.comp_rad_pix)
        path = P.save_figure(fig, diff_dir / f'star_{sid}.png')
        _close(fig)
        written.append(path)

    return written


def _close(fig):
    """Close a figure to free memory (best-effort)."""
    try:
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:  # noqa: BLE001
        pass


def _render_grids(cat, star_ids, out_dir, args, per_page=16):
    """Render grid pages of the selected stars. Returns list of written paths."""
    grids_dir = out_dir / 'grids'
    written = []
    grid_kind = 'diff' if args.kind == 'diff' else 'raw'
    for page, start in enumerate(range(0, len(star_ids), per_page)):
        chunk = star_ids[start:start + per_page]
        if not chunk:
            continue
        out_path = grids_dir / f'{grid_kind}_grid_{page}.png'
        fig = P.plot_multi_grid(
            cat, chunk, kind=grid_kind, ncols=4,
            bin_seconds=args.bin_seconds, out_path=out_path,
            diff_method=args.diff_method, comp_rad_pix=args.comp_rad_pix)
        _close(fig)
        written.append(out_path)
    return written


def main(argv=None):
    """CLI entry point. Returns 0 on success."""
    args = build_parser().parse_args(argv)

    # Force a non-interactive backend for batch rendering.
    P.use_headless()

    # Resolve archive_root / date, honoring --night-dir.
    archive_root = args.archive_root
    obsdate = args.date
    if args.night_dir:
        archive_root, obsdate = _derive_from_night_dir(args.night_dir)

    obs_hyph = _paths.normalize_obsdate(obsdate)

    print(f"[lightcurve_analysis] {args.telescope} {obs_hyph}")
    print("  building night catalog ...")
    cat = build_night_catalog(
        args.telescope, obsdate,
        archive_root=archive_root, sim_root=args.sim_root,
        cache_dir=args.cache_dir, rebuild=args.rebuild)
    print(f"  catalog: {len(cat.stars)} stars, "
          f"{len(cat.time_axis)} time samples, "
          f"{len(cat.minutes)} minute(s)")

    if args.attach_gaia:
        print("  attaching Gaia magnitudes ...")
        try:
            if __package__:
                from .gaia import attach_gaia
            else:
                from gaia import attach_gaia  # type: ignore
            cat = attach_gaia(cat, search_radius_deg=args.gaia_radius,
                              cache_dir=args.cache_dir)
            n_gmag = int(cat.stars['gmag'].notna().sum())
            print(f"  Gaia: matched gmag for {n_gmag} star(s)")
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            print(f"  [warn] attach_gaia failed: {type(exc).__name__}: {exc}")

    star_ids = _select_star_ids(cat, args)
    print(f"  rendering {len(star_ids)} star(s) (kind={args.kind})")

    out_dir = Path(args.out) / args.telescope / obs_hyph
    out_dir.mkdir(parents=True, exist_ok=True)

    # catalog summary
    summary_path = out_dir / 'catalog_summary.csv'
    cat.stars.to_csv(summary_path, index=False)

    # per-star figures
    written = []
    for sid in tqdm(star_ids, desc='stars'):
        try:
            written.extend(_render_star(cat, int(sid), out_dir, args))
        except Exception as exc:  # noqa: BLE001 - one bad star shouldn't abort
            print(f"  [warn] star {sid} failed: "
                  f"{type(exc).__name__}: {exc}")

    grid_paths = []
    if args.grid and star_ids:
        print("  rendering grid pages ...")
        try:
            grid_paths = _render_grids(cat, star_ids, out_dir, args)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] grid rendering failed: "
                  f"{type(exc).__name__}: {exc}")

    # manifest
    manifest = {
        'telescope': args.telescope,
        'obsdate': obs_hyph,
        'generated_at': _dt.datetime.now(_dt.timezone.utc).isoformat(),
        'params': {
            'kind': args.kind,
            'bin_seconds': args.bin_seconds,
            'snr_min': args.snr_min,
            'gmag': list(args.gmag) if args.gmag else None,
            'star_ids': args.star_ids,
            'max_stars': args.max_stars,
            'diff_method': args.diff_method,
            'comp_rad_pix': args.comp_rad_pix,
            'attach_gaia': args.attach_gaia,
            'gaia_radius': args.gaia_radius,
            'rebuild': args.rebuild,
            'grid': args.grid,
            'archive_root': str(archive_root) if archive_root else None,
            'sim_root': args.sim_root,
        },
        'counts': {
            'n_stars_catalog': int(len(cat.stars)),
            'n_stars_rendered': len(star_ids),
            'n_figures': len(written),
            'n_grid_pages': len(grid_paths),
            'n_time_samples': int(len(cat.time_axis)),
            'n_minutes': len(cat.minutes),
        },
        'cache_dir': str(cat.cache_dir) if cat.cache_dir else None,
        'star_ids_rendered': [int(s) for s in star_ids],
        'output_dir': str(out_dir),
    }
    manifest_path = out_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"  wrote {len(written)} figure(s) + "
          f"{len(grid_paths)} grid page(s)")
    print(f"  summary  -> {summary_path}")
    print(f"  manifest -> {manifest_path}")
    print(f"  output   -> {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
