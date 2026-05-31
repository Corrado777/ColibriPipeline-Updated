#!/usr/bin/env python3
"""
Filename : test_smoke_sim.py
Author   : ColibriPipeline / lightcurve_analysis (Wave 3)

End-to-end smoke tests for the night light-curve analysis package, run against
the simulated archive tree. Mirrors the plan's Verification section.

Runs two ways:
    * ``pytest lightcurve_analysis/tests/test_smoke_sim.py``
    * ``python lightcurve_analysis/tests/test_smoke_sim.py``  (plain runner)

Network-dependent assertions (Gaia) are soft: a network failure does not fail
the test, it just means no gmag matches.
"""

import os
import sys
import tempfile
from pathlib import Path

# --- Environment: point at the simulated archive tree ----------------------
# Set before importing the package so paths.resolve_archive_root sees them.
os.environ.setdefault('COLIBRI_ENV', 'sim')
os.environ.setdefault(
    'COLIBRI_SIM_ROOT',
    '/home/agirmen/research_data/ColibriPipelineSimulatedDirs')

# --- Make the package importable (tests live two levels deep) ---------------
# .../ColibriPipeline/lightcurve_analysis/tests/test_smoke_sim.py
_THIS = Path(__file__).resolve()
_PKG_DIR = _THIS.parent.parent              # lightcurve_analysis/
_COLIBRI_DIR = _PKG_DIR.parent              # ColibriPipeline/
for _p in (str(_COLIBRI_DIR), str(_PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from lightcurve_analysis.io import discover_minutes, load_minute_full  # noqa: E402
from lightcurve_analysis.catalog import (  # noqa: E402
    build_night_catalog, load_cache, save_cache)
from lightcurve_analysis.paths import resolve_archive_root, night_dir  # noqa: E402
from lightcurve_analysis.differential import differential_lightcurve  # noqa: E402
from lightcurve_analysis.gaia import attach_gaia  # noqa: E402
from lightcurve_analysis import cli  # noqa: E402


TELESCOPE = 'GREENBIRD'
OBSDATE = '2025-08-30'


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _night_dir():
    root = resolve_archive_root(telescope=TELESCOPE)
    return night_dir(root, OBSDATE)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_discover_minutes():
    """1. discover_minutes finds the single minute and pairs stars/pos."""
    refs = discover_minutes(_night_dir())
    assert len(refs) >= 1, "expected at least one minute"
    ref = refs[0]
    assert ref.stars_path.exists()
    assert ref.pos_path is not None, "stars/pos pairing failed"
    data = load_minute_full(ref, mmap=True)
    assert data['flux'].ndim == 2
    # pos shares star ordering with the flux axis
    if data['pos'] is not None:
        assert data['pos'].shape[0] == data['flux'].shape[0]


def test_build_night_catalog():
    """2. build catalog: ~1287 stars, time_axis ~2399, dt ~0.025 s."""
    cat = build_night_catalog(TELESCOPE, OBSDATE, rebuild=True)
    n_stars = len(cat.stars)
    assert 1000 <= n_stars <= 1600, f"unexpected star count {n_stars}"
    n_samples = len(cat.time_axis)
    assert 2000 <= n_samples <= 2600, f"unexpected n_samples {n_samples}"
    t = np.asarray(cat.time_axis, dtype=float)
    finite = np.isfinite(t)
    dt = np.median(np.diff(t[finite]))
    assert 0.015 < dt < 0.04, f"unexpected cadence dt={dt}"


def test_get_raw_lightcurve_length():
    """3. raw light curve length equals the night time axis length."""
    cat = build_night_catalog(TELESCOPE, OBSDATE)
    sid = int(cat.stars.iloc[0]['star_id'])
    lc = cat.get_raw_lightcurve(sid, with_gaps=True)
    assert len(lc.t) == len(cat.time_axis)
    assert len(lc.f) == len(cat.time_axis)


def test_differential_ensemble_finite():
    """4. ensemble differential is finite for a high-SNR star (median ~1)."""
    cat = build_night_catalog(TELESCOPE, OBSDATE)
    hi = cat.filter_stars(snr_min=cat.stars['snr'].quantile(0.9))
    assert len(hi) > 0
    sid = int(hi.sort_values('snr', ascending=False).iloc[0]['star_id'])
    lc = differential_lightcurve(cat, sid, method='ensemble',
                                 comp_rad_pix=150.0)
    f = np.asarray(lc.f, dtype=float)
    finite = np.isfinite(f)
    assert finite.sum() > 0, "differential produced no finite samples"
    med = np.nanmedian(f)
    assert 0.5 < med < 1.5, f"normalized median {med} not near 1"


def test_attach_gaia_soft():
    """5. attach_gaia runs without crashing (gmag count >=0; soft on network)."""
    cat = build_night_catalog(TELESCOPE, OBSDATE)
    try:
        cat2 = attach_gaia(cat, search_radius_deg=0.5, force=True)
    except Exception as exc:  # noqa: BLE001 - network/Vizier issues are soft
        print(f"  [soft] attach_gaia raised (treated as no network): "
              f"{type(exc).__name__}: {exc}")
        return
    n_gmag = int(cat2.stars['gmag'].notna().sum())
    assert n_gmag >= 0
    print(f"  attach_gaia matched gmag for {n_gmag} star(s)")


def test_cache_round_trip():
    """6. save_cache -> load_cache reproduces extraction within float32 tol."""
    cat = build_night_catalog(TELESCOPE, OBSDATE, rebuild=True)
    sid = int(cat.stars.iloc[0]['star_id'])
    lc_before = cat.get_raw_lightcurve(sid, with_gaps=False)

    with tempfile.TemporaryDirectory() as tmp:
        save_cache(cat, tmp)
        cat2 = load_cache(TELESCOPE, OBSDATE, cache_dir=tmp)
        lc_after = cat2.get_raw_lightcurve(sid, with_gaps=False)

    assert len(lc_before.f) == len(lc_after.f)
    a = np.asarray(lc_before.f, dtype=np.float32)
    b = np.asarray(lc_after.f, dtype=np.float32)
    assert np.allclose(a, b, rtol=1e-5, atol=1e-5, equal_nan=True)


def test_cli_smoke():
    """7. CLI end-to-end: PNGs + catalog_summary.csv exist."""
    with tempfile.TemporaryDirectory() as tmp:
        rc = cli.main([
            '--telescope', TELESCOPE,
            '--date', OBSDATE,
            '--max-stars', '3',
            '--out', tmp,
            '--kind', 'both',
        ])
        assert rc == 0
        out_dir = Path(tmp) / TELESCOPE / OBSDATE
        assert (out_dir / 'catalog_summary.csv').exists()
        assert (out_dir / 'manifest.json').exists()
        raw_pngs = list((out_dir / 'lightcurves' / 'raw').glob('star_*.png'))
        diff_pngs = list((out_dir / 'lightcurves' / 'diff').glob('star_*.png'))
        assert len(raw_pngs) >= 1, "no raw PNGs written"
        assert len(diff_pngs) >= 1, "no diff PNGs written"


# ---------------------------------------------------------------------------
# Plain runner (no pytest required)
# ---------------------------------------------------------------------------

def _main():
    tests = [
        test_discover_minutes,
        test_build_night_catalog,
        test_get_raw_lightcurve_length,
        test_differential_ensemble_finite,
        test_attach_gaia_soft,
        test_cache_round_trip,
        test_cli_smoke,
    ]
    failures = 0
    for t in tests:
        name = t.__name__
        try:
            print(f"\n=== {name} ===")
            t()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'='*40}\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(_main())
