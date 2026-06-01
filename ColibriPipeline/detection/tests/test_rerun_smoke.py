"""Smoke test for detection.rerun against the simulated archive tree.

Mirrors lightcurve_analysis/tests/test_smoke_sim.py: points at the sim archive
via COLIBRI_ENV/COLIBRI_SIM_ROOT, and SKIPS (rather than fails) when the sim
data isn't present. No network.

Asserts that ``rerun_night``:
  * returns without error,
  * writes into ``<night>/rerun_box`` (out_dir ends with ``rerun_box``),
  * out_dir exists and is DISTINCT from the live night dir,
  * does NOT clobber the live night dir's pre-existing ``det_*.txt`` (count
    unchanged before/after).
"""

import os
import sys
from pathlib import Path

import pytest

# --- Environment: point at the simulated archive tree (set before imports) ---
os.environ.setdefault('COLIBRI_ENV', 'sim')
os.environ.setdefault(
    'COLIBRI_SIM_ROOT',
    '/home/agirmen/research_data/ColibriPipelineSimulatedDirs')

# --- Make the package importable (tests live two levels deep) ----------------
# .../ColibriPipeline/detection/tests/test_rerun_smoke.py
_THIS = Path(__file__).resolve()
_PKG_DIR = _THIS.parent.parent              # detection/
_COLIBRI_DIR = _PKG_DIR.parent              # ColibriPipeline/
for _p in (str(_COLIBRI_DIR), str(_PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from detection.config import DetectionConfig          # noqa: E402
from detection.rerun import rerun_night               # noqa: E402

try:
    from lightcurve_analysis.paths import (            # noqa: E402
        resolve_archive_root, night_dir)
except Exception:  # pragma: no cover - sibling package missing
    resolve_archive_root = None
    night_dir = None


TELESCOPE = 'GREENBIRD'
OBSDATE = '2025-08-30'


def _live_night_dir():
    root = resolve_archive_root(telescope=TELESCOPE)
    return night_dir(root, OBSDATE)


def _require_sim_data():
    """Skip the test unless the sim night dir with a stars.npy is present."""
    if resolve_archive_root is None or night_dir is None:
        pytest.skip("lightcurve_analysis.paths unavailable")
    try:
        ndir = _live_night_dir()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"could not resolve archive root: {exc}")
    if not ndir.exists():
        pytest.skip(f"sim night dir not present: {ndir}")
    if not list(ndir.glob('*_stars.npy')):
        pytest.skip(f"no *_stars.npy in sim night dir: {ndir}")
    return ndir


def test_rerun_night_smoke():
    ndir = _require_sim_data()

    # Capture the live det_*.txt count BEFORE the rerun (these must not change).
    live_dets_before = sorted(ndir.glob('det_*.txt'))
    n_live_before = len(live_dets_before)

    config = DetectionConfig(detector='box', sigma_threshold=5.0)
    result = rerun_night(TELESCOPE, OBSDATE, config)

    # Returns a well-formed dict.
    assert isinstance(result, dict)
    assert 'out_dir' in result and 'n_minutes' in result
    assert 'n_detections' in result and 'detections' in result

    out_dir = Path(result['out_dir'])

    # out_dir ends with rerun_box, exists, and is distinct from the live night dir.
    assert out_dir.name == 'rerun_box', out_dir
    assert out_dir.exists()
    assert out_dir.resolve() != ndir.resolve()
    assert out_dir.parent.resolve() == ndir.resolve()

    # Summary CSV written.
    assert (out_dir / 'rerun_summary.csv').exists()

    # Live det_*.txt count unchanged (separate output dir => no clobbering).
    n_live_after = len(list(ndir.glob('det_*.txt')))
    assert n_live_after == n_live_before, (
        f"live det_*.txt count changed: {n_live_before} -> {n_live_after}")

    # At least the minute(s) were processed; detection count is non-negative.
    assert result['n_minutes'] >= 1
    assert result['n_detections'] >= 0
    # Any detection files actually live inside out_dir, not the live night dir.
    for d in result['detections']:
        assert Path(d['path']).parent.resolve() == out_dir.resolve()
