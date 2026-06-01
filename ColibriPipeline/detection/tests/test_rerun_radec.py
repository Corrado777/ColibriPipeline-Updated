"""Sim-guarded test: rerun det files carry real RA/Dec from the pos.npy file.

Points at the sim archive via COLIBRI_ENV/COLIBRI_SIM_ROOT, SKIPS when the sim
data isn't present. Asserts the RA/Dec written into header line index 6 of a
produced det_*.txt are finite and equal pos[star_idx, 3:5] for that minute.
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault('COLIBRI_ENV', 'sim')
os.environ.setdefault(
    'COLIBRI_SIM_ROOT',
    '/home/agirmen/research_data/ColibriPipelineSimulatedDirs')

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
except Exception:  # pragma: no cover
    resolve_archive_root = None
    night_dir = None

TELESCOPE = 'GREENBIRD'
OBSDATE = '2025-08-30'

_SIM_ROOT = '/home/agirmen/research_data/ColibriPipelineSimulatedDirs'


def _require_sim():
    if not Path(_SIM_ROOT).exists():
        pytest.skip(f"sim root absent: {_SIM_ROOT}")
    if resolve_archive_root is None:
        pytest.skip("lightcurve_analysis.paths unavailable")
    ndir = night_dir(resolve_archive_root(telescope=TELESCOPE), OBSDATE)
    if not ndir.exists() or not list(ndir.glob('*_stars.npy')):
        pytest.skip(f"sim night dir missing stars.npy: {ndir}")
    return ndir


def test_rerun_radec_matches_pos():
    ndir = _require_sim()

    config = DetectionConfig(detector='box')
    result = rerun_night(TELESCOPE, OBSDATE, config)
    assert result['n_detections'] >= 1, "expected at least one detection"

    out_dir = Path(result['out_dir'])
    det_files = sorted(out_dir.glob('det_*.txt'))
    assert det_files, "no det_*.txt produced"

    # Pick a det file, recover its star index from the _star<idx>_ token.
    det = det_files[0]
    m = re.search(r'_star(\d+)_', det.name)
    assert m, f"could not parse star idx from {det.name}"
    star_idx = int(m.group(1))

    # Parse header line index 6: '#    RA Dec Coords: <ra> <dec>'
    with open(det) as fh:
        lines = fh.readlines()
    line6 = lines[6]
    assert 'RA Dec Coords' in line6, line6
    tail = line6.split(':', 1)[1].split()
    ra, dec = float(tail[0]), float(tail[1])
    assert np.isfinite(ra) and np.isfinite(dec), (ra, dec)

    # Load the sibling pos file and compare.
    pos_files = sorted(ndir.glob('*sig_pos.npy'))
    assert pos_files, "no sig_pos.npy in night dir"
    pos = np.asarray(np.load(pos_files[0], allow_pickle=True), dtype=float)
    assert pos.ndim == 2 and pos.shape[1] >= 5
    assert abs(ra - pos[star_idx, 3]) < 1e-6
    assert abs(dec - pos[star_idx, 4]) < 1e-6
