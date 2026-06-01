"""Sim-guarded test for detection.rerun.rerun_match across Green + Blue.

Green and Blue sim data are byte-identical, so every Green detection has a
coincident Blue partner -> tier-2 matches. SKIPS when sim data (or the Blue
dir) is absent.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault('COLIBRI_ENV', 'sim')
os.environ.setdefault(
    'COLIBRI_SIM_ROOT',
    '/home/agirmen/research_data/ColibriPipelineSimulatedDirs')

_THIS = Path(__file__).resolve()
_PKG_DIR = _THIS.parent.parent
_COLIBRI_DIR = _PKG_DIR.parent
for _p in (str(_COLIBRI_DIR), str(_PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from detection.config import DetectionConfig          # noqa: E402
from detection.rerun import rerun_match               # noqa: E402

_SIM_ROOT = '/home/agirmen/research_data/ColibriPipelineSimulatedDirs'
OBSDATE = '2025-08-30'


def _require_sim():
    root = Path(_SIM_ROOT)
    if not root.exists():
        pytest.skip(f"sim root absent: {_SIM_ROOT}")
    green = root / 'Green' / 'ColibriArchive' / OBSDATE
    blue = root / 'Blue' / 'ColibriArchive' / OBSDATE
    if not green.exists() or not list(green.glob('*_stars.npy')):
        pytest.skip(f"green sim night missing: {green}")
    if not blue.exists() or not list(blue.glob('*_stars.npy')):
        pytest.skip(f"blue sim night missing: {blue}")


def test_rerun_match_green_blue():
    _require_sim()

    config = DetectionConfig(detector='box')
    result = rerun_match(OBSDATE, config,
                         telescopes=['GREENBIRD', 'BLUEBIRD'],
                         sim_root=_SIM_ROOT)

    assert result['n_active'] == 2
    assert set(result['active']) == {'GREENBIRD', 'BLUEBIRD'}

    matched_dir = Path(result['matched_dir'])
    assert matched_dir.exists()

    # At least one Tier directory with >= 1 det file (Green/Blue identical).
    tier_dirs = sorted(matched_dir.glob('*-Tier*'))
    assert tier_dirs, f"no Tier dirs in {matched_dir}"
    assert any(list(td.glob('det_*.txt')) for td in tier_dirs)
