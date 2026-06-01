"""Tests for detection.width.canonical_width_from_radec (no data needed)."""

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_COLIBRI_DIR = _THIS.parent.parent.parent   # ColibriPipeline/
if str(_COLIBRI_DIR) not in sys.path:
    sys.path.insert(0, str(_COLIBRI_DIR))

from detection.config import DetectionConfig            # noqa: E402
from detection.width import canonical_width_from_radec  # noqa: E402

OBSDATE = '2025-08-30'


def test_at_opposition_returns_anchor():
    # opposition_angle_deg=0 hits the formula path (cos 0 == 1) so width == anchor.
    cfg = DetectionConfig(width_at_opposition=6, opposition_angle_deg=0.0)
    w = canonical_width_from_radec(123.0, -10.0, OBSDATE, cfg.exposure_time, cfg)
    assert w == 6


def test_off_opposition_is_wider():
    cfg0 = DetectionConfig(width_at_opposition=6, opposition_angle_deg=0.0)
    cfg60 = DetectionConfig(width_at_opposition=6, opposition_angle_deg=60.0)
    w0 = canonical_width_from_radec(0.0, 0.0, OBSDATE, cfg0.exposure_time, cfg0)
    w60 = canonical_width_from_radec(0.0, 0.0, OBSDATE, cfg60.exposure_time, cfg60)
    assert w60 > w0


def test_explicit_canonical_width_override():
    cfg = DetectionConfig(canonical_width=11)
    w = canonical_width_from_radec(0.0, 0.0, OBSDATE, cfg.exposure_time, cfg)
    assert w == 11


def test_real_radec_at_opposition_returns_anchor():
    # Compute an RA/Dec that is exactly at opposition for OBSDATE (anti-solar
    # point), so opposition_angle_from_radec ~ 0 -> width == anchor. Skips if
    # astropy/ephemeris is unavailable.
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord, get_sun
        from astropy.time import Time
    except Exception:
        import pytest
        pytest.skip("astropy unavailable")

    t = Time(OBSDATE)
    sun = get_sun(t)
    anti = SkyCoord(ra=sun.ra + 180 * u.deg, dec=-sun.dec,
                    frame='gcrs', obstime=t).icrs
    cfg = DetectionConfig(width_at_opposition=6)  # no angle override -> uses radec
    w = canonical_width_from_radec(float(anti.ra.deg), float(anti.dec.deg),
                                   OBSDATE, cfg.exposure_time, cfg)
    assert w == 6
