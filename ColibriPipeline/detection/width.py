"""Canonical box width from the field's opposition angle.

Motivation: a single matched-filter width keeps the per-frame false-alarm rate
low (one test per frame instead of a width bank) while still spanning the core
dip of occultations across a range of event lengths. Off opposition the KBO
shadow crosses more slowly, so events get longer -> the canonical width grows.

Pipeline reality: at primary-detection time the field is only a categorical
name (one of ~24 fields); RCD headers carry no RA/Dec. So auto-compute needs a
field-name -> (RA, Dec) lookup (``field_centers.json``) plus a solar ephemeris.
An explicit override (``config.canonical_width`` or
``config.opposition_angle_deg``) is the safety valve when a field is missing.

Calibration (the formula chosen here)
--------------------------------------
The shadow's transverse velocity across the telescope is dominated by Earth's
reflex motion. At opposition the line of sight points exactly anti-solar, so
the full reflex velocity projects onto the sky; the trade-study uses the KBO
reflex velocity ``vT(a, vE) = vE * (1 - sqrt(1/a))`` (Earth orbital speed
``vE = 29800 m/s``, ``a ~ 40 AU``) as the opposition shadow speed.

Off opposition by an angle ``theta`` (separation from the anti-solar point), the
component of Earth's reflex motion that is transverse to the line of sight is
reduced by ``cos(theta)`` -- the geometry projects less of the reflex velocity
across the field of view -- so::

    v_shadow(theta) = vT(a, vE) * cos(theta)

The Fresnel-scale core-dip *length on the sky* is fixed by the diffraction
physics (independent of where we look), so the event *duration* (and hence the
matched-filter width in frames) scales as ``1 / v_shadow``::

    width(theta) = width_at_opposition / cos(theta)

This anchors exactly at opposition (``theta = 0`` -> ``cos = 1`` ->
``width_at_opposition`` frames, ~6) and grows as the field moves off
opposition (slower shadow -> longer event -> wider box). The absolute vT scale
cancels out of the ratio; it is documented above for the physical reasoning and
to make the opposition anchor explicit. Result is rounded to the nearest int
and clamped to >= 1. ``cos(theta)`` is floored at a small positive value so the
width stays finite near the (unobserved) 90-degree limit.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from .config import DetectionConfig

# Physical constants (ported from detection_trade_study/fresnel_physics.py).
_V_EARTH = 29800.0       # Earth orbital speed, m/s
_KBO_DISTANCE_AU = 40.0  # nominal KBO distance, AU

# Don't let 1/cos blow up near the horizon of observability.
_MIN_COS = 0.05

_FIELD_CENTERS_PATH = Path(__file__).with_name("field_centers.json")


def _vT(a: float, vE: float) -> float:
    """KBO reflex transverse velocity (m/s); ported from fresnel_physics.vT."""
    return vE * (1.0 - (1.0 / a) ** 0.5)


def _width_from_angle(opposition_angle_deg: float, config: DetectionConfig) -> int:
    """width = width_at_opposition / cos(theta); rounded, clamped >= 1."""
    theta = math.radians(float(opposition_angle_deg))
    cos_theta = max(_MIN_COS, abs(math.cos(theta)))
    width = config.width_at_opposition / cos_theta
    return max(1, int(round(width)))


def opposition_angle_from_radec(ra_deg, dec_deg, obs_date) -> Optional[float]:
    """Angular separation (deg) between a sky point and the anti-solar point.

    Computes the separation between ``(ra_deg, dec_deg)`` (ICRS, degrees) and the
    anti-solar (opposition) direction at ``obs_date`` using astropy
    ``get_sun(Time(obs_date))``. Returns None on any failure (no ephemeris, bad
    coords) so callers can fall back gracefully.
    """
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord, get_sun
        from astropy.time import Time

        t = Time(obs_date)
        sun = get_sun(t)
        # Anti-solar point = opposition direction.
        anti_solar = SkyCoord(
            ra=(sun.ra + 180 * u.deg),
            dec=-sun.dec,
            frame="gcrs",
            obstime=t,
        ).icrs

        field = SkyCoord(
            ra=float(ra_deg) * u.deg,
            dec=float(dec_deg) * u.deg,
            frame="icrs",
        )
        sep = field.separation(anti_solar)
        return float(sep.deg)
    except Exception:
        return None


def _opposition_angle_from_field(field_name: str, obs_date) -> Optional[float]:
    """Angular separation (deg) between the field center and the anti-solar
    point at ``obs_date``. Returns None on any failure (missing field, no
    ephemeris) so the caller can fall back gracefully.
    """
    try:
        if not _FIELD_CENTERS_PATH.exists():
            return None
        with open(_FIELD_CENTERS_PATH) as fh:
            centers = json.load(fh)

        entry = centers.get(field_name)
        if not isinstance(entry, dict) or "ra_deg" not in entry or "dec_deg" not in entry:
            return None

        return opposition_angle_from_radec(entry["ra_deg"], entry["dec_deg"], obs_date)
    except Exception:
        return None


def canonical_width_frames(
    field_name: Optional[str],
    obs_date,
    exposure_time: float,
    config: DetectionConfig,
) -> int:
    """Return the canonical box width in frames.

    Resolution precedence (highest first):
      1. ``config.canonical_width`` (explicit width override) -> returned as-is.
      2. ``config.opposition_angle_deg`` (explicit angle) -> width from formula.
      3. Auto: field_name -> RA/Dec (field_centers.json); opposition angle via
         astropy ``get_sun`` at ``obs_date`` (anti-solar separation); width from
         the cos-scaling formula (see module docstring).
      4. Fallback (field missing / no ephemeris): ``config.width_at_opposition``.

    Always returns a positive int >= 1; never raises.
    """
    # 1. Explicit width override.
    if config.canonical_width is not None:
        return max(1, int(config.canonical_width))

    # 2. Explicit opposition angle.
    if config.opposition_angle_deg is not None:
        return _width_from_angle(config.opposition_angle_deg, config)

    # 3. Auto from field name + ephemeris.
    if field_name is not None:
        angle = _opposition_angle_from_field(field_name, obs_date)
        if angle is not None:
            return _width_from_angle(angle, config)

    # 4. Fallback.
    return max(1, int(config.width_at_opposition))


def canonical_width_from_radec(
    ra_deg: float,
    dec_deg: float,
    obs_date,
    exposure_time: float,
    config: DetectionConfig,
) -> int:
    """Return the canonical box width in frames from an explicit RA/Dec center.

    Like :func:`canonical_width_frames` but the opposition angle is derived from
    an actual sky position (e.g. a minute's field-center RA/Dec from the pos
    file) rather than a categorical field name.

    Resolution precedence (highest first):
      1. ``config.canonical_width`` (explicit width override) -> returned as-is.
      2. ``config.opposition_angle_deg`` (explicit angle) -> width from formula.
      3. ``opposition_angle_from_radec(ra_deg, dec_deg, obs_date)`` -> width from
         the cos-scaling formula (see module docstring).
      4. Fallback (no ephemeris / bad coords): ``config.width_at_opposition``.

    Always returns a positive int >= 1; never raises.
    """
    # 1. Explicit width override.
    if config.canonical_width is not None:
        return max(1, int(config.canonical_width))

    # 2. Explicit opposition angle.
    if config.opposition_angle_deg is not None:
        return _width_from_angle(config.opposition_angle_deg, config)

    # 3. Auto from RA/Dec + ephemeris.
    angle = opposition_angle_from_radec(ra_deg, dec_deg, obs_date)
    if angle is not None:
        return _width_from_angle(angle, config)

    # 4. Fallback.
    return max(1, int(config.width_at_opposition))
