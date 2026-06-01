# -*- coding: utf-8 -*-
"""
Core photometry: re-exports of the in-place Cython ``colibri_photometry`` module.

The Cython source (``colibri_photometry.py`` / ``.pyx``) is compiled in place by
the ``Init/`` build scripts and MUST stay at the top level so the build
artifacts (.so/.pyd) resolve correctly. This module does NOT relocate it; it
just provides a ``photometry.core`` namespace that re-exports its public
functions, so callers can use either ``import colibri_photometry`` (legacy) or
``from photometry import initialFind`` (new).

Mirrors the wrapping pattern of ``colibri_io.raw`` (which wraps the in-place
Cython ``colibri_image_reader``).

``dipDetection`` still lives in ``colibri_photometry`` and is re-exported here
for back-compat only -- new code should use the ``detection/`` package, which is
the canonical home of dip-detection logic. No detection logic is duplicated here.
"""

# Stages run flat from the ColibriPipeline/ dir, so plain top-level imports
# work. When imported as part of the package (python -m / pytest) the package
# parent is on sys.path via the same flat layout; guard just in case.
try:
    import colibri_photometry as _cp
except ImportError:  # pragma: no cover - defensive for non-flat invocation
    import sys as _sys
    from pathlib import Path as _Path
    _pkg_parent = _Path(__file__).resolve().parent.parent
    if str(_pkg_parent) not in _sys.path:
        _sys.path.insert(0, str(_pkg_parent))
    import colibri_photometry as _cp

# --- star detection / centroiding (verified against colibri_photometry) ---
initialFind = _cp.initialFind
refineCentroid = _cp.refineCentroid

# --- image manipulation / aperture tools ---
sumFlux = _cp.sumFlux
clipCutStars = _cp.clipCutStars
clipCutStars3D = _cp.clipCutStars3D

# --- star drift correction + per-frame flux extraction ---
averageDrift = _cp.averageDrift
timeEvolve = _cp.timeEvolve
timeEvolve3D = _cp.timeEvolve3D
getStationaryFlux = _cp.getStationaryFlux

# --- dip detection (back-compat re-export; canonical home is detection/) ---
dipDetection = _cp.dipDetection

__all__ = [
    "initialFind",
    "refineCentroid",
    "sumFlux",
    "clipCutStars",
    "clipCutStars3D",
    "averageDrift",
    "timeEvolve",
    "timeEvolve3D",
    "getStationaryFlux",
    "dipDetection",
]
