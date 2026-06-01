"""Colibri photometry module.

Pure photometry: turning frames into per-star flux time series. No I/O, no
detection logic (dip detection lives in the ``detection/`` package).

This package is a thin, curated facade over the in-place Cython
``colibri_photometry`` module: the Cython source (``colibri_photometry.py`` /
``.pyx``) STAYS at the top level (the ``Init/`` build scripts compile it on the
telescopes), and this package re-exports its public functions so callers import
from one clean surface. Mirrors the wrapping pattern of ``colibri_io.raw``
(which wraps the in-place ``colibri_image_reader``) and the package conventions
of ``detection/`` and ``lightcurve_analysis/``.

The legacy top-level import path (``import colibri_photometry`` /
``from colibri_photometry import initialFind``) still works unchanged; this
package is additive.

Public API
----------
``initialFind``       -- SEP background + source extraction (star list).
``refineCentroid``    -- SEP windowed centroiding.
``averageDrift``      -- median x/y drift rate from first/last frame positions.
``timeEvolve``        -- per-frame drift-corrected coords + aperture flux.
``getStationaryFlux`` -- bulk aperture flux for non-drifting stars (stacked).
``sumFlux``           -- square-aperture flux sum (deprecated; kept for compat).
``clipCutStars`` / ``clipCutStars3D`` -- edge-proximity star rejection.
``timeEvolve3D``      -- stacked-array drift flux (NOT implemented upstream).
``dipDetection``      -- back-compat re-export only; new code uses ``detection/``.
"""

from photometry.core import (
    initialFind,
    refineCentroid,
    sumFlux,
    clipCutStars,
    clipCutStars3D,
    averageDrift,
    timeEvolve,
    timeEvolve3D,
    getStationaryFlux,
    dipDetection,
)

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
