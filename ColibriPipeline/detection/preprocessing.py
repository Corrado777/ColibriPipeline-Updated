"""Light-curve preprocessing for the detection module.

Ports the legacy detrend used by ``colibri_photometry.dipDetection`` (and the
trade-study ``MeanSubtract`` preprocessor): subtract a boxcar running mean and
report the scalar residual std as the noise level. Dependency-light: numpy +
scipy only.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


def mean_subtract(flux, window: int = 200):
    """Boxcar running-mean subtraction (legacy detrend).

    Mirrors ``colibri_photometry.py`` lines 495-497 exactly::

        smoothed = uniform_filter1d(flux, size=window, mode='reflect')
        residual = flux - smoothed

    Parameters
    ----------
    flux : array_like
        1-D flux profile of a single star.
    window : int
        Boxcar width in frames (default 200 = 5 s at 40 Hz).

    Returns
    -------
    residual : np.ndarray
        ``flux - smoothed``; same length as ``flux``. Flat-baseline residual
        ready for matched filtering.
    sigma : float
        Scalar noise estimate ``float(np.std(residual))``.
    """
    flux = np.asarray(flux, dtype=float)
    smoothed = uniform_filter1d(flux, size=int(window), mode="reflect")
    residual = flux - smoothed
    sigma = float(np.std(residual))
    return residual, sigma
