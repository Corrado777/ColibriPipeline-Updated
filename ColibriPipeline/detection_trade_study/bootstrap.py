"""
Bootstrap independent synthetic telescopes from one real light curve.

Motivation
----------
A true "power of three" test needs three telescopes that see the SAME stellar
signal through INDEPENDENT noise.  The bundled sim minute does not provide this
(Green and Blue are byte-identical copies and Red is a 5 s partial), so a joint
statistic computed over the raw scopes would just stack correlated noise.

This module synthesises N independent telescopes from a single real, positive-
flux light curve by separating it into a smooth baseline plus residual noise,
then re-drawing the residuals with a moving-block bootstrap (which preserves the
short-timescale correlation of scintillation).  Each synthetic telescope =
baseline + a fresh residual draw.  A real occultation injected at the same time
into all of them is coincident; the bootstrapped noise is independent -- exactly
the regime in which the sqrt(N) joint statistic pays off.

This directly serves the user's request to "bootstrap the minute of light-curve
data to test sensitivity".  When genuine independent per-telescope minutes are
available, the harness can instead run in 'real' mode and skip this module.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


def decompose(flux, window_frames=200):
    """Split a light curve into (smooth baseline, residual noise).

    baseline = uniform_filter1d(flux, window_frames, mode='reflect')
    residual = flux - baseline
    """
    flux = np.asarray(flux, dtype=float)
    baseline = uniform_filter1d(flux, size=window_frames, mode='reflect')
    return baseline, flux - baseline


def block_bootstrap_residual(residual, rng, block=20):
    """Moving-block bootstrap of a residual series, preserving its length.

    Draws random length-`block` contiguous segments (with replacement) from the
    residual and concatenates them until the original length is reached.  Block
    resampling retains short-timescale correlation (e.g. scintillation) that an
    i.i.d. shuffle would destroy, giving a realistic false-alarm background.

    Parameters
    ----------
    residual : ndarray, 1-D
    rng : numpy.random.Generator
    block : int
        Block length in frames (default 20 = 0.5 s at 40 Hz).

    Returns
    -------
    ndarray, same length as residual.
    """
    residual = np.asarray(residual, dtype=float)
    n = len(residual)
    if n == 0:
        return residual.copy()
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    max_start = n - block
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    out = np.concatenate([residual[s:s + block] for s in starts])
    return out[:n]


def make_independent_scopes(flux, n_scopes, rng, window_frames=200, block=20,
                            names=None):
    """Build N independent synthetic telescope light curves from one real curve.

    Each output scope shares the real smooth baseline but gets an independent
    block-bootstrap draw of the real residual noise.

    Parameters
    ----------
    flux : ndarray, 1-D
        A real, positive-flux light curve for one star.
    n_scopes : int
    rng : numpy.random.Generator
    window_frames, block : int
        Passed to decompose / block_bootstrap_residual.
    names : list of str, optional
        Scope names (default ['T1', 'T2', ...]).

    Returns
    -------
    dict
        scope name -> synthetic flux ndarray (same length as `flux`).
    """
    baseline, residual = decompose(flux, window_frames)
    if names is None:
        names = [f"T{i + 1}" for i in range(n_scopes)]
    return {names[i]: baseline + block_bootstrap_residual(residual, rng, block)
            for i in range(n_scopes)}
