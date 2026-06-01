"""Tests for detection.preprocessing.mean_subtract."""

import numpy as np

from detection.preprocessing import mean_subtract


def test_removes_constant():
    flux = np.full(500, 1234.0)
    residual, sigma = mean_subtract(flux, window=200)
    assert len(residual) == len(flux)
    assert np.allclose(residual, 0.0, atol=1e-9)
    assert sigma == 0.0 or abs(sigma) < 1e-9


def test_removes_linear_trend_interior():
    # A boxcar running-mean detrend turns a linear ramp into a FLAT residual in
    # the interior (an even-width boxcar leaves a constant offset, not exactly
    # zero, but the slope is fully removed -- which is what detrending needs).
    n = 1000
    flux = 100.0 + 0.5 * np.arange(n)
    residual, sigma = mean_subtract(flux, window=200)
    assert len(residual) == n
    interior = residual[300:700]
    assert np.allclose(interior, interior[0], atol=1e-6)
    assert np.allclose(np.diff(interior), 0.0, atol=1e-6)


def test_residual_length_matches_input():
    rng = np.random.default_rng(0)
    flux = 1000.0 + rng.normal(0, 5, size=777)
    residual, sigma = mean_subtract(flux, window=200)
    assert residual.shape == flux.shape
    assert sigma > 0.0
