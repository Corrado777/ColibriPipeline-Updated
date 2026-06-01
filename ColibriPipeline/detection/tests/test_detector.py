"""Tests for detection.detector (box + ricker 9-tuple contract)."""

import numpy as np
import pytest

from detection.config import DetectionConfig
from detection.detector import BoxDipDetector, RickerDipDetector, make_detector


def _make_box_flux(n=1000, base=1000.0, depth=200.0, width=6, center=500):
    flux = np.full(n, base, dtype=float)
    half = width // 2
    start = center - half
    flux[start:start + width] -= depth
    return flux


def test_box_recovers_injected_dip():
    width = 6
    center = 500
    flux = _make_box_flux(width=width, center=center)
    cfg = DetectionConfig(detector="box")
    det = BoxDipDetector(width, cfg)
    out = det.detect(flux)
    frameNum, lc_arr, conv, lc_std, lc_mean, bkg_std, bkg_mean, minVal, sig = out

    assert frameNum != -1 and frameNum != -2
    # Peak should land within a couple frames of the injected center.
    assert abs(frameNum - center) <= 2
    assert sig >= cfg.sigma_threshold
    assert len(conv) == len(lc_arr)


def test_box_reject_empty():
    cfg = DetectionConfig(detector="box")
    det = BoxDipDetector(6, cfg)
    out = det.detect(np.array([]))
    assert out[0] == -2
    assert list(out[1]) == []
    assert out[7] == -2


def test_box_reject_short():
    cfg = DetectionConfig(detector="box")
    det = BoxDipDetector(6, cfg)
    flux = np.full(50, 1000.0)  # < 100 frames after trim
    out = det.detect(flux)
    assert out[0] == -2
    assert out[7] == -2


def test_box_conv_length():
    flux = _make_box_flux()
    cfg = DetectionConfig(detector="box")
    det = BoxDipDetector(6, cfg)
    out = det.detect(flux)
    assert len(out[2]) == len(out[1])


def test_ricker_parity_with_legacy():
    """RickerDipDetector must reproduce colibri_photometry.dipDetection."""
    legacy = pytest.importorskip("colibri_photometry")
    from astropy.convolution import RickerWavelet1DKernel

    rng = np.random.default_rng(42)
    flux = 1000.0 + rng.normal(0, 8, size=1200)
    flux[600:606] -= 150.0  # inject a dip
    sigma = 5.0

    kernel = RickerWavelet1DKernel(6)
    legacy_out = legacy.dipDetection(flux.copy(), kernel, 0, sigma)

    cfg = DetectionConfig(detector="ricker", sigma_threshold=sigma)
    det = RickerDipDetector(cfg, kernel=RickerWavelet1DKernel(6))
    new_out = det.detect(flux.copy())

    # frameNum
    assert legacy_out[0] == new_out[0]
    # lc_arr
    np.testing.assert_allclose(np.asarray(legacy_out[1], dtype=float),
                               np.asarray(new_out[1], dtype=float), rtol=0, atol=1e-9)
    # conv_padded
    np.testing.assert_allclose(np.asarray(legacy_out[2], dtype=float),
                               np.asarray(new_out[2], dtype=float), rtol=1e-9, atol=1e-6)
    # scalar fields 3..8 (handle nan)
    for i in range(3, 9):
        a = float(legacy_out[i])
        b = float(new_out[i])
        if np.isnan(a):
            assert np.isnan(b)
        else:
            assert b == pytest.approx(a, rel=1e-9, abs=1e-6)

    # conv_padded length == lc_arr length
    assert len(new_out[2]) == len(new_out[1])


def test_make_detector_routing():
    cfg_box = DetectionConfig(detector="box")
    cfg_rk = DetectionConfig(detector="ricker")
    assert isinstance(make_detector(cfg_box, 6), BoxDipDetector)
    assert isinstance(make_detector(cfg_rk, 6), RickerDipDetector)
