"""Dip detectors (interface contract + implementations).

This file fixes the contract that ``colibri_main_py3`` depends on.

THE 9-TUPLE CONTRACT (must match colibri_photometry.dipDetection exactly)
------------------------------------------------------------------------
``detect(flux_1d)`` returns, in order::

    (frameNum,        # int: event frame in the *trimmed* light curve;
                      #      -1 = no detection, -2 = unusable data
     lc_arr,          # np.ndarray: trimmed light curve (np.trim_zeros); [] if -2
     conv_padded,     # np.ndarray: detection statistic, SAME LENGTH as lc_arr
                      #      (padded with bkg mean) so main can slice it by frame
     lightcurve_std,  # float
     lightcurve_mean, # float
     bkg_std,         # float: std of the statistic's background zone
     bkg_mean,        # float: mean of the statistic's background zone
     minVal,          # float: extremum of the statistic at the dip
     significance)    # float: (bkg_mean - minVal) / bkg_std  [or box equivalent]

Rejection returns mirror the legacy code:
``-2, [], [], nan, nan, nan, nan, -2, nan`` (empty/short) and on no-detection
``-1, lc_arr, conv_padded, nan, nan, nan, nan, nan, significance``.

Preprocessing is mean-subtract (see preprocessing.mean_subtract); both detectors
run it before computing their statistic. ``frameNum`` must be expressed in the
trimmed light curve's index frame, accounting for the detector's lag (the box
correlate uses 'same'; legacy Ricker uses ``minLoc + len(kernel.array)//2``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.signal
from astropy.convolution import RickerWavelet1DKernel

from .config import DetectionConfig, EXCLUSION_ZONE
from .preprocessing import mean_subtract


MIN_LIGHTCURVE_LEN = 100  # legacy minLightcurveLen


def box_template(width: int) -> np.ndarray:
    """Flat box dip template of ``width`` frames (negative; unit step)."""
    return -np.ones(int(width), dtype=float)


class DipDetector(ABC):
    """Common detector interface."""

    @abstractmethod
    def detect(self, flux_1d):
        """Run detection on one star's flux profile. Returns the 9-tuple above."""
        raise NotImplementedError


def _reject_tuple():
    """The legacy empty/short rejection 9-tuple."""
    return -2, [], [], np.nan, np.nan, np.nan, np.nan, -2, np.nan


class BoxDipDetector(DipDetector):
    """Flat box matched filter (scalar-noise path). Default detector.

    Ports the scalar matched filter from
    ``detection_trade_study/matched_filter.py`` (unit-L2-normalised template,
    ``scipy.signal.correlate(..., mode='same')``, ``score = raw / sigma``) onto
    the mean-subtract residual, then emits the legacy 9-tuple.

    Significance definition
    -----------------------
    A real negative dip in the light curve aligned with the negative box
    template produces a **positive** peak in the matched-filter score. So,
    unlike the legacy Ricker path (which looks at the convolution *minimum*),
    the box detector looks at the score **maximum**. To keep the resulting
    number directly comparable to ``sigma_threshold``, significance is defined
    symmetrically to the legacy code with the sign flipped to follow the peak::

        significance = (peak_score - bkg_mean) / bkg_std

    where ``peak_score`` is the maximum of the score array and the background
    mean/std are computed over the score array EXCLUDING a +/-EXCLUSION_ZONE
    window around the peak. Because the template is unit-L2-normalised and the
    residual is divided by its scalar std, ``score`` is already in
    sigma-like units, so ``bkg_mean ~ 0`` and ``bkg_std ~ 1``; the expression
    above is therefore the peak height in sigma -- the box analogue of the
    Ricker significance and directly comparable to ``sigma_threshold``.

    The ``minVal`` field of the 9-tuple carries the extremum used here, i.e. the
    peak score (the maximum), preserving the slot's "statistic at the dip"
    meaning while reflecting the box's positive-peak convention.
    """

    def __init__(self, width: int, config: DetectionConfig):
        self.width = int(width)
        self.config = config

    def detect(self, flux_1d):
        sigma_threshold = self.config.sigma_threshold

        light_curve = np.trim_zeros(np.asarray(flux_1d, dtype=float))

        if len(light_curve) == 0:
            return _reject_tuple()
        if len(light_curve) < MIN_LIGHTCURVE_LEN:
            return _reject_tuple()

        # Detrend (mean-subtract) -> residual + scalar noise.
        residual, sigma = mean_subtract(light_curve, window=self.config.preproc_window)
        if sigma == 0.0:
            sigma = 1.0  # guard, mirrors matched_filter.py

        # Unit-L2-normalised box template.
        template = box_template(self.width)
        t_norm = np.linalg.norm(template)
        tmpl_unit = template / t_norm

        # Matched filter; 'same' -> score length == light-curve length.
        raw = scipy.signal.correlate(residual, tmpl_unit, mode="same")
        score = raw / sigma  # this IS conv_padded (already same length)

        # A negative dip => positive score peak.
        peakLoc = int(np.argmax(score))
        peakVal = float(score[peakLoc])

        # Background zone: score excluding +/-EXCLUSION_ZONE around the peak.
        start_exclusion = max(0, peakLoc - EXCLUSION_ZONE)
        end_exclusion = min(len(score), peakLoc + EXCLUSION_ZONE + 1)
        bkgZone = np.concatenate((score[:start_exclusion], score[end_exclusion:]))

        lightcurve_std = float(np.std(light_curve))
        bkg_mean = float(np.mean(bkgZone))
        bkg_std = float(np.std(bkgZone))
        if bkg_std == 0.0:
            bkg_std = 1.0  # guard against degenerate (noise-free) inputs

        significance = (peakVal - bkg_mean) / bkg_std

        if significance >= sigma_threshold:
            return (
                peakLoc,
                light_curve,
                score,
                lightcurve_std,
                float(np.mean(light_curve)),
                bkg_std,
                bkg_mean,
                peakVal,
                significance,
            )
        return (
            -1,
            light_curve,
            score,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            significance,
        )


class RickerDipDetector(DipDetector):
    """Legacy Ricker-wavelet detector preserved as a selectable option.

    Wraps the existing ``colibri_photometry.dipDetection`` math (astropy
    ``RickerWavelet1DKernel`` + ``scipy.signal.convolve``) so behaviour is
    byte-for-byte identical to the legacy path. When ``self.kernel`` is None a
    ``RickerWavelet1DKernel(6)`` is built (matching ``colibri_main_py3``).
    """

    def __init__(self, config: DetectionConfig, kernel=None):
        self.config = config
        self.kernel = kernel

    def detect(self, flux_1d):
        sigma_threshold = self.config.sigma_threshold

        kernel = self.kernel
        if kernel is None:
            kernel = RickerWavelet1DKernel(6)

        # --- verbatim port of colibri_photometry.dipDetection -----------------
        light_curve = np.trim_zeros(np.asarray(flux_1d, dtype=float))

        if len(light_curve) == 0:
            return _reject_tuple()

        minLightcurveLen = MIN_LIGHTCURVE_LEN
        if len(light_curve) < minLightcurveLen:
            return _reject_tuple()

        # Detrend the light curve for better dip detection (legacy hard-codes
        # window_size = 40*5 = 200; preproc_window defaults to that).
        detrended_light_curve, _ = mean_subtract(
            light_curve, window=self.config.preproc_window
        )

        conv = scipy.signal.convolve(detrended_light_curve, kernel, mode="valid")
        minLoc = int(np.argmin(conv))
        minVal = float(np.min(conv))

        exclusion_zone = EXCLUSION_ZONE
        start_exclusion = max(0, minLoc - exclusion_zone)
        end_exclusion = min(len(conv), minLoc + exclusion_zone + 1)
        bkgZone = np.concatenate((conv[:start_exclusion], conv[end_exclusion:]))

        lightcurve_std = float(np.std(light_curve))
        conv_bkg_mean = float(np.mean(bkgZone))
        significance = (conv_bkg_mean - minVal) / np.std(bkgZone)

        # Pad the convolution out to the light-curve length with bkg mean.
        padding_length = (len(light_curve) - len(conv)) // 2
        padding = np.ones(padding_length) * conv_bkg_mean
        conv_padded = np.concatenate((padding, conv, padding))
        if len(conv_padded) < len(light_curve):
            conv_padded = np.concatenate((conv_padded, [0]))

        if significance >= sigma_threshold:
            critFrame = minLoc + (len(kernel.array) // 2)
            return (
                critFrame,
                light_curve,
                conv_padded,
                lightcurve_std,
                float(np.mean(light_curve)),
                float(np.std(bkgZone)),
                conv_bkg_mean,
                minVal,
                significance,
            )
        return (
            -1,
            light_curve,
            conv_padded,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            significance,
        )


def make_detector(config: DetectionConfig, width: int) -> DipDetector:
    """Build the configured detector.

    Parameters
    ----------
    config : DetectionConfig
    width : int
        Canonical width in frames (from ``canonical_width_frames``); ignored by
        the Ricker detector, which uses its own scale.
    """
    if config.detector == "box":
        return BoxDipDetector(width, config)
    return RickerDipDetector(config)
