"""
Preprocessing methods for the offline occultation-detection trade study.

Each Preprocessor produces a Conditioned object whose ``series`` is a
flat-baseline residual ready for matched filtering and whose ``noise`` is one
of the three tagged noise-model tuples consumed by matched_filter.matched_filter:

    ('scalar',   float_std)
    ('perframe', ndarray_sigma_same_length_as_series)
    ('psd',      onesided_power_ndarray, fs_float)

The PSD variant uses the one-sided rfft convention (length = N//2+1 for an
N-frame series) and carries the sampling frequency fs so that the matched
filter can construct the corresponding rfft frequency grid.
"""

from dataclasses import dataclass

import numpy as np
import scipy.ndimage
import scipy.signal


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

@dataclass
class Conditioned:
    """Output of any Preprocessor.

    Attributes
    ----------
    series : ndarray
        1-D residual of the same length as the input flux.  The baseline has
        been removed; the array is ready for matched filtering.
    noise : tuple
        Tagged noise model.  One of:
            ('scalar',   float_std)
            ('perframe', ndarray_sigma)
            ('psd',      onesided_power_ndarray, fs_float)
    """
    series: np.ndarray
    noise: tuple


class Preprocessor:
    """Abstract base class for all preprocessors.

    Subclasses must set ``name`` (str) and implement ``apply``.
    """
    name: str = "base"

    def apply(self, flux, exposure_time=0.025) -> Conditioned:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _mad_std(x):
    """Robust standard-deviation estimate via median absolute deviation.

    Returns 1.4826 * MAD(x), which equals std for Gaussian noise.
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad


# ---------------------------------------------------------------------------
# Preprocessor 1: MeanSubtract  (mirrors current pipeline baseline)
# ---------------------------------------------------------------------------

class MeanSubtract(Preprocessor):
    """Boxcar running-mean subtraction (current-pipeline baseline).

    Mirrors colibri_photometry.py lines 495-497:
        smoothed = uniform_filter1d(flux, size=200, mode='reflect')
        series   = flux - smoothed

    Parameters
    ----------
    window : int
        Width of the boxcar in frames (default 200 = 5 s at 40 Hz).

    Noise model
    -----------
    ('scalar', np.std(series))
    """

    name = "MeanSubtract"

    def __init__(self, window=200):
        self.window = window

    def apply(self, flux, exposure_time=0.025) -> Conditioned:
        flux = np.asarray(flux, dtype=float)
        smoothed = scipy.ndimage.uniform_filter1d(flux, size=self.window, mode='reflect')
        series = flux - smoothed
        noise = ('scalar', float(np.std(series)))
        return Conditioned(series=series, noise=noise)


# ---------------------------------------------------------------------------
# Preprocessor 2: MedianDivide  (robust fractional normalisation)
# ---------------------------------------------------------------------------

class MedianDivide(Preprocessor):
    """Robust fractional normalisation via running median.

    Baseline is computed with a running median; the residual is
    ``flux / baseline - 1``.  Robust to the dip itself and to outliers.
    Noise is estimated with the scaled MAD so it is also outlier-robust.

    Parameters
    ----------
    window : int
        Width of the median filter in frames (default 200 = 5 s at 40 Hz).

    Noise model
    -----------
    ('scalar', 1.4826 * MAD(series))
    """

    name = "MedianDivide"

    def __init__(self, window=200):
        self.window = window

    def apply(self, flux, exposure_time=0.025) -> Conditioned:
        flux = np.asarray(flux, dtype=float)
        baseline = scipy.ndimage.median_filter(flux, size=self.window, mode='reflect')
        # Guard against near-zero baseline
        safe_baseline = np.where(np.abs(baseline) < 1e-10, 1e-10, baseline)
        series = flux / safe_baseline - 1.0
        noise = ('scalar', float(_mad_std(series)))
        return Conditioned(series=series, noise=noise)


# ---------------------------------------------------------------------------
# Preprocessor 3: HighPass  (tunable short-window mean subtract)
# ---------------------------------------------------------------------------

class HighPass(Preprocessor):
    """Running-mean subtraction with a short, tunable window.

    Identical to MeanSubtract but the default window is 1 s (40 frames) to
    pass higher-frequency variations.

    Parameters
    ----------
    window : int
        Width of the boxcar in frames (default 40 = 1 s at 40 Hz).

    Noise model
    -----------
    ('scalar', np.std(series))
    """

    name = "HighPass"

    def __init__(self, window=40):
        self.window = window

    def apply(self, flux, exposure_time=0.025) -> Conditioned:
        flux = np.asarray(flux, dtype=float)
        smoothed = scipy.ndimage.uniform_filter1d(flux, size=self.window, mode='reflect')
        series = flux - smoothed
        noise = ('scalar', float(np.std(series)))
        return Conditioned(series=series, noise=noise)


# ---------------------------------------------------------------------------
# Preprocessor 4: Bandpass  (zero-phase Butterworth)
# ---------------------------------------------------------------------------

class Bandpass(Preprocessor):
    """Zero-phase Butterworth bandpass filter.

    Uses ``scipy.signal.butter`` with ``output='sos'`` and
    ``scipy.signal.sosfiltfilt`` for zero-phase filtering.  The upper cutoff
    is clipped to just below the Nyquist frequency if necessary.

    Parameters
    ----------
    lo_hz : float
        Lower passband edge in Hz (default 1.0 Hz).
    hi_hz : float
        Upper passband edge in Hz (default 15.0 Hz).
    order : int
        Butterworth filter order (default 4).

    Noise model
    -----------
    ('scalar', np.std(series))
    """

    name = "Bandpass"

    def __init__(self, lo_hz=1.0, hi_hz=15.0, order=4):
        self.lo_hz = lo_hz
        self.hi_hz = hi_hz
        self.order = order

    def apply(self, flux, exposure_time=0.025) -> Conditioned:
        flux = np.asarray(flux, dtype=float)
        fs = 1.0 / exposure_time          # sampling frequency (Hz)
        nyquist = fs / 2.0
        # Clip hi_hz strictly below Nyquist to keep the filter stable
        hi = min(self.hi_hz, nyquist * 0.999)
        lo = self.lo_hz
        sos = scipy.signal.butter(
            self.order, [lo, hi], btype='band', fs=fs, output='sos'
        )
        series = scipy.signal.sosfiltfilt(sos, flux)
        noise = ('scalar', float(np.std(series)))
        return Conditioned(series=series, noise=noise)


# ---------------------------------------------------------------------------
# Preprocessor 5: Whiten  (PSD-based optimal-filter noise model)
# ---------------------------------------------------------------------------

class Whiten(Preprocessor):
    """Detrend then estimate the one-sided noise PSD for optimal whitening.

    Steps:
    1. Subtract a 200-frame running mean to remove the slow baseline.
    2. Estimate the PSD with Welch's method (``scipy.signal.welch``).
    3. Interpolate the Welch PSD onto the full rfft frequency grid so it can
       be used directly by ``matched_filter.matched_filter``.
    4. Floor the PSD at a small positive value to avoid divide-by-zero.

    Parameters
    ----------
    seg_frames : int
        Welch segment length in frames (default 256).

    Noise model
    -----------
    ('psd', onesided_power_ndarray, fs_float)

    The ``onesided_power_ndarray`` has length ``len(series) // 2 + 1`` and
    corresponds to the rfft frequency bins ``np.fft.rfftfreq(len(series),
    d=exposure_time)``.
    """

    name = "Whiten"

    def __init__(self, seg_frames=256):
        self.seg_frames = seg_frames

    def apply(self, flux, exposure_time=0.025) -> Conditioned:
        flux = np.asarray(flux, dtype=float)
        fs = 1.0 / exposure_time

        # Step 1: detrend with a 200-frame boxcar mean
        smoothed = scipy.ndimage.uniform_filter1d(flux, size=200, mode='reflect')
        residual = flux - smoothed

        n = len(residual)
        nperseg = min(self.seg_frames, n)

        # Step 2: Welch PSD estimate on the Welch frequency grid
        f_welch, Pxx_welch = scipy.signal.welch(
            residual,
            fs=fs,
            nperseg=nperseg,
            return_onesided=True,
        )

        # Step 3: interpolate onto the full rfft frequency grid
        f_rfft = np.fft.rfftfreq(n, d=exposure_time)   # length n//2+1
        psd = np.interp(f_rfft, f_welch, Pxx_welch)

        # Step 4: floor to avoid divide-by-zero
        psd_floor = float(np.max(psd)) * 1e-12
        psd = np.maximum(psd, psd_floor)

        noise = ('psd', psd, float(fs))
        return Conditioned(series=residual, noise=noise)


# ---------------------------------------------------------------------------
# Convenience accessor
# ---------------------------------------------------------------------------

def ALL_PREPROCESSORS():
    """Return a dict of {name: instance} for the five default preprocessors."""
    preps = [
        MeanSubtract(window=200),
        MedianDivide(window=200),
        HighPass(window=40),
        Bandpass(lo_hz=1.0, hi_hz=15.0, order=4),
        Whiten(seg_frames=256),
    ]
    return {p.name: p for p in preps}
