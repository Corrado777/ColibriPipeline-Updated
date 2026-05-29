"""
Detection algorithms for the offline occultation-detection trade study.

All detectors share the DetectionResult / Detector contract and return a
full-length per-frame score array so that results can be combined across
telescopes.

Reference implementations:
  colibri_photometry.py  dipDetection()          -- RickerDetector baseline
  colibri_secondary.py   ~line 330               -- Poisson sigma (Pass 2018)
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import convolve, correlate
from astropy.convolution import RickerWavelet1DKernel


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Result of running a single detector on a single light curve.

    Attributes
    ----------
    score : ndarray
        Per-frame detection statistic, SAME length as input flux.
        Edges where the statistic is undefined are padded with np.nan.
    peak_frame : int
        Index of the maximum score (NaN-safe argmax).
    peak_score : float
        Maximum score value (NaN-safe).
    """
    score: np.ndarray
    peak_frame: int
    peak_score: float


class Detector:
    """Abstract base class for all detectors.

    Subclasses must set ``name`` (str) and implement ``run``.
    """
    name: str = "base"

    def run(self, flux, exposure_time=0.025) -> DetectionResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def detrend(flux, window_frames=200):
    """Boxcar detrend: subtract a running mean from the flux.

    Mirrors colibri_photometry.py line 495-497:
        window_size = 40*5  (5-second window = 200 frames at 40 Hz)
        smoothed = uniform_filter1d(lc, size=200, mode='reflect')
        detrended = lc - smoothed

    Parameters
    ----------
    flux : array_like, 1-D
    window_frames : int
        Width of the boxcar (default 200 = 5 s at 40 Hz).

    Returns
    -------
    ndarray
        Detrended flux of the same length.
    """
    flux = np.asarray(flux, dtype=float)
    smoothed = uniform_filter1d(flux, size=window_frames, mode='reflect')
    return flux - smoothed


def poisson_sigma(flux):
    """Per-frame normalised Poisson uncertainty (Pass 2018, Eq. 7).

    Mirrors colibri_secondary.py lines 326-331:
        Gain = 0.82  (high-gain setting)
        sigmaP = sqrt(|f| / median(f) / Gain) * Gain
        replace zeros with 0.01

    Parameters
    ----------
    flux : array_like, 1-D

    Returns
    -------
    ndarray
        Same length as flux, all values > 0.
    """
    flux = np.asarray(flux, dtype=float)
    Gain = 0.82
    med = np.median(flux)
    if med == 0:
        med = 1.0  # guard against all-zero input
    sigma = np.sqrt(np.abs(flux) / np.abs(med) / Gain) * Gain
    sigma = np.where(sigma == 0, 0.01, sigma)
    return sigma


def _bkg_significance(conv, exclusion_zone=5):
    """Convert a convolution/statistic trace to per-frame significance.

    Mirrors colibri_photometry.py lines 510-523:
        background zone excludes a window of 2*exclusion_zone+1 frames around
        the peak (argmin for dip-style statistics);
        significance = (bkg_mean - value) / bkg_std  (dips → positive).

    The returned array is (bkg_mean - conv) / bkg_std, making dips positive.
    bkg_mean and bkg_std are computed from the exclusion zone around argmin(conv).

    Parameters
    ----------
    conv : ndarray, 1-D
        Raw convolution or statistic trace.
    exclusion_zone : int
        Half-width of the exclusion window in frames (default 5).

    Returns
    -------
    ndarray
        Per-frame significance, same length as conv.
    """
    min_loc = int(np.argmin(conv))
    start_ex = max(0, min_loc - exclusion_zone)
    end_ex = min(len(conv), min_loc + exclusion_zone + 1)
    bkg = np.concatenate((conv[:start_ex], conv[end_ex:]))
    if len(bkg) < 2:
        bkg = conv  # fallback: use whole trace
    bkg_mean = float(np.mean(bkg))
    bkg_std = float(np.std(bkg))
    if bkg_std == 0:
        bkg_std = 1.0
    return (bkg_mean - conv) / bkg_std


def _make_result(score_full):
    """Build a DetectionResult from a full-length score array."""
    if np.all(np.isnan(score_full)):
        return DetectionResult(score=score_full, peak_frame=0, peak_score=float('nan'))
    peak_frame = int(np.nanargmax(score_full))
    peak_score = float(np.nanmax(score_full))
    return DetectionResult(score=score_full, peak_frame=peak_frame, peak_score=peak_score)


# ---------------------------------------------------------------------------
# Detector 1: RickerDetector  (BASELINE — mirrors production pipeline)
# ---------------------------------------------------------------------------

class RickerDetector(Detector):
    """BASELINE: detrend, convolve with Ricker wavelet, compute significance.

    Reproduces the production path in colibri_photometry.dipDetection
    (lines 494-544).  Uses scipy.signal.convolve(mode='same') so the output
    length equals the input length (production uses mode='valid' and then
    pads manually; 'same' is equivalent for the full-length score contract).
    """

    name = "RickerDetector"

    def __init__(self, width_frames=6):
        self.width_frames = width_frames
        self._kernel = None

    def _get_kernel(self):
        if self._kernel is None:
            self._kernel = np.array(
                RickerWavelet1DKernel(self.width_frames).array)
        return self._kernel

    def run(self, flux, exposure_time=0.025):
        flux = np.asarray(flux, dtype=float)
        n = len(flux)
        det = detrend(flux)
        k = self._get_kernel()
        conv = convolve(det, k, mode='same')
        score = _bkg_significance(conv)
        return _make_result(score)


# ---------------------------------------------------------------------------
# Detector 2: BoxDetector
# ---------------------------------------------------------------------------

class BoxDetector(Detector):
    """Geometric box detector: running mean over box_frames, then significance.

    Detrends, computes a running mean (uniform_filter1d with box_frames),
    converts via the background-significance helper.  Detects flat-bottomed
    depth without wavelet shape-matching.
    """

    name = "BoxDetector"

    def __init__(self, box_frames=6):
        self.box_frames = box_frames

    def run(self, flux, exposure_time=0.025):
        flux = np.asarray(flux, dtype=float)
        det = detrend(flux)
        running = uniform_filter1d(det, size=self.box_frames, mode='reflect')
        score = _bkg_significance(running)
        return _make_result(score)


# ---------------------------------------------------------------------------
# Detector 3: MultiWidthRickerDetector
# ---------------------------------------------------------------------------

class MultiWidthRickerDetector(Detector):
    """Bank of Ricker kernels; per-frame score = max over widths.

    Runs RickerDetector for each width in ``widths`` and takes the per-frame
    maximum significance.  Robust to uncertainty in the true dip duration.
    """

    name = "MultiWidthRickerDetector"

    def __init__(self, widths=None):
        if widths is None:
            widths = range(1, 13)
        self.widths = list(widths)
        self._detectors = [RickerDetector(w) for w in self.widths]

    def run(self, flux, exposure_time=0.025):
        flux = np.asarray(flux, dtype=float)
        scores = np.stack(
            [d.run(flux, exposure_time).score for d in self._detectors],
            axis=0
        )
        # NaN-safe max over width axis
        with np.errstate(all='ignore'):
            score = np.nanmax(scores, axis=0)
        return _make_result(score)


# ---------------------------------------------------------------------------
# Detector 4: FresnelMatchedFilterDetector
# ---------------------------------------------------------------------------

_DEFAULT_BANK_PATH = (
    Path(__file__).parents[2]
    / "KernelGeneratorGUI_RAB032922"
    / "kernels_40hz_20230206.txt"
)

# Subsample stride: the bank has 2769 kernels.  Using every 10th gives 277
# templates which is sufficient for an offline trade study while keeping
# runtime manageable.  Increase BANK_STRIDE to 1 for production-quality
# exhaustive matching.
_BANK_STRIDE = 10


def _load_kernel_bank(bank_path, stride=_BANK_STRIDE):
    """Load and subsample the Fresnel kernel bank.

    Each row of the text file is one kernel (whitespace-delimited floats,
    values near 1.0 representing relative flux during a diffraction event).
    Returns an ndarray of shape (n_templates, kernel_len).
    """
    kernels = []
    with open(bank_path, 'r') as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            if i % stride == 0:
                kernels.append(np.fromstring(line, sep=' '))
    # Pad / truncate to a common length (all rows should be equal length)
    lengths = [len(k) for k in kernels]
    if len(set(lengths)) > 1:
        min_len = min(lengths)
        kernels = [k[:min_len] for k in kernels]
    return np.array(kernels, dtype=float)


class FresnelMatchedFilterDetector(Detector):
    """Fresnel diffraction matched-filter bank detector.

    Loads the pre-computed 40 Hz Fresnel kernel bank, converts each template
    to a zero-mean matched filter, cross-correlates with the detrended flux,
    normalises by each template's norm, then takes the per-frame maximum
    response over the bank and converts to significance.

    The bank has 2769 kernels; every BANK_STRIDE-th (default 10th) is used,
    giving ~277 templates.  Set bank_path=None to use the project default.

    Notes
    -----
    Subsampling stride is _BANK_STRIDE = 10, documented above.  Set stride=1
    in __init__ for exhaustive matching at the cost of ~10x runtime.
    """

    name = "FresnelMatchedFilterDetector"

    def __init__(self, bank_path=None, stride=_BANK_STRIDE):
        if bank_path is None:
            bank_path = _DEFAULT_BANK_PATH
        self._bank_path = Path(bank_path)
        self._stride = stride
        self._bank = None  # loaded lazily

    def _get_bank(self):
        if self._bank is None:
            self._bank = _load_kernel_bank(self._bank_path, stride=self._stride)
        return self._bank

    def run(self, flux, exposure_time=0.025):
        flux = np.asarray(flux, dtype=float)
        n = len(flux)
        det = detrend(flux)
        bank = self._get_bank()

        # Build zero-mean, unit-norm matched filters from the bank templates.
        # The templates are relative-flux curves (~1.0 baseline, dip < 1.0).
        # A dip manifests as a template mean departure below 1 — subtract mean
        # to make the filter zero-mean, then normalise to unit L2 norm.
        filters = bank - bank.mean(axis=1, keepdims=True)  # zero-mean
        norms = np.linalg.norm(filters, axis=1)
        # Avoid division by zero for flat (no-dip) templates
        norms = np.where(norms == 0, 1.0, norms)
        filters = filters / norms[:, np.newaxis]  # unit norm

        # Cross-correlate each filter with the detrended flux.
        # scipy.signal.correlate(mode='same') preserves length.
        # A real dip (negative in det) cross-correlated with a negative-going
        # filter (dip below zero) gives a positive response.
        responses = np.zeros((len(filters), n), dtype=float)
        for i, filt in enumerate(filters):
            responses[i] = correlate(det, filt, mode='same')

        # Per-frame max over the bank
        raw_score = np.max(responses, axis=0)

        # Convert to significance using background exclusion zone helper.
        # raw_score is largest where the dip is → positive peak; the helper
        # is designed for conv (dip = negative), so we negate first, compute
        # significance (which inverts sign back), yielding positive at dip.
        score = _bkg_significance(-raw_score)
        return _make_result(score)


# ---------------------------------------------------------------------------
# Detector 5: NormalizedSNRDetector
# ---------------------------------------------------------------------------

class NormalizedSNRDetector(Detector):
    """Pass-style per-frame z-score detector.

    Divides the flux by a smooth baseline (uniform_filter1d with 200-frame
    window), subtracts 1.0, divides by poisson_sigma to get a per-frame
    z-score series.  score = -zscore so that dips (negative z-scores) produce
    positive scores.  This is the natural input for multi-telescope summing.
    """

    name = "NormalizedSNRDetector"

    def run(self, flux, exposure_time=0.025):
        flux = np.asarray(flux, dtype=float)
        # Smooth baseline (same window as detrend helper)
        baseline = uniform_filter1d(flux, size=200, mode='reflect')
        # Guard against zero baseline
        safe_baseline = np.where(np.abs(baseline) < 1e-10, 1e-10, baseline)
        normalised = flux / safe_baseline - 1.0
        sigma = poisson_sigma(flux)
        zscore = normalised / sigma
        score = -zscore  # dips (negative z) become positive scores
        return _make_result(score)


# ---------------------------------------------------------------------------
# Convenience accessor
# ---------------------------------------------------------------------------

def ALL_DETECTORS():
    """Return a dict of {name: instance} for all five default-configured detectors."""
    detectors = [
        RickerDetector(width_frames=6),
        BoxDetector(box_frames=6),
        MultiWidthRickerDetector(widths=range(1, 13)),
        FresnelMatchedFilterDetector(),
        NormalizedSNRDetector(),
    ]
    return {d.name: d for d in detectors}
