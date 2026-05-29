"""
Detection algorithms for the offline occultation-detection trade study.

A detector is now factored into two swappable pieces:

  * a PREPROCESSOR (preprocessing.py) that conditions the raw flux into a flat
    residual + a noise model (scalar / per-frame / PSD), and
  * a TEMPLATE SHAPE (or bank of shapes) matched-filtered against that residual
    via the shared optimal core (matched_filter.py).

This lets the harness trade-study the full grid of {preprocessing} x {shape}
(`build_grid`), while `ALL_DETECTORS()` keeps the five round-1 shapes wired to
the production-style `MeanSubtract` preprocessing for a stable baseline.

All detectors return the same `DetectionResult` contract (full-length per-frame
`score`, dips positive), so `combine.py` and `harness.py` are unchanged.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.convolution import RickerWavelet1DKernel

from .preprocessing import MeanSubtract, ALL_PREPROCESSORS
from . import matched_filter as mf


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Result of running a single detector on a single light curve.

    score : per-frame detection statistic, SAME length as input flux (dips
            positive; NaN where undefined).
    peak_frame : NaN-safe argmax of score.
    peak_score : NaN-safe max of score.
    """
    score: np.ndarray
    peak_frame: int
    peak_score: float


class Detector:
    """Abstract base: set ``name`` and implement ``run``."""
    name: str = "base"

    def run(self, flux, exposure_time=0.025) -> DetectionResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Dip-template factories (dip = NEGATIVE; matched_filter re-zero-means anyway)
# ---------------------------------------------------------------------------

def box_template(width):
    """Flat box dip of the given width (frames)."""
    return -np.ones(int(width), dtype=float)


def ricker_template(width):
    """Ricker (Mexican-hat) wavelet, negated so the central lobe is a dip.

    Mirrors the production kernel (astropy RickerWavelet1DKernel) used in
    colibri_main_py3 / colibri_photometry.dipDetection.
    """
    k = np.asarray(RickerWavelet1DKernel(width).array, dtype=float)
    return -k


def delta_template():
    """Single-frame dip -- a per-frame deviation ('Pass-style' SNR detector)."""
    return np.array([-1.0])


_DEFAULT_BANK_PATH = (
    Path(__file__).parents[2] / "KernelGeneratorGUI_RAB032922"
    / "kernels_40hz_20230206.txt"
)
_BANK_STRIDE = 10  # 2769 kernels -> ~277 templates; raise for speed


def load_fresnel_templates(bank_path=None, stride=_BANK_STRIDE):
    """Load the 40 Hz Fresnel kernel bank as dip templates (baseline removed).

    Each bank row is a relative-flux curve (~1.0 baseline with a diffraction
    dip).  Subtracting 1.0 yields a dip-negative template; matched_filter
    zero-means and unit-normalises internally.
    """
    if bank_path is None:
        bank_path = _DEFAULT_BANK_PATH
    rows = []
    with open(bank_path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if line and i % stride == 0:
                rows.append(np.fromstring(line, sep=' '))
    if rows and len({len(r) for r in rows}) > 1:
        m = min(len(r) for r in rows)
        rows = [r[:m] for r in rows]
    return [r - 1.0 for r in rows]


# ---------------------------------------------------------------------------
# Unified matched-filter detector
# ---------------------------------------------------------------------------

class MatchedFilterDetector(Detector):
    """Preprocess, then take the per-frame max matched-filter response over a
    bank of dip templates.

    A single-template bank is an ordinary matched filter; a multi-template bank
    (multi-width Ricker, Fresnel) takes the best-matching shape per frame.
    """

    def __init__(self, name, templates, preprocessor=None):
        self.name = name
        self.templates = list(templates)
        self.preprocessor = preprocessor if preprocessor is not None else MeanSubtract()

    def run(self, flux, exposure_time=0.025):
        cond = self.preprocessor.apply(flux, exposure_time)
        scores = [mf.matched_filter(cond.series, t, cond.noise) for t in self.templates]
        score = scores[0] if len(scores) == 1 else np.fmax.reduce(scores)
        pf, ps = mf.peak(score)
        return DetectionResult(score, pf, ps)


# ---------------------------------------------------------------------------
# Named shapes + accessors
# ---------------------------------------------------------------------------

def _shape_templates():
    """Map of shape name -> template bank (built once)."""
    return {
        'RickerDetector': [ricker_template(6)],
        'BoxDetector': [box_template(6)],
        'MultiWidthRickerDetector': [ricker_template(w) for w in range(1, 13)],
        'FresnelMatchedFilterDetector': load_fresnel_templates(),
        'NormalizedSNRDetector': [delta_template()],
    }


def ALL_DETECTORS(preprocessor=None):
    """The five shapes wired to `preprocessor` (default MeanSubtract baseline)."""
    return {name: MatchedFilterDetector(name, tmpls, preprocessor)
            for name, tmpls in _shape_templates().items()}


def build_grid(shape_names=None, preprocessor_names=None, fresnel_stride=None):
    """Build the {shape} x {preprocessing} trade-study grid.

    Returns dict {f'{shape}@{prep}': MatchedFilterDetector}.  The '@' separator
    keeps detector names free of '|' so the harness column parsing is unaffected.

    `fresnel_stride` (if given) overrides the Fresnel bank subsampling -- raise it
    to keep the grid affordable, since the Fresnel bank dominates runtime.
    """
    shapes = _shape_templates()
    if fresnel_stride is not None:
        shapes['FresnelMatchedFilterDetector'] = load_fresnel_templates(stride=fresnel_stride)
    if shape_names is not None:
        shapes = {k: shapes[k] for k in shape_names}
    preps = ALL_PREPROCESSORS()
    if preprocessor_names is not None:
        preps = {k: preps[k] for k in preprocessor_names}

    grid = {}
    for sname, tmpls in shapes.items():
        for pname, prep in preps.items():
            grid[f'{sname}@{pname}'] = MatchedFilterDetector(f'{sname}@{pname}', tmpls, prep)
    return grid
