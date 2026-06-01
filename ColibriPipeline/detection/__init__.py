"""Colibri dip-detection module.

Canonical home of occultation dip detection, extracted from
``colibri_photometry.dipDetection`` / ``colibri_main_py3``. Defaults to a flat
box matched filter (the detection-trade-study winner) at a single canonical
width derived from the field's opposition angle; the Ricker-wavelet detector is
preserved as a selectable option.

Public API
----------
``DetectionConfig``        -- configuration dataclass (detector choice, sigma,
                              preprocessing window, canonical-width source,
                              coincidence mode + tolerances).
``make_detector(config)``  -- factory returning a ``DipDetector`` whose
                              ``detect(flux_1d)`` reproduces the legacy 9-tuple.
``canonical_width_frames`` -- opposition-angle -> box width (frames).
``post_threshold_match`` / ``joint_statistic_detect`` / ``active_telescopes``
                           -- multi-telescope coincidence entry points.

Depends only on numpy/scipy/astropy. Does NOT import ``detection_trade_study``
(that package keeps its own copy); the minimal winning pieces are ported here.
"""

from .config import DetectionConfig
from .detector import make_detector, DipDetector, BoxDipDetector, RickerDipDetector, box_template
from .width import canonical_width_frames

__all__ = [
    "DetectionConfig",
    "make_detector",
    "DipDetector",
    "BoxDipDetector",
    "RickerDipDetector",
    "box_template",
    "canonical_width_frames",
]
