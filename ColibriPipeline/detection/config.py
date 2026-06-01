"""Detection configuration.

Single source of truth for the knobs the detection module exposes. Mirrors the
defaults currently hard-coded in ``colibri_main_py3`` so the box detector is a
drop-in replacement for the Ricker path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# --- legacy defaults pulled from colibri_main_py3 / colibri_photometry -------
DEFAULT_EXPOSURE_TIME = 0.025          # s/frame (40 Hz)
DEFAULT_SIGMA_THRESHOLD = 5.0          # -s/--sigma in colibri_main_py3
DEFAULT_PREPROC_WINDOW = 40 * 5        # 200 frames (5 s) boxcar detrend
DEFAULT_CANONICAL_WIDTH_AT_OPPOSITION = 6   # frames (0.15 s) -- trade-study winner
EXCLUSION_ZONE = 5                     # frames excluded around the dip for bkg stats

# coincidence tolerances (match simultaneous_occults TIME_TOLERANCE / COORD_TOLERANCE)
DEFAULT_TIME_TOLERANCE_S = 0.2         # s
DEFAULT_COORD_TOLERANCE_DEG = 0.002    # deg (~7 arcsec)


@dataclass
class DetectionConfig:
    """Configuration for a detection run.

    Attributes
    ----------
    detector : {'box', 'ricker'}
        Which detector ``make_detector`` builds. 'box' (default) is the canonical
        flat matched filter; 'ricker' preserves the legacy astropy path.
    sigma_threshold : float
        Significance threshold for a detection.
    preproc_window : int
        Boxcar window (frames) for mean-subtract systematics removal.
    exposure_time : float
        Seconds per frame; used to convert the canonical width / event durations.
    canonical_width : Optional[int]
        Explicit box width in frames. If set, overrides opposition-angle auto-compute
        (the ``--canonical-width`` safety valve). ``None`` -> auto-compute.
    opposition_angle_deg : Optional[float]
        Explicit opposition angle (deg). If set, used instead of computing it from
        field pointing + date (the ``--opposition-angle`` safety valve).
    width_at_opposition : int
        Calibration anchor: canonical width (frames) when the field is at opposition.
    coincidence_mode : {'post_threshold', 'joint_statistic'}
        'post_threshold' (default) = adaptive time+coord AND match across active scopes.
        'joint_statistic' = pre-threshold coherent sum/sqrt(N) of aligned per-frame scores.
    time_tolerance_s : float
        Temporal coincidence tolerance.
    coord_tolerance_deg : float
        Spatial (RA/Dec) coincidence tolerance.
    """

    detector: str = "box"
    sigma_threshold: float = DEFAULT_SIGMA_THRESHOLD
    preproc_window: int = DEFAULT_PREPROC_WINDOW
    exposure_time: float = DEFAULT_EXPOSURE_TIME
    canonical_width: Optional[int] = None
    opposition_angle_deg: Optional[float] = None
    width_at_opposition: int = DEFAULT_CANONICAL_WIDTH_AT_OPPOSITION
    coincidence_mode: str = "post_threshold"
    time_tolerance_s: float = DEFAULT_TIME_TOLERANCE_S
    coord_tolerance_deg: float = DEFAULT_COORD_TOLERANCE_DEG

    def __post_init__(self) -> None:
        if self.detector not in ("box", "ricker"):
            raise ValueError(f"detector must be 'box' or 'ricker', got {self.detector!r}")
        if self.coincidence_mode not in ("post_threshold", "joint_statistic"):
            raise ValueError(
                f"coincidence_mode must be 'post_threshold' or 'joint_statistic', "
                f"got {self.coincidence_mode!r}"
            )
