"""Colibri statistics & timeline module -- the reporting end of the pipeline.

Per-night and cumulative summaries plus the operations timeline. Mirrors the
package conventions established by ``colibri_io/``, ``photometry/`` and
``archival/``: real code lives in submodules, with thin back-compat shims left
at the old top-level import paths.

Planned scope (behaviour-preserving)
------------------------------------
  - darks        <- image_stats_dark.py (dark frame stats + read noise).
  - sensitivity  <- sensitivity.py (mag-vs-SNR limiting-magnitude curves;
                    Gaia cross-match). Network (Gaia) assertions stay soft.
  - cumulative   <- cumulative_stats.py (nightly aggregate counts, star-hours,
                    detection histograms, central CSV log).
  - timeline     <- timeline.py (ACP-log-driven multi-panel operations plot).
  - star_hours   <- getStarHour.py.
  - snplots      <- snplots.py (SNR plotting helpers).

The four big stage scripts (``image_stats_dark.py``, ``sensitivity.py``,
``cumulative_stats.py``, ``timeline.py``) are stage entry points with CLIs the
orchestrator calls; they will be slimmed into thin CLI shims in a later wave.
They are NOT relocated yet, so this package currently owns only ``star_hours``
and ``snplots``.

Submodules (present today)
--------------------------
``star_hours`` -- star-hours per observed field (relocated ``getStarHour.py``);
                  main entry ``getStarHour``. Imported by ``colibri_tools.py``
                  (at module load) and ``timeline.py`` via the
                  ``from getStarHour import getStarHour`` shim.
``snplots``    -- SNR plotting helpers (relocated ``snplots.py``); entry point
                  ``snr_single``, used by ``sensitivity.py`` via the
                  ``import snplots`` shim.

Heavy matplotlib stays inside this package; pure calculations (airmass, altitude
and twilight times already live in ``colibri_tools`` -- ``calculateAlt`` /
``calculateAirmass`` / ``twilightTimesJD`` -- and are NOT duplicated here).
Existing PNG/CSV/txt outputs are unchanged.
"""

from stats.star_hours import getStarHour, fieldCoords
from stats.snplots import snr_single

__all__ = [
    # star-hours per field
    "getStarHour",
    "fieldCoords",
    # SNR plotting helpers
    "snr_single",
]
