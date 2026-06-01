"""Colibri archival module.

Owns the production and writing of per-night data products: per-minute light
curves, their plots, synthetic (forced) light curves, mean stacks, and the
single seam through which all night products are written. Mirrors the package
conventions established by ``colibri_io/`` and ``detection/``.

Submodules
----------
``lightcurves`` -- per-minute star light-curve production (relocated
                   ``lightcurve_maker``); main entry ``getLightcurves``.
``plots``       -- light-curve plotting (relocated ``lightcurve_looker``):
                   ``plot_wholecurves`` / ``plot_event``.
``synthetic``   -- synthetic / forced light curves for a specific star+time,
                   used to fill in missing scopes (relocated
                   ``generate_specific_lightcurve``); entry point ``main``.
``products``    -- ``write_night_product(...)``, the ONE function every night-
                   product write routes through. Today a behaviour-preserving
                   pass-through reproducing the legacy bytes; it is the
                   extension point for the (not-yet-built) per-night JSON
                   manifest, consolidated mean-stack archive, Parquet/NPZ night
                   catalog, and nightly report bundle.

Mean stacks: the implementation stays in the Cython ``colibri_image_reader``
(re-exported by ``colibri_io.raw``). ``stackImages`` is re-exported here only
for a tidy "mean-stack archival" surface; it is NOT reimplemented.

Every legacy top-level import path (``import lightcurve_maker``,
``import lightcurve_looker``, ``import generate_specific_lightcurve``) still
works via thin back-compat shims that re-export from these submodules.

Existing outputs stay byte-identical; this package only relocates the code and
introduces the write seam.
"""

from archival.lightcurves import getLightcurves
from archival.plots import plot_wholecurves, plot_event
from archival.synthetic import main as generate_specific_lightcurve
from archival.products import write_night_product
from colibri_io.raw import stackImages

__all__ = [
    # per-minute light curves
    "getLightcurves",
    # plots
    "plot_wholecurves",
    "plot_event",
    # synthetic / forced light curves
    "generate_specific_lightcurve",
    # write seam
    "write_night_product",
    # mean-stack archival surface (impl lives in colibri_io / Cython)
    "stackImages",
]
