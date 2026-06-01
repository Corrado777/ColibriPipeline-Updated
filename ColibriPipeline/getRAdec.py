"""Back-compat shim: merged into colibri_io.astrometry during the module refactor.

The WCS pixel->RA/Dec transforms now live alongside the astrometry.net
plate-solve functions in ``colibri_io.astrometry``. Both ``import getRAdec``
(legacy) and ``from colibri_io.astrometry import getRAdec`` (new) keep working.

The previously-duplicated ``getRAdec_arrays()`` (also defined in coordsfinder.py)
is now consolidated as the single canonical copy in ``colibri_io.astrometry``.
"""
from colibri_io.astrometry import *  # noqa: F401,F403
from colibri_io.astrometry import (  # noqa: F401
    getRAdec,
    getRAdec_arrays,
    getRAdecfromFile,
    getXY,
    getRAdecSingle,
    getXYSingle,
)
