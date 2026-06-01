"""Back-compat shim: merged into colibri_io.astrometry during the module refactor.

The plate-solve functions now live alongside the WCS pixel->RA/Dec transforms
in ``colibri_io.astrometry``. Both ``import astrometrynet_funcs`` (legacy) and
``from colibri_io.astrometry import getLocalSolution`` (new) keep working.
"""
from colibri_io.astrometry import *                       # noqa: F401,F403
from colibri_io.astrometry import getSolution, getLocalSolution  # noqa: F401
