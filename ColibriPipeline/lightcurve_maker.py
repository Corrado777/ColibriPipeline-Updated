"""Back-compat shim: moved to archival.lightcurves during the module refactor.

Both ``import lightcurve_maker`` (legacy) and ``from archival.lightcurves
import getLightcurves`` (new) keep working. The real per-minute light-curve
production code now lives in ``archival/lightcurves.py``.

Callers in this tree use ``lightcurve_maker.getLightcurves`` (sensitivity.py)
and ``lightcurve_maker.importFramesRCD`` (image_stats_dark.py); both -- and the
rest of the module's helpers -- are re-exported here unchanged.
"""
from archival.lightcurves import *  # noqa: F401,F403
from archival.lightcurves import (  # noqa: F401
    getLightcurves,
    importFramesRCD,
    importFramesFITS,
    stackImages,
    getSizeRCD,
    getSizeFITS,
    getDateTime,
    getDark,
    makeDarkSet,
    chooseDark,
    initialFindFITS,
    refineCentroid,
    averageDrift,
    timeEvolveFITS,
    timeEvolveFITSNoDrift,
    clipCutStars,
    fluxCheck,
    split_images,
    nb_read_data,
    readxbytes,
)
