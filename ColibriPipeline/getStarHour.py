"""Back-compat shim: moved to stats.star_hours during the module refactor.

Both ``from getStarHour import getStarHour`` (legacy) and ``from
stats.star_hours import getStarHour`` (new) keep working. The real star-hours
code now lives in ``stats/star_hours.py``.

Callers in this tree use ``from getStarHour import getStarHour`` (colibri_tools.py
imports it at module load, and timeline.py). That symbol -- and the rest of the
module's RCD/dark/field helpers -- are re-exported here unchanged.
"""
from stats.star_hours import *  # noqa: F401,F403
from stats.star_hours import (  # noqa: F401
    getStarHour,
    fieldCoords,
    importFramesRCD,
    readRCD,
    readxbytes,
    nb_read_data,
    split_images,
    initialFindFITS,
    getDark,
    getDateTime,
    makeDarkSet,
    chooseDark,
)
