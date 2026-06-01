# -*- coding: utf-8 -*-
"""Back-compat shim: moved to archival.plots during the module refactor.

Both ``import lightcurve_looker`` (legacy) and ``from archival.plots import
plot_wholecurves`` (new) keep working. The light-curve plotting code now lives
in ``archival/plots.py``.
"""
from archival.plots import *  # noqa: F401,F403
from archival.plots import (  # noqa: F401
    plot_wholecurves,
    plot_event,
)
