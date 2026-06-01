"""Back-compat shim: moved to stats.snplots during the module refactor.

Both ``import snplots`` / ``from snplots import snr_single`` (legacy) and
``from stats.snplots import snr_single`` (new) keep working. The real SNR
plotting helper code now lives in ``stats/snplots.py``.

Caller in this tree: sensitivity.py uses ``snplots.snr_single``. That symbol is
re-exported here unchanged.
"""
from stats.snplots import *  # noqa: F401,F403
from stats.snplots import snr_single  # noqa: F401
