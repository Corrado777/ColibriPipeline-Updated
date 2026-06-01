"""Back-compat shim: moved to colibri_io.npy during the module refactor.

Both ``import read_npy`` (legacy) and ``from colibri_io.npy import to_ds9``
(new) keep working.
"""
from colibri_io.npy import *                # noqa: F401,F403
from colibri_io.npy import read_plot, to_ds9  # noqa: F401
