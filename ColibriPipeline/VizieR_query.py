"""Back-compat shim: moved to colibri_io.catalog during the module refactor.

Both ``import VizieR_query`` (legacy) and ``from colibri_io.catalog import
makeQuery`` (new) keep working.
"""
from colibri_io.catalog import *           # noqa: F401,F403
from colibri_io.catalog import makeQuery   # noqa: F401
