"""Back-compat shim: moved to archival.synthetic during the module refactor.

Both ``import generate_specific_lightcurve`` (legacy; e.g. simultaneous_occults
imports it as ``gsl``) and ``from archival.synthetic import main`` (new) keep
working. The ``python generate_specific_lightcurve.py <date> <ts> <ra> <dec>``
CLI is preserved by delegating to ``archival.synthetic._cli``.
"""
from archival.synthetic import *  # noqa: F401,F403
from archival.synthetic import (  # noqa: F401
    main,
    generateLightcurve,
    saveLightcurve,
    findMinute,
    findMatchedDir,
    improveCentralFrame,
    reversePixelMapping,
    hyphonateDate,
    _cli,
)

if __name__ == '__main__':
    _cli()
