"""Colibri I/O module.

Named ``colibri_io`` (not ``io``) deliberately: a top-level ``io`` package would
shadow Python's stdlib ``io`` for every script run from this directory.

Canonical home of everything that reads from or writes to disk / external
services, so the stage scripts can stop mixing I/O with computation. Mirrors the
package conventions established by ``detection/`` and ``lightcurve_analysis/``.

Submodules
----------
``raw``        -- raw frame I/O. Re-exports the in-place Cython
                  ``colibri_image_reader`` (RCD/FITS read, dark sets, image
                  stacks) plus ``conv_12to16`` from ``bitconverter``. The Cython
                  source and ``bitconverter`` stay at the top level; this just
                  wraps them.
``npy``        -- ``*_stars.npy`` / ``*_pos.npy`` load + inspect helpers
                  (relocated ``read_npy``).
``catalog``    -- Gaia (VizieR) cone-search query (relocated ``VizieR_query``).
``astrometry`` -- astrometry.net plate solve (relocated ``astrometrynet_funcs``)
                  + pixel->RA/Dec WCS transforms (relocated ``getRAdec``). The
                  duplicated ``getRAdec_arrays()`` (getRAdec.py vs
                  coordsfinder.py) is consolidated here as the single copy.

Every legacy top-level import path (``import VizieR_query``, ``import getRAdec``,
``import read_npy``, ``import astrometrynet_funcs``) still works via thin
back-compat shims that re-export from these submodules.
"""

from colibri_io.catalog import makeQuery
from colibri_io.npy import read_plot, to_ds9
from colibri_io.astrometry import (
    getSolution,
    getLocalSolution,
    getRAdec,
    getRAdec_arrays,
    getRAdecfromFile,
    getXY,
    getRAdecSingle,
    getXYSingle,
)
from colibri_io.raw import (
    getDateTime,
    readRCD,
    readFITS,
    importFramesFITS,
    importFramesRCD,
    testGPSLock,
    split_images,
    getSizeFITS,
    getSizeRCD,
    makeMasterDark,
    makeDarkSet,
    chooseDark,
    stackImages,
    conv_12to16,
)

__all__ = [
    # catalog
    "makeQuery",
    # npy
    "read_plot",
    "to_ds9",
    # astrometry
    "getSolution",
    "getLocalSolution",
    "getRAdec",
    "getRAdec_arrays",
    "getRAdecfromFile",
    "getXY",
    "getRAdecSingle",
    "getXYSingle",
    # raw frames
    "getDateTime",
    "readRCD",
    "readFITS",
    "importFramesFITS",
    "importFramesRCD",
    "testGPSLock",
    "split_images",
    "getSizeFITS",
    "getSizeRCD",
    "makeMasterDark",
    "makeDarkSet",
    "chooseDark",
    "stackImages",
    "conv_12to16",
]
