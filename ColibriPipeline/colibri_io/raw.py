# -*- coding: utf-8 -*-
"""
Raw frame I/O: re-exports of the in-place Cython ``colibri_image_reader`` module.

The Cython source (``colibri_image_reader.pyx`` / ``.py``) is compiled in place
by the ``Init/`` build scripts and MUST stay at the top level so the build
artifacts (.so/.pyd) resolve correctly. This module does NOT relocate it; it
just provides a ``colibri_io.raw`` namespace that re-exports its public
functions, so callers can use either ``import colibri_image_reader`` (legacy) or
``from colibri_io.raw import importFramesRCD`` (new).

Also re-exports ``conv_12to16`` from the low-level ``bitconverter`` dependency
for convenience; ``bitconverter`` itself stays at the top level unchanged.
"""

# Stages run flat from the ColibriPipeline/ dir, so plain top-level imports
# work. When imported as part of the package (python -m / pytest) the package
# parent is on sys.path via the same flat layout; guard just in case.
try:
    import colibri_image_reader as _cir
    import bitconverter as _bc
except ImportError:  # pragma: no cover - defensive for non-flat invocation
    import sys as _sys
    from pathlib import Path as _Path
    _pkg_parent = _Path(__file__).resolve().parent.parent
    if str(_pkg_parent) not in _sys.path:
        _sys.path.insert(0, str(_pkg_parent))
    import colibri_image_reader as _cir
    import bitconverter as _bc

# --- public frame-reader API (verified against colibri_image_reader) ---
getDateTime = _cir.getDateTime
readRCD = _cir.readRCD
readFITS = _cir.readFITS
importFramesFITS = _cir.importFramesFITS
importFramesRCD = _cir.importFramesRCD
testGPSLock = _cir.testGPSLock
split_images = _cir.split_images
getSizeFITS = _cir.getSizeFITS
getSizeRCD = _cir.getSizeRCD
makeMasterDark = _cir.makeMasterDark
makeDarkSet = _cir.makeDarkSet
chooseDark = _cir.chooseDark
stackImages = _cir.stackImages

# --- low-level bit conversion (from bitconverter) ---
conv_12to16 = _bc.conv_12to16

__all__ = [
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
