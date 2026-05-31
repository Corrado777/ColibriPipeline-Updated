#!/usr/bin/env python3
"""
Filename : paths.py
Author   : ColibriPipeline / lightcurve_analysis (Wave 1)

Flexible resolution of the ColibriArchive root and per-night directory so the
night light-curve analysis can run on a telescope (D:/), on a sim tree
(<root>/<color>/ColibriArchive), or on a dedicated analysis box pointed at an
arbitrary downloaded archive folder.

Resolution precedence (first match wins):
    1. explicit ``archive_root`` argument (or ``night_dir`` directly upstream)
    2. env ``COLIBRI_ARCHIVE_ROOT``
    3. sim layout: env ``COLIBRI_ENV == 'sim'`` ->
       ``COLIBRI_SIM_ROOT/<color>/ColibriArchive`` (color from TELESCOPE_COLORS)
    4. default ``D:/ColibriArchive``
"""

import os
from pathlib import Path

try:
    from .. import colibri_tools as ct
except (ImportError, ValueError):
    import sys as _sys
    from pathlib import Path as _Path
    _pkg_parent = _Path(__file__).resolve().parent.parent
    if str(_pkg_parent) not in _sys.path:
        _sys.path.insert(0, str(_pkg_parent))
    import colibri_tools as ct


TELESCOPE_COLORS = {
    'REDBIRD': 'Red',
    'GREENBIRD': 'Green',
    'BLUEBIRD': 'Blue',
}


def _telescope_color(telescope):
    """Map a telescope label to its sim color subdirectory.

    Accepts canonical names (REDBIRD/GREENBIRD/BLUEBIRD), colors (Red/Green/
    Blue) case-insensitively, and falls back to the raw label.
    """
    if telescope is None:
        return None
    key = str(telescope).upper()
    if key in TELESCOPE_COLORS:
        return TELESCOPE_COLORS[key]
    # already a color?
    for color in TELESCOPE_COLORS.values():
        if key == color.upper():
            return color
    return telescope


def resolve_archive_root(archive_root=None, telescope=None, sim_root=None):
    """Resolve the directory that contains per-night ``<YYYY-MM-DD>/`` folders.

    Parameters
    ----------
    archive_root : str or Path, optional
        Explicit archive root. Wins over everything else.
    telescope : str, optional
        Telescope label (REDBIRD/GREENBIRD/BLUEBIRD or a color). Only used for
        the sim layout.
    sim_root : str or Path, optional
        Override for the sim root (else env ``COLIBRI_SIM_ROOT``).

    Returns
    -------
    pathlib.Path
    """
    # 1. explicit argument
    if archive_root is not None:
        return Path(archive_root)

    # 2. env COLIBRI_ARCHIVE_ROOT
    env_root = os.environ.get('COLIBRI_ARCHIVE_ROOT')
    if env_root:
        return Path(env_root)

    # 3. sim layout
    if os.environ.get('COLIBRI_ENV', '').lower() == 'sim':
        root = sim_root or os.environ.get('COLIBRI_SIM_ROOT')
        if root is None:
            raise ValueError(
                "COLIBRI_ENV=sim but no sim_root given and "
                "COLIBRI_SIM_ROOT is unset.")
        color = _telescope_color(telescope)
        if color is None:
            raise ValueError(
                "Sim layout requires a telescope label to pick the color "
                "subdirectory.")
        return Path(root) / color / 'ColibriArchive'

    # 4. telescope default
    return Path('D:/ColibriArchive')


def night_dir(archive_root, obsdate):
    """Return the night directory ``<archive_root>/<YYYY-MM-DD>``.

    Parameters
    ----------
    archive_root : str or Path
    obsdate : str
        Either ``YYYYMMDD`` or ``YYYY-MM-DD``.

    Returns
    -------
    pathlib.Path
    """
    archive_root = Path(archive_root)
    hyphenated = normalize_obsdate(obsdate)
    return archive_root / hyphenated


def normalize_obsdate(obsdate):
    """Return obsdate in ``YYYY-MM-DD`` form, accepting either input format."""
    s = str(obsdate)
    if '-' in s:
        return s
    return ct.hyphonateDate(s)
