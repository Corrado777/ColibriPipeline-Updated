#!/usr/bin/env python3
"""
Filename : io.py
Author   : ColibriPipeline / lightcurve_analysis (Wave 1)

Minute discovery and lazy/mmap loading of per-minute ``*_stars.npy`` files and
their sibling ``*sig_pos.npy`` position files.

Per-minute file conventions (confirmed):
    <minute>_stars.npy      : (n_frames, n_stars, 4) float64, [x, y, flux, unix_t]
    <minute>_<thresh>sig_pos.npy : (n_stars, 5) [x, y, hlr, RA_deg, Dec_deg]
                                   (or (n_stars, 3) before WCS).
The pos file shares star ordering with axis-1 of the stars file, so star i's
RA/Dec is pos[i, 3:5].
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from ..detection_trade_study import lightcurves as lc
    from .. import colibri_tools as ct
except (ImportError, ValueError):
    import sys as _sys
    from pathlib import Path as _Path
    _pkg_parent = _Path(__file__).resolve().parent.parent
    for _p in (str(_pkg_parent), str(_pkg_parent / 'detection_trade_study')):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
    import lightcurves as lc
    import colibri_tools as ct


@dataclass(frozen=True)
class MinuteRef:
    """A reference to one minute's on-disk products."""
    minute_key: str          # e.g. '20250830_01.54.12.147'
    t_start: float           # unix start time (from minute_key)
    stars_path: Path
    pos_path: Path | None    # *sig_pos.npy sibling, or None
    wcs_path: Path | None    # *_wcs.fits sibling, or None


def _minute_key_from_name(name):
    """Extract the canonical minute key prefix from a filename."""
    m = ct.MINDIR_REGEX.search(name)
    if m is None:
        return None
    return m.group()


def _t_start_from_key(minute_key):
    """Convert a minute key to a unix start time."""
    # MINDIR_FORMAT expects microseconds; key has milliseconds -> pad.
    ts = minute_key + '000'
    dt = datetime.strptime(ts, ct.MINDIR_FORMAT)
    return dt.timestamp()


def discover_minutes(night_path):
    """Glob ``*_stars.npy`` in a night dir and pair siblings by minute key.

    Parameters
    ----------
    night_path : str or Path
        A ``<YYYY-MM-DD>`` directory.

    Returns
    -------
    list[MinuteRef]
        Sorted by ``t_start``.
    """
    night_path = Path(night_path)
    refs = []
    for stars_path in sorted(night_path.glob('*_stars.npy')):
        key = _minute_key_from_name(stars_path.name)
        if key is None:
            continue

        # pair sibling pos file (any *sig_pos.npy with same minute key)
        pos_path = None
        for cand in night_path.glob('*sig_pos.npy'):
            if _minute_key_from_name(cand.name) == key:
                pos_path = cand
                break

        # pair sibling wcs file (any *_wcs.fits with same minute key)
        wcs_path = None
        for cand in night_path.glob('*_wcs.fits'):
            if _minute_key_from_name(cand.name) == key:
                wcs_path = cand
                break

        try:
            t_start = _t_start_from_key(key)
        except ValueError:
            t_start = float('nan')

        refs.append(MinuteRef(
            minute_key=key,
            t_start=t_start,
            stars_path=stars_path,
            pos_path=pos_path,
            wcs_path=wcs_path,
        ))

    refs.sort(key=lambda r: (np.isnan(r.t_start), r.t_start, r.minute_key))
    return refs


def load_minute_full(ref, mmap=True):
    """Load one minute's flux/time/xy plus its pos array.

    Parameters
    ----------
    ref : MinuteRef
    mmap : bool
        If True, mmap the (large) stars array and slice lazily, keeping peak
        RAM to roughly one minute. If False, use ``lc.load_minute`` which
        reads the whole array.

    Returns
    -------
    dict with keys:
        'flux' : (n_stars, n_frames) float64
        'time' : (n_frames,) float64
        'xy'   : (n_stars, 2) float64
        'pos'  : (n_stars, k) float64 or None  -- raw pos array
        'radec': (n_stars, 2) float64 or None  -- pos[:, 3:5] if k >= 5
    """
    if mmap:
        arr = np.load(ref.stars_path, mmap_mode='r')
        flux = np.asarray(arr[:, :, 2]).T   # (n_stars, n_frames)
        time = np.asarray(arr[:, 0, 3])     # (n_frames,)
        xy = np.asarray(arr[0, :, 0:2])     # (n_stars, 2)
        del arr
        data = {'flux': flux, 'time': time, 'xy': xy}
    else:
        data = lc.load_minute(ref.stars_path)

    pos, radec = _load_pos(ref)
    data['pos'] = pos
    data['radec'] = radec
    return data


def _load_pos(ref):
    """Load the pos array and derived (radec) for a minute ref."""
    if ref.pos_path is None:
        return None, None
    pos = np.load(ref.pos_path, allow_pickle=True)
    pos = np.atleast_2d(np.asarray(pos, dtype=float))
    radec = pos[:, 3:5] if pos.shape[1] >= 5 else None
    return pos, radec


def minute_radec(ref):
    """Return ``(n_stars, 2)`` RA/Dec for a minute, or None if unavailable.

    Available only when the pos file has >= 5 columns (WCS solved).
    """
    _, radec = _load_pos(ref)
    return radec


def minute_xy(ref):
    """Return ``(n_stars, 2)`` pixel x/y for a minute from the pos file.

    Falls back to reading the first frame of the stars array if no pos file.
    """
    if ref.pos_path is not None:
        pos, _ = _load_pos(ref)
        if pos is not None and pos.shape[1] >= 2:
            return pos[:, 0:2]
    arr = np.load(ref.stars_path, mmap_mode='r')
    xy = np.asarray(arr[0, :, 0:2])
    del arr
    return xy
