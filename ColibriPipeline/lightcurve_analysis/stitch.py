#!/usr/bin/env python3
"""
Filename : stitch.py
Author   : ColibriPipeline / lightcurve_analysis (Wave 1)

Night time-axis construction, gap masking, and binning. Also defines the
``LightCurve`` result object that is the common currency between extraction,
binning, differential photometry, and plotting.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class LightCurve:
    """A single star's light curve (raw, binned, or differential).

    Attributes
    ----------
    t : np.ndarray
        Time axis (unix seconds).
    f : np.ndarray
        Flux (NaN at gaps / empty bins).
    kind : str
        'raw', 'binned', 'diff', 'diff_binned', ...
    bin_seconds : float or None
        Bin width in seconds if binned, else None.
    star_id : int or None
        Global star id this curve belongs to.
    """
    t: np.ndarray
    f: np.ndarray
    kind: str = 'raw'
    bin_seconds: float | None = None
    star_id: int | None = None


def build_time_axis(minutes, minute_times, nominal_hz=40.0):
    """Build the common night time axis by concatenating per-minute times.

    The per-minute ``time`` vector is used directly when it has adequate
    resolution. Some (e.g. simulated) data carry coarsely quantized timestamps
    where many consecutive frames share a value; in that case a uniform axis
    at ``nominal_hz`` is synthesized within the minute's [t0, t0+span] range so
    that downstream binning and dt assumptions stay sane. Gaps between minutes
    are preserved by the real start times.

    Parameters
    ----------
    minutes : list[MinuteRef]
    minute_times : list[np.ndarray]
        Per-minute raw time vectors (same order as ``minutes``).
    nominal_hz : float
        Nominal sampling rate used to synthesize coarse axes.

    Returns
    -------
    (time_axis, minute_bounds)
        time_axis : (n_samples,) float64 concatenated times.
        minute_bounds : (n_minutes + 1,) int64 offsets into time_axis; minute
            m occupies time_axis[minute_bounds[m]:minute_bounds[m+1]].
    """
    segments = []
    bounds = [0]
    dt_nominal = 1.0 / nominal_hz
    for ref, t in zip(minutes, minute_times):
        t = np.asarray(t, dtype=float)
        n = len(t)
        t_fixed = _maybe_synthesize(t, ref.t_start, dt_nominal)
        segments.append(t_fixed)
        bounds.append(bounds[-1] + n)
    if segments:
        time_axis = np.concatenate(segments)
    else:
        time_axis = np.zeros(0, dtype=float)
    return time_axis, np.asarray(bounds, dtype=np.int64)


def _maybe_synthesize(t, t_start, dt_nominal):
    """Return t unchanged if well-resolved, else a uniform synthetic axis.

    "Coarse" means more than half of consecutive samples have a zero forward
    difference (a hallmark of integer-second-quantized sim timestamps).
    """
    n = len(t)
    if n < 2:
        return t
    d = np.diff(t)
    zero_frac = np.mean(d == 0.0)
    if zero_frac > 0.5:
        t0 = t[0] if np.isfinite(t[0]) else (
            t_start if np.isfinite(t_start) else 0.0)
        return t0 + dt_nominal * np.arange(n, dtype=float)
    return t


def gap_mask(time_axis, nominal_hz=40.0, gap_factor=1.5):
    """Mark samples that start a new segment after a time gap.

    A True at index i means the dt from i-1 to i exceeds
    ``gap_factor / nominal_hz`` (i.e. a gap precedes sample i). Index 0 is
    always False. Plotting code can insert NaNs at these boundaries to break
    the line.

    Returns
    -------
    np.ndarray(bool), shape (n_samples,)
    """
    time_axis = np.asarray(time_axis, dtype=float)
    n = len(time_axis)
    mask = np.zeros(n, dtype=bool)
    if n < 2:
        return mask
    threshold = gap_factor / nominal_hz
    d = np.diff(time_axis)
    mask[1:] = d > threshold
    return mask


def bin_lightcurve(t, f, bin_seconds, how='mean'):
    """Bin a light curve into fixed-width time bins.

    Empty bins get NaN. Reductions are nan-aware. Bins span the full
    [t.min(), t.max()] range.

    Parameters
    ----------
    t, f : array_like
    bin_seconds : float
    how : {'mean', 'median', 'sum'}

    Returns
    -------
    (centers, binned)
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    valid = np.isfinite(t)
    if not valid.any() or bin_seconds is None or bin_seconds <= 0:
        return np.zeros(0), np.zeros(0)
    t = t[valid]
    f = f[valid]

    t0 = t.min()
    t1 = t.max()
    n_bins = max(1, int(np.ceil((t1 - t0) / bin_seconds)))
    edges = t0 + bin_seconds * np.arange(n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    idx = np.clip(np.floor((t - t0) / bin_seconds).astype(int), 0, n_bins - 1)
    binned = np.full(n_bins, np.nan)

    reducer = {
        'mean': np.nanmean,
        'median': np.nanmedian,
        'sum': np.nansum,
    }.get(how)
    if reducer is None:
        raise ValueError(f"Unknown how={how!r}; use mean/median/sum.")

    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        vals = f[sel]
        if np.isfinite(vals).any():
            binned[b] = reducer(vals)
    return centers, binned


def bin_by_count(t, f, n_per_bin):
    """Bin a light curve into groups of ``n_per_bin`` consecutive samples.

    Time center is the nan-aware mean of the group's times; flux is the
    nan-aware mean of the group's flux. A trailing partial group is kept.

    Returns
    -------
    (centers, binned)
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    n = len(t)
    if n == 0 or n_per_bin is None or n_per_bin < 1:
        return np.zeros(0), np.zeros(0)
    n_bins = int(np.ceil(n / n_per_bin))
    centers = np.full(n_bins, np.nan)
    binned = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sl = slice(b * n_per_bin, (b + 1) * n_per_bin)
        tt = t[sl]
        ff = f[sl]
        if np.isfinite(tt).any():
            centers[b] = np.nanmean(tt)
        if np.isfinite(ff).any():
            binned[b] = np.nanmean(ff)
    return centers, binned
