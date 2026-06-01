#!/usr/bin/env python3
"""
Filename : plotting.py
Author   : ColibriPipeline / lightcurve_analysis (Wave 2)

Headless-safe figure production for night-long light curves.

Public functions
----------------
    use_headless()                       force the Agg backend (CLI use)
    plot_single(lc, ...)                 one LightCurve -> (fig, ax)
    plot_raw_vs_diff(cat, star_id, ...)  stacked raw + differential -> fig
    plot_multi_grid(cat, star_ids, ...)  grid of curves -> fig
    plot_full_and_zoom(cat, star_id,...) full night + zoom slice -> fig
    save_figure(fig, out_path, ...)      save to disk (mkdir, bbox tight)

Matches the repo's plain-matplotlib conventions (see ``lightcurve_looker.py``):
``plt.subplots()``, ``bbox_inches='tight'``, descriptive axis labels and a star
identifier in the title. Default save dpi is 150.

The differential light curve is imported *lazily* inside the functions that
need it (``differential.py`` may be built in parallel), so this module imports
cleanly even when ``differential.py`` does not yet exist.
"""

from pathlib import Path

import numpy as np

# --- Headless-safe backend selection ---------------------------------------
# Decide the backend before importing pyplot. If matplotlib already selected a
# non-interactive backend (e.g. 'agg'), leave it. If no display is available
# (no DISPLAY / WAYLAND_DISPLAY on a Unix box and not on Windows/macOS), force
# Agg so figures can still be produced. Interactive backends (notebook inline,
# Qt, Tk, ...) are left untouched so notebook inline keeps working.
import os
import sys

import matplotlib

_BACKEND = matplotlib.get_backend()


def _no_display_available():
    """True when there is clearly no interactive display to draw on."""
    if sys.platform.startswith('win') or sys.platform == 'darwin':
        return False
    return not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


def _in_ipython_kernel():
    """True when running inside an IPython/Jupyter kernel."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None
    except ImportError:
        return False


_NON_INTERACTIVE = {'agg', 'pdf', 'ps', 'svg', 'cairo', 'template'}
# Only force Agg when truly headless AND not running inside a Jupyter kernel.
# Jupyter handles its own rendering (inline / widget backend) regardless of
# whether $DISPLAY is set on the host, so forcing Agg there would break
# inline figure display.
if (_BACKEND.lower() not in _NON_INTERACTIVE
        and _no_display_available()
        and not _in_ipython_kernel()):
    matplotlib.use('Agg')
    _BACKEND = matplotlib.get_backend()

import matplotlib.pyplot as plt  # noqa: E402


def use_headless():
    """Force the non-interactive Agg backend (CLI entry points call this)."""
    global _BACKEND
    matplotlib.use('Agg')
    _BACKEND = matplotlib.get_backend()
    # Rebind pyplot's backend in-process.
    plt.switch_backend('Agg')


# --- Repo-matched styling defaults -----------------------------------------
_DEFAULT_DPI = 150
_LINE_COLOR = 'tab:blue'
_DIFF_COLOR = 'tab:purple'


def _flux_label(kind):
    """Axis label appropriate to a LightCurve's ``kind``."""
    if kind in ('diff', 'differential', 'diff_binned'):
        return 'Relative flux'
    return 'Counts/circular aperture'


def _time_for_plot(t, mode='hours'):
    """Convert a unix-time vector to plottable x-values and an axis label.

    Parameters
    ----------
    t : array_like
        Unix seconds (may contain NaN; ignored for the origin estimate).
    mode : {'hours', 'seconds', 'datetime'}
        'hours'    -> hours elapsed from the first finite sample.
        'seconds'  -> seconds elapsed from the first finite sample.
        'datetime' -> matplotlib datetime floats (UTC) for date-formatted axes.

    Returns
    -------
    (x_values, xlabel)
        x_values is a float ndarray the same length as ``t``.
    """
    t = np.asarray(t, dtype=float)
    finite = np.isfinite(t)
    if not finite.any():
        return t, 'time'
    t0 = t[finite].min()

    if mode == 'seconds':
        return t - t0, 'Time from night start (s)'
    if mode == 'datetime':
        import datetime as _dt
        import matplotlib.dates as _mdates
        x = np.full(t.shape, np.nan)
        idx = np.flatnonzero(finite)
        x[idx] = _mdates.date2num([
            _dt.datetime.fromtimestamp(v, tz=_dt.timezone.utc)
            for v in t[idx]
        ])
        return x, 'Time (UTC)'
    # default: hours
    return (t - t0) / 3600.0, 'Time from night start (hr)'


def _apply_datetime_axis(ax, mode):
    """Format the x-axis as dates when ``mode == 'datetime'``."""
    if mode != 'datetime':
        return
    import matplotlib.dates as _mdates
    ax.xaxis.set_major_formatter(_mdates.DateFormatter('%H:%M'))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_horizontalalignment('right')


def _insert_gap_breaks(t, f):
    """Return (t, f) with NaN inserted at segment boundaries so a line breaks.

    Uses ``stitch.gap_mask`` to find samples preceded by a time gap and injects
    a single NaN flux sample just before each, breaking the connecting line.
    """
    try:
        if __package__:
            from .stitch import gap_mask
        else:  # flat execution from inside lightcurve_analysis/
            from stitch import gap_mask  # type: ignore
    except ImportError:
        return t, f
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    if t.size < 2:
        return t, f
    mask = gap_mask(t)
    breaks = np.flatnonzero(mask)
    if breaks.size == 0:
        return t, f
    # Insert a NaN flux row at each break index (before the sample).
    new_t = np.insert(t, breaks, t[breaks])
    new_f = np.insert(f, breaks, np.nan)
    return new_t, new_f


def plot_single(lc, ax=None, title=None, show_gaps=True, time_mode='hours',
                style='line'):
    """Plot one :class:`LightCurve`.

    Parameters
    ----------
    lc : LightCurve
        Has ``t`` (unix seconds), ``f`` (flux, NaN at gaps), ``kind``,
        ``bin_seconds``, ``star_id``.
    ax : matplotlib Axes, optional
        Draw onto this axis; if None a new figure/axis is created.
    title : str, optional
        Title. Defaults to a star-id label.
    show_gaps : bool
        If True, break the line at time gaps (line style only).
    time_mode : {'hours', 'seconds', 'datetime'}
        X-axis units (see :func:`_time_for_plot`).
    style : {'line', 'scatter'}
        Line plot (NaNs break the line) or scatter (NaNs masked out).

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))
    else:
        fig = ax.figure

    t = np.asarray(lc.t, dtype=float)
    f = np.asarray(lc.f, dtype=float)

    x, xlabel = _time_for_plot(t, mode=time_mode)

    if style == 'scatter':
        good = np.isfinite(x) & np.isfinite(f)
        ax.scatter(x[good], f[good], s=4, color=_LINE_COLOR,
                   label=lc.kind, rasterized=True)
    else:
        if show_gaps and not lc.bin_seconds:
            # Re-derive gap breaks on the unix-time axis, then map to plot x.
            # Skip for binned data: gap_mask's 1/40 Hz threshold flags every
            # bin-to-bin interval as a gap, inserting NaN between every point
            # and making isolated points invisible in a line-only plot.
            # Binned LightCurves already use NaN for empty bins, so matplotlib
            # breaks the line at real gaps naturally.
            tb, fb = _insert_gap_breaks(t, f)
            xb, _ = _time_for_plot(tb, mode=time_mode)
            ax.plot(xb, fb, '-', lw=0.8, color=_LINE_COLOR, label=lc.kind)
        else:
            ax.plot(x, f, '-', lw=0.8, color=_LINE_COLOR, label=lc.kind)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(_flux_label(lc.kind))
    _apply_datetime_axis(ax, time_mode)

    if title is None:
        sid = '' if lc.star_id is None else f'Star #{lc.star_id}'
        kind = lc.kind
        if lc.bin_seconds:
            kind = f'{kind} ({lc.bin_seconds:g}s bins)'
        title = f'{sid}  [{kind}]'.strip()
    ax.set_title(title)

    return fig, ax


def _diff_lightcurve_or_none(cat, star_id, diff_method, comp_rad_pix,
                             bin_seconds):
    """Lazily compute a differential light curve, or None if unavailable.

    Returns
    -------
    (LightCurve or None, reason)
        ``reason`` is None on success, else a short message for an annotation.
    """
    try:
        if __package__:
            from .differential import differential_lightcurve
        else:  # flat execution from inside lightcurve_analysis/
            from differential import differential_lightcurve  # type: ignore
    except ImportError as exc:
        return None, f'differential module unavailable\n({exc})'
    try:
        lc = differential_lightcurve(
            cat, int(star_id), method=diff_method,
            comp_rad_pix=comp_rad_pix, bin_seconds=bin_seconds,
        )
        return lc, None
    except Exception as exc:  # noqa: BLE001 - degrade gracefully in figures
        return None, f'differential failed\n({type(exc).__name__}: {exc})'


def _star_meta(cat, star_id):
    """Return a (gmag, snr) tuple for a star from cat.stars (NaN if absent)."""
    try:
        row = cat.stars.loc[cat.stars['star_id'] == int(star_id)]
        if len(row) == 0:
            return np.nan, np.nan
        return float(row.iloc[0]['gmag']), float(row.iloc[0]['snr'])
    except Exception:  # noqa: BLE001
        return np.nan, np.nan


def plot_raw_vs_diff(cat, star_id, bin_seconds=None, diff_method='ensemble',
                     comp_rad_pix=100.0, time_mode='hours'):
    """Two stacked panels: raw (top) and differential (bottom), shared x-axis.

    Parameters
    ----------
    cat : NightCatalog
    star_id : int
    bin_seconds : float, optional
        If given, the raw panel shows the binned curve and the differential is
        computed at the same binning.
    diff_method : {'ensemble', 'pca'}
    comp_rad_pix : float
        Comparison-star selection radius (pixels) for the differential.
    time_mode : {'hours', 'seconds', 'datetime'}

    Returns
    -------
    matplotlib Figure
    """
    star_id = int(star_id)
    fig, (ax_raw, ax_diff) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True)

    # --- top: raw (or binned) ---
    if bin_seconds:
        raw_lc = cat.get_binned_lightcurve(star_id, bin_seconds)
    else:
        raw_lc = cat.get_raw_lightcurve(star_id, with_gaps=True)
    plot_single(raw_lc, ax=ax_raw, time_mode=time_mode,
                title=None, show_gaps=True)
    ax_raw.set_xlabel('')  # shared axis; label only on the bottom panel

    # --- bottom: differential (lazy) ---
    diff_lc, reason = _diff_lightcurve_or_none(
        cat, star_id, diff_method, comp_rad_pix, bin_seconds)
    if diff_lc is not None:
        plot_single(diff_lc, ax=ax_diff, time_mode=time_mode,
                    title=f'Differential ({diff_method})', show_gaps=True)
        # recolor the diff line for visual distinction
        for line in ax_diff.get_lines():
            line.set_color(_DIFF_COLOR)
    else:
        ax_diff.text(0.5, 0.5, reason, ha='center', va='center',
                     transform=ax_diff.transAxes, fontsize=11,
                     color='firebrick', wrap=True)
        ax_diff.set_ylabel(_flux_label('diff'))
        ax_diff.set_title(f'Differential ({diff_method})')
        x, xlabel = _time_for_plot(np.asarray(raw_lc.t, dtype=float),
                                   mode=time_mode)
        ax_diff.set_xlabel(xlabel)

    gmag, snr = _star_meta(cat, star_id)
    fig.suptitle(
        f'{cat.telescope} {cat.obsdate} — Star #{star_id} '
        f'(gmag={gmag:.2f}, SNR={snr:.2f})',
        fontsize=12)
    fig.tight_layout()
    return fig


def plot_multi_grid(cat, star_ids, kind='raw', ncols=4, bin_seconds=None,
                    time_mode='hours', out_path=None, diff_method='ensemble',
                    comp_rad_pix=100.0):
    """Grid of light curves, one subplot per star.

    Parameters
    ----------
    cat : NightCatalog
    star_ids : sequence[int]
    kind : {'raw', 'binned', 'diff'}
        Which curve to draw for each star.
    ncols : int
        Columns in the grid.
    bin_seconds : float, optional
        Required for ``kind='binned'``; also passed to the differential.
    time_mode : {'hours', 'seconds', 'datetime'}
    out_path : str or Path, optional
        If given, the figure is saved here (via :func:`save_figure`).
    diff_method : {'ensemble', 'pca'}
    comp_rad_pix : float

    Returns
    -------
    matplotlib Figure
    """
    star_ids = [int(s) for s in np.asarray(star_ids).ravel().tolist()]
    n = len(star_ids)
    if n == 0:
        raise ValueError("plot_multi_grid: star_ids is empty.")
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.0 * ncols, 2.6 * nrows),
                             squeeze=False)
    flat = axes.ravel()

    for i, sid in enumerate(star_ids):
        ax = flat[i]
        if kind == 'binned':
            if not bin_seconds:
                raise ValueError("kind='binned' requires bin_seconds.")
            lc = cat.get_binned_lightcurve(sid, bin_seconds)
        elif kind == 'diff':
            lc, reason = _diff_lightcurve_or_none(
                cat, sid, diff_method, comp_rad_pix, bin_seconds)
            if lc is None:
                ax.text(0.5, 0.5, reason, ha='center', va='center',
                        transform=ax.transAxes, fontsize=8,
                        color='firebrick')
                ax.set_title(f'Star #{sid}', fontsize=9)
                continue
        else:  # 'raw'
            lc = cat.get_raw_lightcurve(sid, with_gaps=True)

        plot_single(lc, ax=ax, time_mode=time_mode,
                    title=f'Star #{sid}', show_gaps=True)
        ax.title.set_fontsize(9)
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_size(8)
        ax.yaxis.label.set_size(8)

    # hide unused axes
    for j in range(n, len(flat)):
        flat[j].set_visible(False)

    fig.suptitle(f'{cat.telescope} {cat.obsdate} — {kind} light curves',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    if out_path is not None:
        save_figure(fig, out_path)
    return fig


def plot_full_and_zoom(cat, star_id, zoom_window_s=60, t_zoom_center=None,
                       kind='raw', bin_seconds=None, time_mode='hours',
                       diff_method='ensemble', comp_rad_pix=100.0):
    """Full-night curve (top, with shaded zoom region) + zoomed slice (bottom).

    Parameters
    ----------
    cat : NightCatalog
    star_id : int
    zoom_window_s : float
        Width (seconds) of the zoom window.
    t_zoom_center : float, optional
        Unix-time center of the zoom. Defaults to the deepest dip (argmin of
        flux) or, failing that, mid-night.
    kind : {'raw', 'binned', 'diff'}
    bin_seconds : float, optional
        Required for ``kind='binned'``.
    time_mode : {'hours', 'seconds', 'datetime'}
        Units for the full-night (top) panel; the zoom panel uses seconds.
    diff_method : {'ensemble', 'pca'}
    comp_rad_pix : float

    Returns
    -------
    matplotlib Figure
    """
    star_id = int(star_id)

    if kind == 'binned':
        if not bin_seconds:
            raise ValueError("kind='binned' requires bin_seconds.")
        lc = cat.get_binned_lightcurve(star_id, bin_seconds)
    elif kind == 'diff':
        lc, reason = _diff_lightcurve_or_none(
            cat, star_id, diff_method, comp_rad_pix, bin_seconds)
        if lc is None:
            # fall back to raw so the figure still renders
            lc = cat.get_raw_lightcurve(star_id, with_gaps=True)
    else:
        lc = cat.get_raw_lightcurve(star_id, with_gaps=True)

    t = np.asarray(lc.t, dtype=float)
    f = np.asarray(lc.f, dtype=float)

    # choose zoom center
    if t_zoom_center is None:
        finite = np.isfinite(f) & np.isfinite(t)
        if finite.any():
            sub_t = t[finite]
            sub_f = f[finite]
            t_zoom_center = float(sub_t[int(np.argmin(sub_f))])
        else:
            fin_t = t[np.isfinite(t)]
            t_zoom_center = float(np.median(fin_t)) if fin_t.size else 0.0

    half = float(zoom_window_s) / 2.0
    lo = t_zoom_center - half
    hi = t_zoom_center + half

    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(11, 7))

    # --- top: full night ---
    plot_single(lc, ax=ax_full, time_mode=time_mode,
                title='Full night', show_gaps=True)
    # shade the zoom region in the full panel's x-units
    xfull, _ = _time_for_plot(np.array([lo, hi], dtype=float), mode=time_mode)
    # _time_for_plot uses the curve's own origin; recompute against full t to
    # keep the shaded span consistent with the plotted x-axis.
    x_lohi, _ = _time_for_plot(t, mode=time_mode)
    finite_t = np.isfinite(t)
    if finite_t.any():
        t0 = t[finite_t].min()
        if time_mode == 'seconds':
            xlo, xhi = lo - t0, hi - t0
        elif time_mode == 'hours':
            xlo, xhi = (lo - t0) / 3600.0, (hi - t0) / 3600.0
        else:  # datetime
            xlo, xhi = xfull[0], xfull[1]
        ax_full.axvspan(xlo, xhi, color='orange', alpha=0.25,
                        label='zoom region')
        ax_full.legend(loc='best', fontsize=8)

    # --- bottom: zoom slice (seconds relative to window start) ---
    sel = np.isfinite(t) & (t >= lo) & (t <= hi)
    zt = t[sel]
    zf = f[sel]
    if zt.size > 0:
        xz = zt - zt.min()
        ax_zoom.plot(xz, zf, '-', lw=0.9, color=_LINE_COLOR)
        ax_zoom.scatter(xz, zf, s=8, color=_LINE_COLOR)
    else:
        ax_zoom.text(0.5, 0.5, 'no samples in zoom window',
                     ha='center', va='center', transform=ax_zoom.transAxes,
                     color='firebrick')
    ax_zoom.set_xlabel(f'Time from zoom start (s)  '
                       f'[window = {zoom_window_s:g}s]')
    ax_zoom.set_ylabel(_flux_label(lc.kind))
    ax_zoom.set_title('Zoom')

    gmag, snr = _star_meta(cat, star_id)
    fig.suptitle(
        f'{cat.telescope} {cat.obsdate} — Star #{star_id} '
        f'(gmag={gmag:.2f}, SNR={snr:.2f}, kind={lc.kind})',
        fontsize=12)
    fig.tight_layout()
    return fig


def save_figure(fig, out_path, dpi=_DEFAULT_DPI):
    """Save a figure to ``out_path`` (creating parent dirs), bbox tight.

    Matches the repo convention (``bbox_inches='tight'``); default dpi=150.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    return out_path
