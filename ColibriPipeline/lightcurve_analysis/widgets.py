#!/usr/bin/env python3
"""
Filename : widgets.py
Author   : ColibriPipeline / lightcurve_analysis (Wave 3)

Optional interactive explorer for a NightCatalog, built on ipywidgets.

The interactive layer is an *optional extra*: importing this module never
fails just because ipywidgets is missing. When ipywidgets / IPython are
available, :func:`explore` builds a small UI (filters, a star selector,
raw/diff toggle, binning slider, Next/Prev) wired to the headless-safe
plotting functions. When they are absent, :func:`explore` prints a notice and
returns :func:`explore_static` bound to the catalog so the same parameters can
be driven from plain function calls.

Install the optional extras with::

    pip install -r requirements-notebook.txt
"""

from functools import partial

import numpy as np

# --- Guarded optional dependency -------------------------------------------
try:
    import ipywidgets as W
    from IPython.display import display
    HAVE_WIDGETS = True
except ImportError:  # pragma: no cover - exercised when extras absent
    HAVE_WIDGETS = False

# --- Plotting (intra-package, guarded for flat execution) ------------------
if __package__:
    from . import plotting as P
else:  # flat execution from inside lightcurve_analysis/
    import plotting as P  # type: ignore


def explore_static(cat, star_id, kind='raw', bin_seconds=None,
                   diff_method='ensemble', comp_rad_pix=100.0,
                   time_mode='hours'):
    """Render a star's light curve without any widget dependency.

    This is the always-available fallback / programmatic API. It returns the
    matplotlib Figure so it works equally well inline in a notebook or in a
    plain script.

    Parameters
    ----------
    cat : NightCatalog
    star_id : int
    kind : {'raw', 'diff', 'both'}
        'raw'  -> single raw (or binned) panel.
        'diff' -> single differential panel.
        'both' -> stacked raw + differential (:func:`plotting.plot_raw_vs_diff`).
    bin_seconds : float or None
        Bin width (seconds); ``None``/0 means native cadence.
    diff_method : {'ensemble', 'pca'}
    comp_rad_pix : float
        Comparison-star selection radius (pixels) for the differential.
    time_mode : {'hours', 'seconds', 'datetime'}

    Returns
    -------
    matplotlib.figure.Figure
    """
    star_id = int(star_id)
    bs = bin_seconds if bin_seconds else None

    if kind == 'both':
        return P.plot_raw_vs_diff(
            cat, star_id, bin_seconds=bs, diff_method=diff_method,
            comp_rad_pix=comp_rad_pix, time_mode=time_mode)

    if kind == 'diff':
        if __package__:
            from .differential import differential_lightcurve
        else:
            from differential import differential_lightcurve  # type: ignore
        lc = differential_lightcurve(
            cat, star_id, method=diff_method, comp_rad_pix=comp_rad_pix,
            bin_seconds=bs)
        fig, _ = P.plot_single(
            lc, title=f'Star #{star_id} (differential {diff_method})',
            time_mode=time_mode)
        return fig

    # default: raw (binned if requested)
    if bs:
        lc = cat.get_binned_lightcurve(star_id, bs)
    else:
        lc = cat.get_raw_lightcurve(star_id, with_gaps=True)
    fig, _ = P.plot_single(lc, title=f'Star #{star_id} (raw)',
                           time_mode=time_mode)
    return fig


def _star_label(row):
    """Human-readable dropdown label for one stars-table row."""
    sid = int(row['star_id'])
    gmag = row.get('gmag', np.nan)
    snr = row.get('snr', np.nan)
    gmag_s = f'{gmag:.2f}' if np.isfinite(gmag) else '   nan'
    snr_s = f'{snr:.2f}' if np.isfinite(snr) else '  nan'
    return f'#{sid}  g={gmag_s}  SNR={snr_s}'


def _filtered_options(cat, gmag_range, snr_min):
    """Return [(label, star_id), ...] for stars passing the filter, by SNR."""
    gmag = None
    if gmag_range is not None:
        lo, hi = gmag_range
        # Only filter on gmag when at least one finite value exists, else the
        # NaN comparison would drop everything.
        if cat.stars['gmag'].notna().any():
            gmag = (lo, hi)
    df = cat.filter_stars(gmag=gmag, snr_min=snr_min)
    df = df.sort_values('snr', ascending=False)
    return [(_star_label(r), int(r['star_id'])) for _, r in df.iterrows()]


def explore(cat):
    """Interactive (ipywidgets) explorer for a NightCatalog.

    If ipywidgets/IPython are available, builds and displays a UI and returns
    the widget container. Otherwise prints a notice and returns
    :func:`explore_static` bound to ``cat`` (a callable with the same
    rendering parameters).

    Parameters
    ----------
    cat : NightCatalog

    Returns
    -------
    ipywidgets.Widget  (interactive UI)
        when ipywidgets is available, or
    functools.partial
        of :func:`explore_static` bound to ``cat`` otherwise.
    """
    if not HAVE_WIDGETS:
        print("ipywidgets is not installed; interactive explorer disabled.\n"
              "Install the optional extras:\n"
              "    pip install -r requirements-notebook.txt\n"
              "Returning a static plotter: call it as\n"
              "    plot = explore(cat)\n"
              "    plot(star_id, kind='both', bin_seconds=1.0)")
        return partial(explore_static, cat)

    # --- gmag slider bounds from the data -----------------------------------
    gmag_vals = cat.stars['gmag'].to_numpy(dtype=float)
    finite_g = gmag_vals[np.isfinite(gmag_vals)]
    if finite_g.size:
        g_lo, g_hi = float(np.floor(finite_g.min())), \
            float(np.ceil(finite_g.max()))
    else:
        g_lo, g_hi = 0.0, 25.0
    if g_hi <= g_lo:
        g_hi = g_lo + 1.0

    snr_vals = cat.stars['snr'].to_numpy(dtype=float)
    finite_s = snr_vals[np.isfinite(snr_vals)]
    snr_max = float(np.ceil(finite_s.max())) if finite_s.size else 100.0

    # --- widgets -------------------------------------------------------------
    gmag_slider = W.FloatRangeSlider(
        value=[g_lo, g_hi], min=g_lo, max=g_hi,
        step=0.5, description='gmag', continuous_update=False)
    snr_slider = W.FloatSlider(
        value=0.0, min=0.0, max=max(snr_max, 1.0),
        step=0.5, description='SNR min', continuous_update=False)
    star_dropdown = W.Dropdown(description='Star')
    kind_toggle = W.ToggleButtons(
        options=['raw', 'diff', 'both'], value='both', description='Kind')
    bin_slider = W.FloatSlider(
        value=0.0, min=0.0, max=10.0, step=0.5,
        description='Bin (s)', continuous_update=False)
    method_dropdown = W.Dropdown(
        options=['ensemble', 'pca'], value='ensemble', description='Diff')
    comp_slider = W.IntSlider(
        value=100, min=20, max=400, step=10, description='Comp px',
        continuous_update=False)
    prev_btn = W.Button(description='< Prev')
    next_btn = W.Button(description='Next >')
    out = W.Output()

    def _refresh_options(*_):
        opts = _filtered_options(
            cat, tuple(gmag_slider.value), snr_slider.value)
        prev = star_dropdown.value
        # Unobserve while mutating options/value so ipywidgets' own auto-
        # adjustment of value (when prev is no longer in opts) doesn't fire
        # _redraw a second time before the explicit call below.
        star_dropdown.unobserve(_redraw, names='value')
        star_dropdown.options = opts
        if opts:
            values = [v for _, v in opts]
            star_dropdown.value = prev if prev in values else values[0]
        star_dropdown.observe(_redraw, names='value')
        _redraw()

    def _redraw(*_):
        import matplotlib.pyplot as plt
        with out:
            out.clear_output(wait=True)
            if star_dropdown.value is None:
                print("No stars pass the current filters.")
                return
            bs = bin_slider.value if bin_slider.value > 0 else None
            try:
                # plt.ioff() suppresses the %matplotlib inline backend's
                # automatic display-on-create, so the figure renders exactly
                # once via the explicit display() call below.
                with plt.ioff():
                    fig = explore_static(
                        cat, int(star_dropdown.value), kind=kind_toggle.value,
                        bin_seconds=bs, diff_method=method_dropdown.value,
                        comp_rad_pix=float(comp_slider.value))
                display(fig)
                plt.close(fig)
            except Exception as exc:  # noqa: BLE001
                print(f"plot failed: {type(exc).__name__}: {exc}")

    def _step(delta):
        values = [v for _, v in star_dropdown.options]
        if not values:
            return
        try:
            i = values.index(star_dropdown.value)
        except ValueError:
            i = 0
        star_dropdown.value = values[(i + delta) % len(values)]

    # --- wiring --------------------------------------------------------------
    gmag_slider.observe(_refresh_options, names='value')
    snr_slider.observe(_refresh_options, names='value')
    star_dropdown.observe(_redraw, names='value')
    kind_toggle.observe(_redraw, names='value')
    bin_slider.observe(_redraw, names='value')
    method_dropdown.observe(_redraw, names='value')
    comp_slider.observe(_redraw, names='value')
    prev_btn.on_click(lambda _b: _step(-1))
    next_btn.on_click(lambda _b: _step(+1))

    _refresh_options()

    controls = W.VBox([
        W.HBox([gmag_slider, snr_slider]),
        W.HBox([star_dropdown, prev_btn, next_btn]),
        W.HBox([kind_toggle, bin_slider]),
        W.HBox([method_dropdown, comp_slider]),
    ])
    ui = W.VBox([controls, out])
    display(ui)
    return ui
