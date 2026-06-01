"""Interactive matched-event viewer for the detection module.

Reads the on-disk ``matched/<ts>-Tier<N>/`` layout written by the rerun +
coincidence flow (see :mod:`detection.rerun` / :mod:`detection.coincidence`) and
renders per-event multi-telescope light curves, mirroring the per-matched-event
plot in ``timeline.py`` and the ipywidgets UI style of
``lightcurve_analysis.widgets.explore``.

On-disk layout (confirmed)::

    <archive>/<date>/<tag>_<detector>/matched/<ts>-Tier<N>/
        det_<...>_star<idx>_<TELESCOPE>.txt   (one per participating scope)

Each ``det_*.txt`` is a copy of a primary-detection file with the header line
indices written by ``detection.rerun._write_det_file``::

    idx 4  : '#    Event File: <minute_stem>_<frame>.rcd'
    idx 5  : '#    Star Coords: <x> <y>'
    idx 6  : '#    RA Dec Coords: <ra> <dec>'
    idx 7  : '#    DATE-OBS: <iso>'
    idx 8  : '#    Telescope: <tel>'
    idx 9  : '#    Field: <field>'
    idx 10 : '#    significance: <sig>'
    idx 18 : '#filename     time      flux     conv_flux'
    idx 19+: data rows  ``<filename> <time> <flux> <conv_flux>``

The header parser here is a local re-implementation of the relevant
``coordsfinder.readFile`` / ``readRAdec`` index logic (coordsfinder is NOT
imported -- it has heavy deps).

Dependencies: numpy + matplotlib at module scope; ipywidgets is imported ONLY
inside :func:`explore_matches`, so the module imports headlessly when ipywidgets
is absent.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Telescope -> plot colour (mirrors timeline.COLOURMAPPING, with the green made
# print-darker for white backgrounds, matching the spec's GREEN='#2ca02c').
SCOPE_COLORS = {"REDBIRD": "r", "GREENBIRD": "#2ca02c", "BLUEBIRD": "b"}
_ALL_SCOPES = ("REDBIRD", "GREENBIRD", "BLUEBIRD")

_TIER_RE = re.compile(r"-Tier(\d+)\s*$")
_STAR_RE = re.compile(r"_star(\d+)_")


# --------------------------------------------------------------------------- #
#  MatchedEvent schema
# --------------------------------------------------------------------------- #

@dataclass
class MatchedEvent:
    """One matched (coincident) dip event across telescopes.

    Attributes
    ----------
    timestamp : str
        The ``<ts>`` part of the ``<ts>-Tier<N>`` directory name.
    tier : int
        Number of participating telescopes (parsed from the ``-TierN`` suffix).
    dir_path : pathlib.Path
        Path to the ``<ts>-Tier<N>`` directory.
    per_scope : dict
        Maps TELESCOPE -> dict with keys ``det_path``, ``star_idx``, ``x``,
        ``y``, ``ra``, ``dec``, ``significance``, ``minute_key``, ``time``,
        ``field``, ``det_time`` (ndarray), ``det_flux`` (ndarray),
        ``det_conv`` (ndarray).
    """

    timestamp: str
    tier: int
    dir_path: Path
    per_scope: Dict[str, dict] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#  det_*.txt parsing
# --------------------------------------------------------------------------- #

def _scope_from_name(name: str) -> Optional[str]:
    """Derive the telescope label from a det filename."""
    upper = name.upper()
    for scope in _ALL_SCOPES:
        if scope in upper:
            return scope
    return None


def _star_idx_from_name(name: str):
    """Extract the ``_star<idx>_`` index from a det filename, or None."""
    m = _STAR_RE.search(name)
    return int(m.group(1)) if m else None


def _minute_key_from_iso(iso: str):
    """Convert a DATE-OBS ISO string to a minute key ``YYYYMMDD_HH.MM.SS.mmm``.

    ``2025-08-30T01:54:38.000000`` -> ``20250830_01.54.38.000``.
    Returns ``None`` if the string cannot be parsed.
    """
    if not iso or "T" not in iso:
        return None
    date_part, time_part = iso.split("T", 1)
    date_part = date_part.strip()
    time_part = time_part.strip()
    try:
        ymd = date_part.replace("-", "")
        if len(ymd) != 8 or not ymd.isdigit():
            return None
        # time_part like HH:MM:SS.mmmmmm
        hms, _, frac = time_part.partition(".")
        hh, mm, ss = hms.split(":")
        millis = (frac + "000")[:3] if frac else "000"
        return f"{ymd}_{hh}.{mm}.{ss}.{millis}"
    except (ValueError, IndexError):
        return None


def _minute_key_from_event_file(event_file: str):
    """Recover the stars.npy minute key from the ``Event File`` header value.

    ``20250830_01.54.12.147_1047.rcd`` -> ``20250830_01.54.12.147``.
    This is the true on-disk stars-file key (the DATE-OBS-derived key may differ
    because DATE-OBS encodes the *event* time, not the minute start).
    """
    if not event_file:
        return None
    stem = event_file.rsplit(".", 1)[0]  # drop extension
    # strip the trailing _<frame>
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem or None


def _parse_det(path: Path) -> dict:
    """Parse one det_*.txt into header fields + data arrays.

    Mirrors the index logic of ``coordsfinder.readFile`` / ``readRAdec`` (line
    5=Star Coords, 6=RA/Dec, 7=DATE-OBS, 9=Field, 10=significance; data rows
    ``filename time flux conv_flux`` after the ``#filename`` header at idx 18).

    Returns a dict with ``x, y, ra, dec, date_obs, field, significance,
    minute_key, event_file, time, det_time, det_flux, det_conv``.
    """
    path = Path(path)
    x = y = ra = dec = significance = float("nan")
    date_obs = None
    field_name = None
    event_file = None

    times: List[float] = []
    fluxes: List[float] = []
    convs: List[float] = []

    with path.open() as fh:
        for i, raw in enumerate(fh):
            line = raw.rstrip("\n")
            if line.startswith("#"):
                if i == 4:  # Event File
                    event_file = line.split(":", 1)[1].strip() if ":" in line else None
                elif i == 5:  # Star Coords: <x> <y>
                    toks = line.split(":", 1)[1].split()
                    if len(toks) >= 2:
                        x, y = _safe_float(toks[0]), _safe_float(toks[1])
                elif i == 6:  # RA Dec Coords: <ra> <dec>
                    toks = line.split(":", 1)[1].split()
                    if len(toks) >= 2:
                        ra, dec = _safe_float(toks[0]), _safe_float(toks[1])
                elif i == 7:  # DATE-OBS: <iso>
                    date_obs = line.split(":", 1)[1].strip() if ":" in line else None
                elif i == 9:  # Field: <field>
                    field_name = line.split(":", 1)[1].strip() if ":" in line else None
                elif i == 10:  # significance: <sig>
                    significance = _safe_float(line.split(":", 1)[1])
                continue

            # data row: filename time flux conv_flux
            toks = line.split()
            if len(toks) >= 4:
                times.append(_safe_float(toks[1]))
                fluxes.append(_safe_float(toks[2]))
                convs.append(_safe_float(toks[3]))

    minute_key = _minute_key_from_iso(date_obs) if date_obs else None
    stars_key = _minute_key_from_event_file(event_file) or minute_key

    # event "time": the seconds field column convention is opaque; prefer the
    # ISO DATE-OBS string for display.
    time_val = date_obs

    return {
        "x": x,
        "y": y,
        "ra": ra,
        "dec": dec,
        "date_obs": date_obs,
        "field": field_name,
        "significance": significance,
        "minute_key": minute_key,
        "stars_key": stars_key,
        "event_file": event_file,
        "time": time_val,
        "det_time": np.asarray(times, dtype=float),
        "det_flux": np.asarray(fluxes, dtype=float),
        "det_conv": np.asarray(convs, dtype=float),
    }


def _safe_float(s) -> float:
    try:
        return float(str(s).strip())
    except (ValueError, TypeError):
        return float("nan")


# --------------------------------------------------------------------------- #
#  Event discovery
# --------------------------------------------------------------------------- #

def load_matched_events(matched_dir) -> List[MatchedEvent]:
    """Load every ``<ts>-Tier<N>/`` matched event under ``matched_dir``.

    Parameters
    ----------
    matched_dir : str or Path
        A ``.../matched`` directory containing ``<ts>-Tier<N>/`` subdirs.

    Returns
    -------
    list of MatchedEvent
        Sorted by timestamp. Malformed dirs / det files are skipped with a
        warning rather than raising.
    """
    matched_dir = Path(matched_dir)
    events: List[MatchedEvent] = []

    if not matched_dir.is_dir():
        warnings.warn(f"matched dir does not exist: {matched_dir}")
        return events

    for sub in sorted(matched_dir.iterdir()):
        if not sub.is_dir():
            continue
        m = _TIER_RE.search(sub.name)
        if m is None:
            continue
        tier = int(m.group(1))
        timestamp = sub.name[: m.start()]

        per_scope: Dict[str, dict] = {}
        for det_path in sorted(sub.glob("det_*.txt")):
            scope = _scope_from_name(det_path.name)
            if scope is None:
                warnings.warn(f"could not derive telescope from {det_path.name}; skipping")
                continue
            star_idx = _star_idx_from_name(det_path.name)
            try:
                parsed = _parse_det(det_path)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"failed to parse {det_path}: {exc!r}; skipping")
                continue

            parsed["det_path"] = det_path
            parsed["star_idx"] = star_idx
            # First det per scope wins (mirrors coincidence path-slot rule).
            if scope not in per_scope:
                per_scope[scope] = parsed

        if not per_scope:
            warnings.warn(f"no parseable det files in {sub}; skipping")
            continue

        events.append(MatchedEvent(
            timestamp=timestamp,
            tier=tier,
            dir_path=sub,
            per_scope=per_scope,
        ))

    events.sort(key=lambda e: e.timestamp)
    return events


# --------------------------------------------------------------------------- #
#  Full-minute light-curve cross-indexing
# --------------------------------------------------------------------------- #

def _date_from_minute_key(minute_key):
    """``20250830_01.54.12.147`` -> ``2025-08-30`` (for night_dir lookup)."""
    if not minute_key:
        return None
    head = minute_key.split("_", 1)[0]
    if len(head) == 8 and head.isdigit():
        return f"{head[:4]}-{head[4:6]}-{head[6:8]}"
    return None


def _find_stars_npy(scope, info, archive_roots):
    """Locate ``<root>/<date>/<minute_key>_stars.npy`` for one scope.

    Tries the explicit ``archive_roots`` override first (a {scope: night_dir}
    map, where night_dir already contains the ``<minute_key>_stars.npy`` files),
    then falls back to ``lightcurve_analysis.paths`` resolution. Returns the Path
    or None. Never raises.
    """
    # candidate minute keys: prefer the true stars-file key, fall back to the
    # DATE-OBS-derived key.
    keys = []
    for k in (info.get("stars_key"), info.get("minute_key")):
        if k and k not in keys:
            keys.append(k)
    if not keys:
        return None

    candidate_dirs = []

    if archive_roots and scope in archive_roots:
        # The override points straight at a night dir (test convenience).
        candidate_dirs.append(Path(archive_roots[scope]))

    if not candidate_dirs:
        date = _date_from_minute_key(keys[0])
        if date is not None:
            try:
                from lightcurve_analysis.paths import resolve_archive_root, night_dir
                root = resolve_archive_root(telescope=scope)
                candidate_dirs.append(night_dir(root, date))
            except Exception:  # noqa: BLE001
                pass

    for ndir in candidate_dirs:
        for key in keys:
            cand = ndir / f"{key}_stars.npy"
            if cand.exists():
                return cand
    return None


def event_lightcurves(event: MatchedEvent, archive_roots=None) -> Dict[str, dict]:
    """Build per-scope full-minute light curves for a matched event.

    For each scope, locate ``<minute_key>_stars.npy`` and cross-index the event
    star by nearest ``[x, y]`` at frame 0 (np.hypot argmin, tol ~3 px). On
    success returns the full-minute flux + time axis; otherwise falls back to
    the det-window arrays. Never raises.

    Returns
    -------
    dict
        scope -> ``{'t', 'flux', 'star_col', 'det_time', 'det_flux',
        'det_conv'}``. ``star_col`` is None when no stars.npy was found (det
        fallback).
    """
    out: Dict[str, dict] = {}
    for scope, info in event.per_scope.items():
        det_time = info.get("det_time", np.asarray([], dtype=float))
        det_flux = info.get("det_flux", np.asarray([], dtype=float))
        det_conv = info.get("det_conv", np.asarray([], dtype=float))

        entry = {
            "t": det_time,
            "flux": det_flux,
            "star_col": None,
            "det_time": det_time,
            "det_flux": det_flux,
            "det_conv": det_conv,
        }

        stars_path = _find_stars_npy(scope, info, archive_roots)
        if stars_path is not None:
            try:
                arr = np.load(stars_path, mmap_mode="r")
                xy0 = np.asarray(arr[0, :, 0:2], dtype=float)  # (n_stars, 2)
                ex, ey = info.get("x", np.nan), info.get("y", np.nan)
                if np.isfinite(ex) and np.isfinite(ey) and xy0.size:
                    dists = np.hypot(xy0[:, 0] - ex, xy0[:, 1] - ey)
                    col = int(np.argmin(dists))
                    if dists[col] <= 3.0:
                        flux = np.asarray(arr[:, col, 2], dtype=float)
                        t = np.asarray(arr[:, 0, 3], dtype=float)
                        entry["t"] = t
                        entry["flux"] = flux
                        entry["star_col"] = col
                del arr
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"failed to load {stars_path}: {exc!r}; using det window")

        out[scope] = entry
    return out


# --------------------------------------------------------------------------- #
#  Plotting
# --------------------------------------------------------------------------- #

def _event_date(event: MatchedEvent):
    """Best-effort date string for the plot title."""
    for info in event.per_scope.values():
        date = _date_from_minute_key(info.get("stars_key") or info.get("minute_key"))
        if date:
            return date
    # fall back to the timestamp dir prefix (e.g. 2025-08-29_215438_000000)
    return event.timestamp.split("_", 1)[0] if "_" in event.timestamp else event.timestamp


def _normalize_flux_curve(flux, scale=None):
    """Normalize a flux curve by a robust positive scale.

    The matched-event viewer compares telescopes with different throughput, so
    the plotted raw flux, convolved flux, and highlighted dip window all share
    the same per-star normalization factor. The baseline is the median of the
    available raw flux samples; if that is unavailable or degenerate, we fall
    back to a scale of 1.0.
    """
    flux = np.asarray(flux, dtype=float)

    if scale is None:
        finite = flux[np.isfinite(flux)]
        scale = float(np.nanmedian(finite)) if finite.size else 1.0

    if not np.isfinite(scale) or scale == 0.0:
        scale = 1.0

    return flux / scale, scale


def plot_event(event: MatchedEvent, archive_roots=None, ax=None):
    """Plot a matched event's per-scope normalized flux with the dip window.

    Mirrors the per-matched-event plot in ``timeline.py``: each scope's curve in
    its colour, the dip window highlighted with an ``axvspan``, the per-scope
    convolution overlaid on a twin axis, the title ``"<date>: <ts>-Tier<N>"``
    plus the per-scope sigma list, and a legend keyed by scope/coords/sigma.

    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    curves = event_lightcurves(event, archive_roots=archive_roots)
    twin = ax.twinx()

    sigma_list = []
    plotted_any = False
    for scope in _ALL_SCOPES:
        if scope not in event.per_scope:
            continue
        info = event.per_scope[scope]
        data = curves.get(scope, {})
        color = SCOPE_COLORS.get(scope, "k")

        t = np.asarray(data.get("t", []), dtype=float)
        flux = np.asarray(data.get("flux", []), dtype=float)
        det_f = np.asarray(info.get("det_flux", []), dtype=float)
        det_c = np.asarray(info.get("det_conv", []), dtype=float)

        flux_norm, flux_scale = _normalize_flux_curve(flux, scale=None)
        det_f_norm, _ = _normalize_flux_curve(det_f, scale=flux_scale)
        det_c_norm, _ = _normalize_flux_curve(det_c, scale=flux_scale)

        ra = info.get("ra", float("nan"))
        dec = info.get("dec", float("nan"))
        sig = info.get("significance", float("nan"))
        sigma_list.append((scope, f"{sig:.2f}" if np.isfinite(sig) else "nan"))

        if t.size and flux_norm.size:
            n = min(t.size, flux_norm.size)
            ax.plot(
                t[:n], flux_norm[:n], color=color, lw=0.8, alpha=0.85,
                label=f"{scope}: ({ra:.4f}, {dec:.4f}) sig={sig:.2f}",
            )
            plotted_any = True

        # convolution overlay on the twin axis (over the det window).
        det_t = np.asarray(info.get("det_time", []), dtype=float)

        # dip-window extent: map the det window onto the full-minute time axis
        # when we have a full-minute curve; otherwise use the det time column.
        if data.get("star_col") is not None and t.size and det_t.size:
            # full-minute curve present: highlight via flux minimum within the
            # det window's flux signature. The det window covers ~2s; centre the
            # span on the deepest det-flux sample mapped back by value proximity.
            _highlight_window(ax, t, flux_norm, det_f_norm, color)
        elif det_t.size:
            ax.axvspan(float(det_t.min()), float(det_t.max()),
                       color=color, alpha=0.08)

        if det_c_norm.size:
            xaxis = det_t if det_t.size == det_c.size else np.arange(det_c.size)
            twin.plot(xaxis, det_c_norm, color=color, lw=0.7, ls="--", alpha=0.5)

    if not plotted_any:
        ax.text(0.5, 0.5, "no light-curve data", ha="center", va="center",
                transform=ax.transAxes)

    date = _event_date(event)
    sig_str = ", ".join(f"{s}={v}" for s, v in sigma_list)
    ax.set_title(f"{date}: {event.timestamp}-Tier{event.tier}\n{sig_str}")
    ax.set_xlabel("time")
    ax.set_ylabel("normalized flux")
    twin.set_ylabel("normalized conv flux")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    return fig


def _highlight_window(ax, t, flux, det_flux, color):
    """Highlight the dip window on the full-minute axis.

    Finds where the det-window flux signature sits in the full-minute curve by
    locating the full-minute frame nearest the det-window minimum, then spans a
    small window around it. Best-effort; silently no-ops on degenerate input.
    """
    try:
        if not det_flux.size or not flux.size:
            return
        target = float(np.nanmin(det_flux))
        idx = int(np.nanargmin(np.abs(flux - target)))
        half = max(1, det_flux.size // 2)
        lo = max(0, idx - half)
        hi = min(t.size - 1, idx + half)
        ax.axvspan(float(t[lo]), float(t[hi]), color=color, alpha=0.08)
    except (ValueError, IndexError):
        return


# --------------------------------------------------------------------------- #
#  Interactive explorer (ipywidgets imported lazily)
# --------------------------------------------------------------------------- #

def explore_matches(matched_dir, archive_roots=None):
    """Interactive ipywidgets explorer for matched events.

    Mirrors ``lightcurve_analysis.widgets.explore``: a tier filter, a
    significance-min slider, Prev/Next buttons, and an Output area that calls
    :func:`plot_event` for the current filtered event. ipywidgets is imported
    INSIDE this function so the module imports headlessly without it.

    Returns the assembled widget (a VBox) so it displays in Jupyter. If
    ipywidgets is unavailable, prints a notice and returns a callable that
    plots an event by index.
    """
    events = load_matched_events(matched_dir)

    try:
        import ipywidgets as W
        from IPython.display import display
    except Exception:  # noqa: BLE001 - exercised when extras absent
        print("ipywidgets is not installed; interactive explorer disabled.\n"
              "Install the optional extras:\n"
              "    pip install -r requirements-notebook.txt\n"
              "Returning a static plotter: call it as plot(index).")

        def _static(index=0):
            if not events:
                print("no matched events found.")
                return None
            index = max(0, min(len(events) - 1, int(index)))
            return plot_event(events[index], archive_roots=archive_roots)

        return _static

    import matplotlib.pyplot as plt

    tiers = sorted({e.tier for e in events})
    tier_options = ["all"] + [str(t) for t in tiers]

    sig_vals = []
    for e in events:
        for info in e.per_scope.values():
            s = info.get("significance", float("nan"))
            if np.isfinite(s):
                sig_vals.append(s)
    sig_max = float(np.ceil(max(sig_vals))) if sig_vals else 20.0

    tier_toggle = W.ToggleButtons(options=tier_options, value="all",
                                  description="Tier")
    sig_slider = W.FloatSlider(value=0.0, min=0.0, max=max(sig_max, 1.0),
                               step=0.5, description="sig min",
                               continuous_update=False)
    prev_btn = W.Button(description="< Prev")
    next_btn = W.Button(description="Next >")
    label = W.Label(value="")
    out = W.Output()

    state = {"index": 0, "filtered": list(events)}

    def _max_sig(ev):
        vals = [info.get("significance", float("nan"))
                for info in ev.per_scope.values()]
        vals = [v for v in vals if np.isfinite(v)]
        return max(vals) if vals else float("-inf")

    def _apply_filter():
        tier_sel = tier_toggle.value
        smin = sig_slider.value
        filtered = []
        for ev in events:
            if tier_sel != "all" and ev.tier != int(tier_sel):
                continue
            if _max_sig(ev) < smin:
                continue
            filtered.append(ev)
        state["filtered"] = filtered
        state["index"] = 0

    def _redraw(*_):
        with out:
            out.clear_output(wait=True)
            filtered = state["filtered"]
            if not filtered:
                label.value = "no matched events pass the current filters."
                print(label.value)
                return
            i = state["index"]
            ev = filtered[i]
            label.value = f"event {i + 1}/{len(filtered)}: {ev.timestamp}-Tier{ev.tier}"
            try:
                with plt.ioff():
                    fig = plot_event(ev, archive_roots=archive_roots)
                display(fig)
                plt.close(fig)
            except Exception as exc:  # noqa: BLE001
                print(f"plot failed: {type(exc).__name__}: {exc}")

    def _on_filter_change(*_):
        _apply_filter()
        _redraw()

    def _step(delta):
        filtered = state["filtered"]
        if not filtered:
            return
        state["index"] = max(0, min(len(filtered) - 1, state["index"] + delta))
        _redraw()

    tier_toggle.observe(_on_filter_change, names="value")
    sig_slider.observe(_on_filter_change, names="value")
    prev_btn.on_click(lambda _b: _step(-1))
    next_btn.on_click(lambda _b: _step(+1))

    _apply_filter()
    _redraw()

    controls = W.VBox([
        W.HBox([tier_toggle, sig_slider]),
        W.HBox([prev_btn, next_btn, label]),
    ])
    ui = W.VBox([controls, out])
    display(ui)
    return ui
