"""Headless tests for detection.viewer.

Builds a tiny synthetic ``matched/<ts>-Tier2/`` fixture with two det_*.txt files
(GREENBIRD + BLUEBIRD) plus matching small ``<minute_key>_stars.npy`` files, and
exercises load_matched_events / event_lightcurves / plot_event. The ipywidgets
path is guarded so the suite passes without ipywidgets installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")  # non-interactive backend before pyplot is imported

from detection.viewer import (
    MatchedEvent,
    load_matched_events,
    event_lightcurves,
    plot_event,
    explore_matches,
)


MINUTE_KEY = "20250830_01.54.12.147"
ISO = "2025-08-30T01:54:38.000000"
N_FRAMES = 200
N_STARS = 5
STAR_COL = 3  # the event star's column in the stars.npy
EVENT_X, EVENT_Y = 687.4, 1199.4


def _write_det(path: Path, telescope: str, star_idx: int, ra: float, dec: float,
               sig: float):
    """Write a det_*.txt with the exact header line indices viewer parses."""
    lines = [
        "#", "#", "#", "#",
        f"#    Event File: {MINUTE_KEY}_1047.rcd",          # idx 4
        f"#    Star Coords: {EVENT_X:f} {EVENT_Y:f}",        # idx 5
        f"#    RA Dec Coords: {ra:f} {dec:f}",               # idx 6
        f"#    DATE-OBS: {ISO}",                             # idx 7
        f"#    Telescope: {telescope}",                      # idx 8
        "#    Field: unknown",                              # idx 9
        f"#    significance: {sig:.3f}",                     # idx 10
        "#    Raw lightcurve std: 1253.5044",               # idx 11
        "#    Raw lightcurve mean: 7378.2602",              # idx 12
        "#    Convolution background std: 1.4107",          # idx 13
        "#    Convolution background mean: -0.0277",        # idx 14
        "#    Convolution minimal value: 7.3062",           # idx 15
        "#", "#",                                            # idx 16, 17
        "#filename     time      flux     conv_flux",        # idx 18
    ]
    rng = np.random.default_rng(star_idx)
    for f in range(40):  # ~40 data rows
        flux = 7000.0 + rng.normal(0, 50)
        conv = rng.normal(0, 1)
        lines.append(f"{MINUTE_KEY}_{1007 + f}.rcd {37.0:f}  {flux:f}  {conv:f}")
    path.write_text("\n".join(lines) + "\n")


def _write_stars_npy(night_dir: Path):
    """Write a small <minute_key>_stars.npy with the event star at STAR_COL."""
    night_dir.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((N_FRAMES, N_STARS, 4), dtype=np.float64)
    # frame-0 xy for each star; put a unique xy per star, event star at STAR_COL.
    for s in range(N_STARS):
        arr[:, s, 0] = 100.0 * s + 10.0
        arr[:, s, 1] = 200.0 * s + 20.0
    arr[:, STAR_COL, 0] = EVENT_X
    arr[:, STAR_COL, 1] = EVENT_Y
    # flux + time axes
    rng = np.random.default_rng(0)
    arr[:, :, 2] = 7000.0 + rng.normal(0, 50, size=(N_FRAMES, N_STARS))
    arr[:, :, 3] = (np.arange(N_FRAMES, dtype=np.float64) * 0.025)[:, None]
    np.save(night_dir / f"{MINUTE_KEY}_stars.npy", arr)


@pytest.fixture
def fixture(tmp_path):
    """Build matched dir + per-scope fake night archives."""
    matched = tmp_path / "matched"
    evt_dir = matched / "2025-08-29_215438_000000-Tier2"
    evt_dir.mkdir(parents=True)

    _write_det(evt_dir / f"det_2025-08-30_015438_000000_star{STAR_COL}_GREENBIRD.txt",
               "GREENBIRD", STAR_COL, 273.977630, -18.523206, 5.199)
    _write_det(evt_dir / f"det_2025-08-30_015438_000000_star{STAR_COL}_BLUEBIRD.txt",
               "BLUEBIRD", STAR_COL, 273.977700, -18.523100, 4.812)

    green_night = tmp_path / "green"
    blue_night = tmp_path / "blue"
    _write_stars_npy(green_night)
    _write_stars_npy(blue_night)

    return {
        "matched": matched,
        "archive_roots": {"GREENBIRD": green_night, "BLUEBIRD": blue_night},
    }


def test_load_matched_events(fixture):
    events = load_matched_events(fixture["matched"])
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, MatchedEvent)
    assert ev.tier == 2
    assert ev.timestamp == "2025-08-29_215438_000000"
    assert set(ev.per_scope) == {"GREENBIRD", "BLUEBIRD"}

    g = ev.per_scope["GREENBIRD"]
    assert g["star_idx"] == STAR_COL
    assert abs(g["ra"] - 273.977630) < 1e-4
    assert abs(g["dec"] - (-18.523206)) < 1e-4
    assert abs(g["significance"] - 5.199) < 1e-3
    assert g["minute_key"] == "20250830_01.54.38.000"
    assert g["stars_key"] == MINUTE_KEY
    assert g["det_flux"].size == 40

    b = ev.per_scope["BLUEBIRD"]
    assert abs(b["significance"] - 4.812) < 1e-3


def test_event_lightcurves_cross_index(fixture):
    ev = load_matched_events(fixture["matched"])[0]
    curves = event_lightcurves(ev, archive_roots=fixture["archive_roots"])

    for scope in ("GREENBIRD", "BLUEBIRD"):
        c = curves[scope]
        assert c["star_col"] == STAR_COL
        assert c["flux"].shape == (N_FRAMES,)
        assert np.all(np.isfinite(c["flux"]))
        assert c["t"].shape == (N_FRAMES,)


def test_event_lightcurves_fallback_no_archive(fixture):
    ev = load_matched_events(fixture["matched"])[0]
    curves = event_lightcurves(ev, archive_roots={})  # no stars.npy locatable
    g = curves["GREENBIRD"]
    assert g["star_col"] is None
    # falls back to det window
    assert g["flux"].size == 40
    assert np.array_equal(g["flux"], g["det_flux"])


def test_plot_event_returns_figure(fixture):
    from matplotlib.figure import Figure
    ev = load_matched_events(fixture["matched"])[0]
    fig = plot_event(ev, archive_roots=fixture["archive_roots"])
    assert isinstance(fig, Figure)


def test_explore_matches(fixture):
    try:
        import ipywidgets  # noqa: F401
    except Exception:
        pytest.skip("ipywidgets not installed")
    ui = explore_matches(fixture["matched"], archive_roots=fixture["archive_roots"])
    assert ui is not None
