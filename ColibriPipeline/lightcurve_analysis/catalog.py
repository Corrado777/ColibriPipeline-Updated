#!/usr/bin/env python3
"""
Filename : catalog.py
Author   : ColibriPipeline / lightcurve_analysis (Wave 1)

The NightCatalog data model and its builder. A NightCatalog ties together:
    * the per-minute references (io.MinuteRef),
    * the cross-minute global match table (matching.GlobalMatchTable),
    * a per-star metadata DataFrame,
    * and a stitched, sparse-per-star, mmap-friendly flux/time payload on a
      common night time axis.

Storage of the stitched payload is "sparse-per-star ragged": each global star
has only its valid samples stored, concatenated end-to-end with a per-star
offset index, so long nights do not require GB-scale NaN-padded dense
matrices. Dense views (NaN in gaps) are reconstructed on demand via
``get_raw_lightcurve`` / ``get_flux_matrix``.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Intra-package modules (always relative when loaded as a package).
if __package__:
    from . import io as _io
    from . import matching as _matching
    from . import paths as _paths
    from . import stitch as _stitch
    from .stitch import LightCurve
    from .io import MinuteRef
    from .matching import GlobalMatchTable
else:  # flat execution from inside lightcurve_analysis/
    import io as _io  # type: ignore
    import matching as _matching  # type: ignore
    import paths as _paths  # type: ignore
    import stitch as _stitch  # type: ignore
    from stitch import LightCurve  # type: ignore
    from io import MinuteRef  # type: ignore
    from matching import GlobalMatchTable  # type: ignore

# Reuse module from the sibling detection_trade_study package.
try:
    from ..detection_trade_study import lightcurves as lc
except (ImportError, ValueError):
    import sys as _sys
    from pathlib import Path as _Path
    _pkg_parent = _Path(__file__).resolve().parent.parent
    for _p in (str(_pkg_parent), str(_pkg_parent / 'detection_trade_study')):
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
    import lightcurves as lc


_STARS_COLUMNS = [
    'star_id', 'ra_deg', 'dec_deg', 'x', 'y',
    'n_minutes', 'n_valid_frames', 'median_flux', 'snr',
    'gmag', 'bp_rp', 'field',
]


@dataclass
class NightCatalog:
    """A whole night's stars and their stitched light curves."""
    telescope: str
    obsdate: str
    minutes: list                  # list[MinuteRef]
    match: GlobalMatchTable
    stars: pd.DataFrame            # per-star metadata, schema in _STARS_COLUMNS

    # stitched payload on the common night time axis
    time_axis: np.ndarray          # (n_samples,)
    minute_bounds: np.ndarray      # (n_minutes + 1,)

    # sparse-per-star ragged storage
    flux_concat: np.ndarray        # (total_valid,) float32
    time_concat: np.ndarray        # (total_valid,) float64
    star_offsets: np.ndarray       # (n_global + 1,) int64

    archive_root: Path | None = None
    cache_dir: Path | None = None

    # -- per-star extraction ------------------------------------------------

    def _star_slice(self, star_id):
        lo = int(self.star_offsets[star_id])
        hi = int(self.star_offsets[star_id + 1])
        return slice(lo, hi)

    def get_raw_lightcurve(self, star_id, with_gaps=True):
        """Return a star's raw light curve on the common time axis.

        Parameters
        ----------
        star_id : int
        with_gaps : bool
            If True (default), the returned curve spans the full night time
            axis with NaN at samples where this star had no valid flux (so
            gaps and missing minutes show as breaks). If False, only the
            star's valid samples are returned.

        Returns
        -------
        LightCurve
        """
        sl = self._star_slice(int(star_id))
        st = self.time_concat[sl]
        sf = self.flux_concat[sl].astype(float)
        if not with_gaps:
            return LightCurve(t=np.asarray(st, dtype=float), f=sf,
                              kind='raw', star_id=int(star_id))
        # place onto the common time axis (NaN elsewhere)
        full = np.full(self.time_axis.shape[0], np.nan)
        if len(st) > 0:
            pos = np.searchsorted(self.time_axis, st)
            pos = np.clip(pos, 0, self.time_axis.shape[0] - 1)
            full[pos] = sf
        return LightCurve(t=np.asarray(self.time_axis, dtype=float), f=full,
                          kind='raw', star_id=int(star_id))

    def get_binned_lightcurve(self, star_id, bin_seconds):
        """Return a binned light curve for a star (nan-aware mean per bin)."""
        raw = self.get_raw_lightcurve(int(star_id), with_gaps=False)
        centers, binned = _stitch.bin_lightcurve(raw.t, raw.f, bin_seconds,
                                                  how='mean')
        return LightCurve(t=centers, f=binned, kind='binned',
                          bin_seconds=bin_seconds, star_id=int(star_id))

    def get_flux_matrix(self, star_ids, time_grid=None):
        """Dense ``(len(star_ids), n_samples)`` flux matrix, NaN in gaps.

        Reconstructs dense rows from the sparse ragged storage on the common
        time axis (or on ``time_grid`` if given). This is the view
        differential photometry needs for reference-star math.

        Parameters
        ----------
        star_ids : sequence[int]
        time_grid : np.ndarray, optional
            Target time axis. Defaults to ``self.time_axis``.

        Returns
        -------
        np.ndarray, shape (len(star_ids), n_samples), float64 (NaN-filled).
        """
        grid = self.time_axis if time_grid is None else np.asarray(time_grid,
                                                                    dtype=float)
        n_samples = grid.shape[0]
        out = np.full((len(star_ids), n_samples), np.nan)
        for r, sid in enumerate(star_ids):
            sl = self._star_slice(int(sid))
            st = self.time_concat[sl]
            sf = self.flux_concat[sl].astype(float)
            if len(st) == 0:
                continue
            pos = np.searchsorted(grid, st)
            pos = np.clip(pos, 0, n_samples - 1)
            out[r, pos] = sf
        return out

    # -- selection ----------------------------------------------------------

    def filter_stars(self, *, gmag=None, snr_min=None, ra_range=None,
                     dec_range=None, x_range=None, y_range=None,
                     min_minutes=None, field=None):
        """Return a filtered view of the ``stars`` DataFrame.

        All criteria combine with AND. ``gmag``/``ra_range``/``dec_range``/
        ``x_range``/``y_range`` are (lo, hi) tuples.
        """
        df = self.stars
        mask = pd.Series(True, index=df.index)
        if gmag is not None:
            lo, hi = gmag
            mask &= df['gmag'].between(lo, hi)
        if snr_min is not None:
            mask &= df['snr'] >= snr_min
        if ra_range is not None:
            lo, hi = ra_range
            mask &= df['ra_deg'].between(lo, hi)
        if dec_range is not None:
            lo, hi = dec_range
            mask &= df['dec_deg'].between(lo, hi)
        if x_range is not None:
            lo, hi = x_range
            mask &= df['x'].between(lo, hi)
        if y_range is not None:
            lo, hi = y_range
            mask &= df['y'].between(lo, hi)
        if min_minutes is not None:
            mask &= df['n_minutes'] >= min_minutes
        if field is not None:
            mask &= df['field'] == field
        return df[mask].copy()

    def nearest_star(self, ra=None, dec=None, x=None, y=None):
        """Return the global star id nearest to the given coordinates.

        Prefers RA/Dec (great-circle separation) when both ``ra`` and ``dec``
        are given and the catalog has sky coords; else uses pixel x/y.
        """
        df = self.stars
        if ra is not None and dec is not None:
            radec = df[['ra_deg', 'dec_deg']].to_numpy()
            if np.isfinite(radec).any():
                from astropy.coordinates import SkyCoord
                from astropy import units as u
                valid = np.isfinite(radec).all(axis=1)
                cand = df[valid]
                cat = SkyCoord(ra=cand['ra_deg'].to_numpy() * u.deg,
                               dec=cand['dec_deg'].to_numpy() * u.deg)
                target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
                idx, _, _ = target.match_to_catalog_sky(cat)
                return int(cand.iloc[int(idx)]['star_id'])
        if x is not None and y is not None:
            d = np.hypot(df['x'].to_numpy() - x, df['y'].to_numpy() - y)
            return int(df.iloc[int(np.argmin(d))]['star_id'])
        raise ValueError("Provide ra+dec or x+y.")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_night_catalog(telescope, obsdate, *, archive_root=None,
                        sim_root=None, tol_arcsec=2.0, tol_px=2.0,
                        min_minutes=1, cache_dir=None, rebuild=False,
                        storage='sparse'):
    """Build (or load cached) a NightCatalog for one telescope/night.

    Parameters
    ----------
    telescope : str
        Telescope label (REDBIRD/GREENBIRD/BLUEBIRD or color).
    obsdate : str
        YYYYMMDD or YYYY-MM-DD.
    archive_root, sim_root : str or Path, optional
        Path resolution overrides (see paths.resolve_archive_root).
    tol_arcsec, tol_px : float
        Cross-minute match tolerances.
    min_minutes : int
        Drop global stars detected in fewer than this many minutes.
    cache_dir : str or Path, optional
        Cache directory. Default: ``<night_dir>/lightcurve_cache``.
    rebuild : bool
        Force a rebuild even if a cache exists.
    storage : {'sparse'}
        Reserved; only 'sparse' is implemented (dense is reconstructed on
        demand via get_flux_matrix).

    Returns
    -------
    NightCatalog
    """
    root = _paths.resolve_archive_root(archive_root, telescope, sim_root)
    ndir = _paths.night_dir(root, obsdate)
    obs_hyph = _paths.normalize_obsdate(obsdate)

    resolved_cache = Path(cache_dir) if cache_dir is not None else (
        ndir / 'lightcurve_cache')

    # try cache
    if not rebuild and _cache_exists(telescope, obs_hyph, resolved_cache):
        return load_cache(telescope, obs_hyph, cache_dir=resolved_cache)

    minutes = _io.discover_minutes(ndir)
    if not minutes:
        raise FileNotFoundError(f"No *_stars.npy files found in {ndir}")

    match = _matching.match_minutes_to_global(minutes, tol_arcsec=tol_arcsec,
                                              tol_px=tol_px)
    n_global = match.n_global

    # collect per-minute time vectors for the time axis (one mmap each)
    minute_times = []
    for ref in minutes:
        arr = np.load(ref.stars_path, mmap_mode='r')
        minute_times.append(np.asarray(arr[:, 0, 3]))
        del arr
    time_axis, minute_bounds = _stitch.build_time_axis(minutes, minute_times)

    # accumulate sparse-per-star samples
    flux_lists = [[] for _ in range(n_global)]
    time_lists = [[] for _ in range(n_global)]
    n_valid_frames = np.zeros(n_global, dtype=np.int64)

    for m_idx, ref in enumerate(minutes):
        data = _io.load_minute_full(ref, mmap=True)
        flux = data['flux']                      # (n_local, n_frames)
        seg_t = time_axis[minute_bounds[m_idx]:minute_bounds[m_idx + 1]]
        l2g = match.per_minute[ref.minute_key]

        for local_idx in range(flux.shape[0]):
            gid = int(l2g[local_idx])
            if gid < 0:
                continue
            raw = flux[local_idx]
            cleaned = lc.clean(raw)
            if cleaned is None:
                continue
            # lc.clean trims leading/trailing zeros; align the time axis to the
            # trimmed flux by locating it within the raw row.
            ct_seg, cf_seg = _aligned_segment(raw, cleaned, seg_t)
            if cf_seg is None:
                continue
            flux_lists[gid].append(cf_seg.astype(np.float32))
            time_lists[gid].append(ct_seg.astype(np.float64))
            n_valid_frames[gid] += len(cf_seg)

    # build ragged concat arrays
    flux_concat, time_concat, star_offsets = _ragged_pack(flux_lists,
                                                           time_lists, n_global)

    # per-star summaries
    median_flux = np.full(n_global, np.nan)
    snr = np.zeros(n_global)
    for gid in range(n_global):
        lo = int(star_offsets[gid])
        hi = int(star_offsets[gid + 1])
        if hi > lo:
            seg = flux_concat[lo:hi].astype(float)
            median_flux[gid] = np.nanmedian(seg)
            snr[gid] = lc.star_snr(seg, window=200)

    stars = _build_stars_df(match, n_valid_frames, median_flux, snr)

    cat = NightCatalog(
        telescope=telescope,
        obsdate=obs_hyph,
        minutes=minutes,
        match=match,
        stars=stars,
        time_axis=time_axis,
        minute_bounds=minute_bounds,
        flux_concat=flux_concat,
        time_concat=time_concat,
        star_offsets=star_offsets,
        archive_root=root,
        cache_dir=resolved_cache,
    )

    # apply min_minutes filter to the visible stars table (keep storage intact)
    if min_minutes > 1:
        cat.stars = cat.stars[cat.stars['n_minutes'] >= min_minutes].copy()

    save_cache(cat, resolved_cache)
    return cat


def _aligned_segment(raw, cleaned, seg_t):
    """Map a trimmed `cleaned` array back onto the raw row's time axis.

    lc.clean uses np.trim_zeros, removing leading/trailing zeros. We find the
    sub-window of the raw row corresponding to `cleaned` and slice seg_t to
    match. Returns (time_segment, flux_segment) or (None, None) if it cannot
    be located (length mismatch safety).
    """
    raw = np.asarray(raw, dtype=float)
    n_clean = len(cleaned)
    if n_clean == 0 or n_clean > len(raw):
        return None, None
    # number of leading zeros trimmed
    nonzero = np.flatnonzero(raw != 0)
    if nonzero.size == 0:
        return None, None
    start = int(nonzero[0])
    end = start + n_clean
    if end > len(raw) or end > len(seg_t):
        return None, None
    return seg_t[start:end], np.asarray(cleaned, dtype=float)


def _ragged_pack(flux_lists, time_lists, n_global):
    """Concatenate per-star lists into ragged arrays + offset index."""
    offsets = np.zeros(n_global + 1, dtype=np.int64)
    flux_parts = []
    time_parts = []
    for gid in range(n_global):
        if flux_lists[gid]:
            f = np.concatenate(flux_lists[gid])
            t = np.concatenate(time_lists[gid])
        else:
            f = np.zeros(0, dtype=np.float32)
            t = np.zeros(0, dtype=np.float64)
        flux_parts.append(f)
        time_parts.append(t)
        offsets[gid + 1] = offsets[gid] + len(f)
    flux_concat = (np.concatenate(flux_parts).astype(np.float32)
                   if flux_parts else np.zeros(0, dtype=np.float32))
    time_concat = (np.concatenate(time_parts).astype(np.float64)
                   if time_parts else np.zeros(0, dtype=np.float64))
    return flux_concat, time_concat, offsets


def _build_stars_df(match, n_valid_frames, median_flux, snr, field='field0'):
    """Assemble the per-star metadata DataFrame."""
    n = match.n_global
    df = pd.DataFrame({
        'star_id': np.arange(n, dtype=np.int64),
        'ra_deg': match.global_radec[:, 0] if n else np.zeros(0),
        'dec_deg': match.global_radec[:, 1] if n else np.zeros(0),
        'x': match.global_xy[:, 0] if n else np.zeros(0),
        'y': match.global_xy[:, 1] if n else np.zeros(0),
        'n_minutes': match.n_detections,
        'n_valid_frames': n_valid_frames,
        'median_flux': median_flux,
        'snr': snr,
        'gmag': np.full(n, np.nan),
        'bp_rp': np.full(n, np.nan),
        'field': [field] * n,
    })
    return df[_STARS_COLUMNS]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_paths(telescope, obsdate, cache_dir):
    cache_dir = Path(cache_dir)
    stem = f"{telescope}_{obsdate}"
    return (
        cache_dir / f"{stem}_lightcurves.npz",
        cache_dir / f"{stem}_stars.parquet",
        cache_dir / f"{stem}_minutes.json",
    )


def _cache_exists(telescope, obsdate, cache_dir):
    npz, parquet, minutes_json = _cache_paths(telescope, obsdate, cache_dir)
    return npz.exists() and parquet.exists() and minutes_json.exists()


def save_cache(cat, cache_dir):
    """Persist a NightCatalog to disk (npz + parquet + json sidecars)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz, parquet, minutes_json = _cache_paths(cat.telescope, cat.obsdate,
                                              cache_dir)

    np.savez_compressed(
        npz,
        time_axis=cat.time_axis,
        minute_bounds=cat.minute_bounds,
        flux_concat=cat.flux_concat.astype(np.float32),
        time_concat=cat.time_concat,
        star_offsets=cat.star_offsets,
        global_radec=cat.match.global_radec,
        global_xy=cat.match.global_xy,
        n_detections=cat.match.n_detections,
    )

    cat.stars.to_parquet(parquet, index=False)

    minutes_payload = {
        'telescope': cat.telescope,
        'obsdate': cat.obsdate,
        'minutes': [
            {
                'minute_key': r.minute_key,
                't_start': r.t_start,
                'stars_path': str(r.stars_path),
                'pos_path': None if r.pos_path is None else str(r.pos_path),
                'wcs_path': None if r.wcs_path is None else str(r.wcs_path),
            }
            for r in cat.minutes
        ],
        'per_minute': {
            k: np.asarray(v, dtype=np.int64).tolist()
            for k, v in cat.match.per_minute.items()
        },
    }
    minutes_json.write_text(json.dumps(minutes_payload))


def load_cache(telescope, obsdate, cache_dir=None, archive_root=None,
               sim_root=None):
    """Load a cached NightCatalog.

    Parameters
    ----------
    telescope : str
    obsdate : str
        YYYYMMDD or YYYY-MM-DD.
    cache_dir : str or Path, optional
        If None, defaults to ``<night_dir>/lightcurve_cache`` resolved via the
        same path precedence as the builder.
    archive_root, sim_root : str or Path, optional
        Used only to resolve the default cache_dir when cache_dir is None.

    Returns
    -------
    NightCatalog
    """
    obs_hyph = _paths.normalize_obsdate(obsdate)
    root = None
    if cache_dir is None:
        root = _paths.resolve_archive_root(archive_root, telescope, sim_root)
        ndir = _paths.night_dir(root, obs_hyph)
        cache_dir = ndir / 'lightcurve_cache'
    cache_dir = Path(cache_dir)

    npz_path, parquet, minutes_json = _cache_paths(telescope, obs_hyph,
                                                   cache_dir)
    if not (npz_path.exists() and parquet.exists() and minutes_json.exists()):
        raise FileNotFoundError(
            f"No cache for {telescope} {obs_hyph} in {cache_dir}")

    npz = np.load(npz_path, mmap_mode='r')
    time_axis = np.asarray(npz['time_axis'])
    minute_bounds = np.asarray(npz['minute_bounds'])
    flux_concat = npz['flux_concat']          # stays mmap-backed
    time_concat = np.asarray(npz['time_concat'])
    star_offsets = np.asarray(npz['star_offsets'])
    global_radec = np.asarray(npz['global_radec'])
    global_xy = np.asarray(npz['global_xy'])
    n_detections = np.asarray(npz['n_detections'])

    stars = pd.read_parquet(parquet)

    payload = json.loads(minutes_json.read_text())
    minutes = [
        MinuteRef(
            minute_key=m['minute_key'],
            t_start=m['t_start'],
            stars_path=Path(m['stars_path']),
            pos_path=None if m['pos_path'] is None else Path(m['pos_path']),
            wcs_path=None if m['wcs_path'] is None else Path(m['wcs_path']),
        )
        for m in payload['minutes']
    ]
    per_minute = {k: np.asarray(v, dtype=np.int64)
                  for k, v in payload['per_minute'].items()}

    match = GlobalMatchTable(
        n_global=len(star_offsets) - 1,
        per_minute=per_minute,
        global_radec=global_radec,
        global_xy=global_xy,
        n_detections=n_detections,
    )

    return NightCatalog(
        telescope=telescope,
        obsdate=obs_hyph,
        minutes=minutes,
        match=match,
        stars=stars,
        time_axis=time_axis,
        minute_bounds=minute_bounds,
        flux_concat=flux_concat,
        time_concat=time_concat,
        star_offsets=star_offsets,
        archive_root=root,
        cache_dir=cache_dir,
    )
