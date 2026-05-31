#!/usr/bin/env python3
"""
Filename : gaia.py
Author   : ColibriPipeline / lightcurve_analysis (Wave 1, Gaia attach)

Attach Gaia DR2 photometry (``gmag``, ``bp_rp``) to a NightCatalog's per-star
metadata table.

The catalog's ``stars`` DataFrame ships with ``gmag``/``bp_rp`` filled with
NaN. This module queries Gaia once per night (via the reused
``VizieR_query.makeQuery``), cross-matches the catalog's RA/Dec to the Gaia
sources with ``astropy``'s ``match_to_catalog_sky``, and writes the matched
``gmag``/``bp_rp`` back into ``cat.stars`` in place.

Results are cached per night as a small parquet keyed by ``star_id`` so the
(network) Gaia query happens only once. A missing WCS, an empty Gaia result,
or a network/astroquery failure is handled gracefully: ``gmag``/``bp_rp`` are
left as NaN, a clear message is printed, and the catalog is returned unchanged.

``VizieR_query.makeQuery(field_centre, SR)`` returns a pandas DataFrame with
columns ``['RA_ICRS', 'DE_ICRS', 'Gmag', 'BP-RP']`` (Gaia DR2, ``I/345/gaia2``).
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Intra-package modules (always relative when loaded as a package).
if __package__:
    from .catalog import NightCatalog
else:  # flat execution from inside lightcurve_analysis/
    from catalog import NightCatalog  # type: ignore

# Reuse module from the ColibriPipeline directory.
try:
    from .. import VizieR_query as vq
except (ImportError, ValueError):
    import sys as _sys
    from pathlib import Path as _Path
    _pkg_parent = _Path(__file__).resolve().parent.parent
    if str(_pkg_parent) not in _sys.path:
        _sys.path.insert(0, str(_pkg_parent))
    import VizieR_query as vq  # type: ignore


def gaia_cache_path(telescope, obsdate, cache_dir):
    """Return the per-night Gaia parquet cache path.

    Parameters
    ----------
    telescope : str
    obsdate : str
        Hyphenated obsdate (YYYY-MM-DD) as stored on the catalog.
    cache_dir : str or Path

    Returns
    -------
    Path
        ``<cache_dir>/<telescope>_<obsdate>_gaia.parquet``
    """
    return Path(cache_dir) / f"{telescope}_{obsdate}_gaia.parquet"


def query_gaia_for_night(cat, search_radius_deg=0.5):
    """Query Gaia (via VizieR) around the night's field centre.

    The field centre is the NaN-aware median of the catalog's global RA/Dec
    (only stars with a valid WCS solution contribute).

    Parameters
    ----------
    cat : NightCatalog
    search_radius_deg : float
        Cone search radius in degrees.

    Returns
    -------
    pandas.DataFrame
        Normalized columns ``['ra', 'dec', 'gmag', 'bp_rp']``. Empty
        DataFrame (with those columns) if there is no WCS, the query fails,
        or no rows are returned. Never raises.
    """
    empty = pd.DataFrame(columns=['ra', 'dec', 'gmag', 'bp_rp'])

    radec = np.asarray(cat.match.global_radec, dtype=float)
    if radec.size == 0:
        print("[gaia] no global RA/Dec available; skipping Gaia query.")
        return empty

    finite = np.isfinite(radec).all(axis=1)
    if not finite.any():
        print("[gaia] no stars with a WCS solution; skipping Gaia query.")
        return empty

    ra_c = float(np.nanmedian(radec[finite, 0]))
    dec_c = float(np.nanmedian(radec[finite, 1]))
    field_centre = [ra_c, dec_c]
    print(f"[gaia] querying Gaia at field centre "
          f"(RA={ra_c:.5f}, Dec={dec_c:.5f}) deg, "
          f"radius={search_radius_deg} deg")

    try:
        raw = vq.makeQuery(field_centre, search_radius_deg)
    except Exception as exc:  # network / astroquery / parsing failure
        print(f"[gaia] Gaia query failed ({type(exc).__name__}: {exc}); "
              f"leaving gmag/bp_rp as NaN.")
        return empty

    if raw is None or len(raw) == 0:
        print("[gaia] Gaia query returned no rows.")
        return empty

    # Normalize VizieR_query.makeQuery columns (RA_ICRS/DE_ICRS/Gmag/BP-RP)
    # to ['ra', 'dec', 'gmag', 'bp_rp'], tolerant of name variants.
    df = pd.DataFrame(raw)
    colmap = {
        'ra': ['RA_ICRS', 'RAJ2000', 'ra', 'RA'],
        'dec': ['DE_ICRS', 'DEJ2000', 'dec', 'DEC', 'Dec'],
        'gmag': ['Gmag', 'gmag', 'phot_g_mean_mag'],
        'bp_rp': ['BP-RP', 'BP_RP', 'bp_rp', 'BPRP'],
    }
    out = {}
    for target, candidates in colmap.items():
        found = next((c for c in candidates if c in df.columns), None)
        out[target] = (pd.to_numeric(df[found], errors='coerce')
                       if found is not None else np.full(len(df), np.nan))

    norm = pd.DataFrame(out)
    # Need at least sky coords to match against.
    if not np.isfinite(norm[['ra', 'dec']].to_numpy()).all(axis=1).any():
        print("[gaia] Gaia result lacked usable RA/Dec columns.")
        return empty

    print(f"[gaia] Gaia query returned {len(norm)} sources.")
    return norm


def attach_gaia(cat, search_radius_deg=0.5, cache_dir=None, force=False,
                match_arcsec=2.0):
    """Attach Gaia ``gmag``/``bp_rp`` to ``cat.stars`` in place.

    If a per-night Gaia parquet cache exists and ``force`` is False, the
    already-matched per-star values are loaded from it (keyed by ``star_id``).
    Otherwise Gaia is queried once for the night, catalog stars are matched to
    the Gaia sources via great-circle separation (accept ``sep < match_arcsec``
    arcsec), the values are written into ``cat.stars``, and the per-star result
    is saved to the parquet cache.

    No WCS, no Gaia rows, or a network/query failure all leave
    ``gmag``/``bp_rp`` as NaN, print a clear message, and return ``cat``
    unchanged.

    Parameters
    ----------
    cat : NightCatalog
    search_radius_deg : float
        Gaia cone search radius (degrees).
    cache_dir : str or Path, optional
        Directory for the Gaia parquet cache. Defaults to ``cat.cache_dir``,
        falling back to the catalog's night directory if that is None.
    force : bool
        If True, re-query Gaia even when a cache exists.
    match_arcsec : float
        Maximum on-sky separation for a catalog<->Gaia match (arcsec).

    Returns
    -------
    NightCatalog
        The same object, with ``cat.stars`` mutated in place.
    """
    # Resolve cache directory.
    if cache_dir is None:
        cache_dir = cat.cache_dir
    if cache_dir is None:
        # fall back to the night directory under the archive root.
        if cat.archive_root is not None:
            cache_dir = Path(cat.archive_root) / cat.obsdate
        else:
            cache_dir = Path('.')
    cache_dir = Path(cache_dir)

    cache_path = gaia_cache_path(cat.telescope, cat.obsdate, cache_dir)

    # --- cache path -------------------------------------------------------
    if cache_path.exists() and not force:
        try:
            cached = pd.read_parquet(cache_path)
            _apply_per_star(cat, cached)
            n = int(np.isfinite(cat.stars['gmag']).sum())
            print(f"[gaia] loaded cached Gaia photometry from {cache_path} "
                  f"({n} stars with gmag).")
            return cat
        except Exception as exc:
            print(f"[gaia] failed to read cache {cache_path} "
                  f"({type(exc).__name__}: {exc}); re-querying.")

    # --- query path -------------------------------------------------------
    gaia_df = query_gaia_for_night(cat, search_radius_deg=search_radius_deg)
    if gaia_df is None or len(gaia_df) == 0:
        print("[gaia] no Gaia photometry attached; gmag/bp_rp remain NaN.")
        return cat

    # Catalog stars with valid sky coordinates.
    star_radec = cat.stars[['ra_deg', 'dec_deg']].to_numpy(dtype=float)
    valid = np.isfinite(star_radec).all(axis=1)
    if not valid.any():
        print("[gaia] no catalog stars with valid RA/Dec to match; "
              "gmag/bp_rp remain NaN.")
        return cat

    gaia_radec = gaia_df[['ra', 'dec']].to_numpy(dtype=float)
    gaia_valid = np.isfinite(gaia_radec).all(axis=1)
    if not gaia_valid.any():
        print("[gaia] no Gaia sources with valid RA/Dec; gmag/bp_rp NaN.")
        return cat
    gaia_df = gaia_df[gaia_valid].reset_index(drop=True)
    gaia_radec = gaia_radec[gaia_valid]

    try:
        from astropy.coordinates import SkyCoord
        from astropy import units as u
    except ImportError as exc:
        print(f"[gaia] astropy unavailable ({exc}); gmag/bp_rp remain NaN.")
        return cat

    cat_coords = SkyCoord(ra=star_radec[valid, 0] * u.deg,
                          dec=star_radec[valid, 1] * u.deg)
    gaia_coords = SkyCoord(ra=gaia_radec[:, 0] * u.deg,
                           dec=gaia_radec[:, 1] * u.deg)

    idx, sep2d, _ = cat_coords.match_to_catalog_sky(gaia_coords)
    accept = sep2d.arcsec < match_arcsec

    # Assemble matched per-star values aligned to the valid catalog stars.
    gmag_v = np.full(int(valid.sum()), np.nan)
    bprp_v = np.full(int(valid.sum()), np.nan)
    matched_idx = np.asarray(idx)[accept]
    gmag_v[accept] = gaia_df['gmag'].to_numpy()[matched_idx]
    bprp_v[accept] = gaia_df['bp_rp'].to_numpy()[matched_idx]

    # Write back into the full-length columns (NaN where no valid coord/match).
    gmag_full = np.full(len(cat.stars), np.nan)
    bprp_full = np.full(len(cat.stars), np.nan)
    valid_pos = np.flatnonzero(valid)
    gmag_full[valid_pos] = gmag_v
    bprp_full[valid_pos] = bprp_v

    cat.stars['gmag'] = gmag_full
    cat.stars['bp_rp'] = bprp_full

    n_matched = int(accept.sum())
    print(f"[gaia] matched {n_matched} catalog stars to Gaia "
          f"(< {match_arcsec} arcsec).")

    # --- save per-star cache ---------------------------------------------
    per_star = pd.DataFrame({
        'star_id': cat.stars['star_id'].to_numpy(),
        'gmag': gmag_full,
        'bp_rp': bprp_full,
    })
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        per_star.to_parquet(cache_path, index=False)
        print(f"[gaia] saved Gaia photometry cache to {cache_path}.")
    except Exception as exc:
        print(f"[gaia] failed to write cache {cache_path} "
              f"({type(exc).__name__}: {exc}); continuing.")

    return cat


def _apply_per_star(cat, cached):
    """Merge a cached per-star (star_id, gmag, bp_rp) frame into cat.stars."""
    lut = cached.set_index('star_id')
    ids = cat.stars['star_id'].to_numpy()
    gmag = lut['gmag'].reindex(ids).to_numpy()
    bprp = lut['bp_rp'].reindex(ids).to_numpy()
    cat.stars['gmag'] = gmag
    cat.stars['bp_rp'] = bprp
