#!/usr/bin/env python3
"""
Filename : matching.py
Author   : ColibriPipeline / lightcurve_analysis (Wave 1)

Cross-minute star matching -> stable global IDs.

A star's index is consistent within a minute but not across minutes (stars are
detected fresh each minute). We build a running global catalog by matching each
new minute's stars to the catalog so far:
    * RA/Dec match (astropy SkyCoord.match_to_catalog_sky, sep < tol_arcsec)
      when BOTH the minute and the running global have sky coordinates;
    * else pixel x/y match (scipy.spatial.cKDTree, dist < tol_px).
Unmatched minute stars append as new global stars. Global IDs are assigned in
append order, which is deterministic across runs.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

if __package__:
    from . import io as _io
else:  # flat execution from inside lightcurve_analysis/
    import io as _io  # type: ignore


@dataclass
class GlobalMatchTable:
    """Mapping from per-minute local star indices to stable global IDs."""
    n_global: int
    per_minute: dict          # minute_key -> np.ndarray[local_idx] = global_id
    global_radec: np.ndarray  # (n_global, 2), NaN if never had WCS
    global_xy: np.ndarray     # (n_global, 2), running-mean representative pixel
    n_detections: np.ndarray  # (n_global,) number of minutes each star appears


def match_minutes_to_global(minutes, tol_arcsec=2.0, tol_px=2.0):
    """Incrementally match per-minute stars to a running global catalog.

    Parameters
    ----------
    minutes : list[MinuteRef]
    tol_arcsec : float
        Sky-match tolerance when both sides have RA/Dec.
    tol_px : float
        Pixel-match tolerance for the x/y fallback.

    Returns
    -------
    GlobalMatchTable
    """
    # running global accumulators (as python lists for cheap appends)
    g_ra, g_dec = [], []          # NaN where no sky coords
    g_x, g_y = [], []             # running sums for the running mean
    g_xy_n = []                   # counts behind the running mean
    g_radec_n = []                # counts of valid radec contributions
    g_ndet = []                   # detection counts
    per_minute = {}

    for ref in minutes:
        xy = _io.minute_xy(ref)              # (n_local, 2)
        radec = _io.minute_radec(ref)        # (n_local, 2) or None
        n_local = xy.shape[0]
        local_to_global = np.full(n_local, -1, dtype=np.int64)

        n_global = len(g_x)

        if n_global == 0:
            # first minute: everything is new
            for i in range(n_local):
                local_to_global[i] = _append_global(
                    g_ra, g_dec, g_x, g_y, g_xy_n, g_radec_n, g_ndet,
                    xy[i], None if radec is None else radec[i])
            per_minute[ref.minute_key] = local_to_global
            continue

        # decide match mode
        gl_radec = np.column_stack([np.asarray(g_ra), np.asarray(g_dec)])
        have_global_sky = np.isfinite(gl_radec).all(axis=1)
        use_sky = (radec is not None) and bool(have_global_sky.any())

        matched_local = np.zeros(n_local, dtype=bool)
        # global indices already claimed this minute (avoid double-assign)
        claimed = np.zeros(n_global, dtype=bool)

        if use_sky:
            sky_global_idx = np.where(have_global_sky)[0]
            local_sky_mask = np.isfinite(radec).all(axis=1)
            local_sky_idx = np.where(local_sky_mask)[0]
            if len(sky_global_idx) > 0 and len(local_sky_idx) > 0:
                pairs = _match_sky(
                    radec[local_sky_idx], gl_radec[sky_global_idx], tol_arcsec)
                for li, gi in pairs:
                    g = sky_global_idx[gi]
                    if claimed[g]:
                        continue
                    l = local_sky_idx[li]
                    claimed[g] = True
                    matched_local[l] = True
                    local_to_global[l] = g
                    _update_global(
                        g_ra, g_dec, g_x, g_y, g_xy_n, g_radec_n, g_ndet,
                        g, xy[l], radec[l])

        # pixel fallback for everything not yet matched
        unmatched_local = np.where(~matched_local)[0]
        if len(unmatched_local) > 0:
            free_global = np.where(~claimed)[0]
            if len(free_global) > 0:
                gxy = np.column_stack(
                    [np.asarray(g_x)[free_global] / np.asarray(g_xy_n)[free_global],
                     np.asarray(g_y)[free_global] / np.asarray(g_xy_n)[free_global]])
                tree = cKDTree(gxy)
                dist, idx = tree.query(xy[unmatched_local], k=1)
                for k, l in enumerate(unmatched_local):
                    if dist[k] < tol_px:
                        g = free_global[idx[k]]
                        if claimed[g]:
                            continue
                        claimed[g] = True
                        matched_local[l] = True
                        local_to_global[l] = g
                        _update_global(
                            g_ra, g_dec, g_x, g_y, g_xy_n, g_radec_n, g_ndet,
                            g, xy[l],
                            None if radec is None else radec[l])

        # append remaining unmatched as new global stars
        still = np.where(~matched_local)[0]
        for l in still:
            local_to_global[l] = _append_global(
                g_ra, g_dec, g_x, g_y, g_xy_n, g_radec_n, g_ndet,
                xy[l], None if radec is None else radec[l])

        per_minute[ref.minute_key] = local_to_global

    n_global = len(g_x)
    xy_n = np.asarray(g_xy_n, dtype=float)
    global_xy = np.column_stack([
        np.asarray(g_x) / np.where(xy_n == 0, 1, xy_n),
        np.asarray(g_y) / np.where(xy_n == 0, 1, xy_n),
    ]) if n_global else np.zeros((0, 2))
    radec_n = np.asarray(g_radec_n, dtype=float)
    ra_arr = np.asarray(g_ra, dtype=float)
    dec_arr = np.asarray(g_dec, dtype=float)
    global_radec = np.column_stack([
        np.where(radec_n > 0, ra_arr / np.where(radec_n == 0, 1, radec_n), np.nan),
        np.where(radec_n > 0, dec_arr / np.where(radec_n == 0, 1, radec_n), np.nan),
    ]) if n_global else np.zeros((0, 2))

    return GlobalMatchTable(
        n_global=n_global,
        per_minute=per_minute,
        global_radec=global_radec,
        global_xy=global_xy,
        n_detections=np.asarray(g_ndet, dtype=np.int64),
    )


def _append_global(g_ra, g_dec, g_x, g_y, g_xy_n, g_radec_n, g_ndet, xy, radec):
    """Append a new global star, return its global id."""
    gid = len(g_x)
    g_x.append(float(xy[0]))
    g_y.append(float(xy[1]))
    g_xy_n.append(1.0)
    if radec is not None and np.isfinite(radec).all():
        g_ra.append(float(radec[0]))
        g_dec.append(float(radec[1]))
        g_radec_n.append(1.0)
    else:
        g_ra.append(0.0)
        g_dec.append(0.0)
        g_radec_n.append(0.0)
    g_ndet.append(1)
    return gid


def _update_global(g_ra, g_dec, g_x, g_y, g_xy_n, g_radec_n, g_ndet, g, xy, radec):
    """Fold a new detection into an existing global star (running sums)."""
    g_x[g] += float(xy[0])
    g_y[g] += float(xy[1])
    g_xy_n[g] += 1.0
    if radec is not None and np.isfinite(radec).all():
        g_ra[g] += float(radec[0])
        g_dec[g] += float(radec[1])
        g_radec_n[g] += 1.0
    g_ndet[g] += 1


def _match_sky(local_radec, global_radec, tol_arcsec):
    """Match local sky coords to global via SkyCoord; yield (li, gi) pairs.

    Each local star is matched to its nearest global star; the pair is kept
    only if the separation is below ``tol_arcsec``.
    """
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    local_dec = np.clip(local_radec[:, 1], -90.0, 90.0)
    glob_dec = np.clip(global_radec[:, 1], -90.0, 90.0)
    local = SkyCoord(ra=local_radec[:, 0] * u.deg, dec=local_dec * u.deg)
    glob = SkyCoord(ra=global_radec[:, 0] * u.deg, dec=glob_dec * u.deg)
    idx, sep2d, _ = local.match_to_catalog_sky(glob)
    pairs = []
    for li in range(len(local)):
        if sep2d[li].arcsec < tol_arcsec:
            pairs.append((li, int(idx[li])))
    return pairs
