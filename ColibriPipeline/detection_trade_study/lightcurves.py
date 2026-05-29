"""
Loader and cleaner for per-telescope stars.npy files used in the offline
detection trade study.  Cleaning rules mirror colibri_photometry.dipDetection
(trim_zeros + minLightcurveLen = 100).
"""

from pathlib import Path
import numpy as np


# Matches colibri_photometry.py line 473
_MIN_LIGHTCURVE_LEN = 100


def load_minute(stars_npy_path):
    """Load a stars.npy file for one minute directory.

    Parameters
    ----------
    stars_npy_path : str or Path
        Path to a ``*_stars.npy`` file of shape ``(n_frames, n_stars, 4)``.
        Last axis layout: ``[0]=x, [1]=y, [2]=flux, [3]=unix_time``.

    Returns
    -------
    dict with keys:
        'flux' : ndarray, shape (n_stars, n_frames)  -- arr[:,:,2].T
        'time' : ndarray, shape (n_frames,)           -- arr[:,0,3]
        'xy'   : ndarray, shape (n_stars, 2)          -- arr[0,:,0:2]
    """
    arr = np.load(stars_npy_path, allow_pickle=True)
    return {
        'flux': arr[:, :, 2].T,      # (n_stars, n_frames)
        'time': arr[:, 0, 3],        # (n_frames,)
        'xy':   arr[0, :, 0:2],      # (n_stars, 2)
    }


def clean(flux_1d):
    """Apply trim_zeros and minimum-length pruning to a single light curve.

    Mirrors the pruning steps in colibri_photometry.dipDetection lines 460-489:
    np.trim_zeros to remove leading/trailing zeros, then reject if shorter
    than minLightcurveLen (100 frames).

    Parameters
    ----------
    flux_1d : array_like, 1-D
        Raw flux time series for one star.

    Returns
    -------
    ndarray or None
        Cleaned 1-D array, or None if the curve is rejected.
    """
    lc = np.trim_zeros(np.asarray(flux_1d, dtype=float))
    if len(lc) < _MIN_LIGHTCURVE_LEN:
        return None
    return lc


def good_star_indices(flux_2d):
    """Return indices of stars whose light curves survive clean().

    Parameters
    ----------
    flux_2d : ndarray, shape (n_stars, n_frames)
        Flux array as returned by load_minute().

    Returns
    -------
    list of int
        Indices (into axis-0 of flux_2d) of stars that pass clean().
    """
    good = []
    for i, row in enumerate(flux_2d):
        if clean(row) is not None:
            good.append(i)
    return good


def match_stars_across_telescopes(xy_by_scope, tol_px=2.0):
    """Nearest-neighbour pixel matching across telescopes.

    Parameters
    ----------
    xy_by_scope : dict
        Keys are telescope names (e.g. 'Red', 'Green', 'Blue'), values are
        ndarrays of shape (n_stars, 2) containing pixel (x, y) coordinates as
        returned by load_minute()['xy'].
    tol_px : float
        Maximum pixel distance for a cross-telescope match (default 2.0).

    Returns
    -------
    list of dict
        One dict per matched physical star.  Each dict maps telescope name to
        the star's integer index in that telescope's array.  Only stars present
        in ALL provided telescopes within tol_px are returned.

    Notes
    -----
    Matching is done in pixel space, which is sufficient for this offline
    study because the simulated data share a common pixel reference frame.
    For production use, RA/Dec matching via WCS would be more robust.
    """
    scopes = list(xy_by_scope.keys())
    if len(scopes) < 2:
        raise ValueError("Need at least two telescopes to match.")

    # Use the first telescope as the reference; iterate over its stars.
    ref_scope = scopes[0]
    ref_xy = np.asarray(xy_by_scope[ref_scope])
    other_scopes = scopes[1:]

    matches = []
    for ref_idx, ref_pos in enumerate(ref_xy):
        entry = {ref_scope: ref_idx}
        for other in other_scopes:
            other_xy = np.asarray(xy_by_scope[other])
            dists = np.hypot(other_xy[:, 0] - ref_pos[0],
                             other_xy[:, 1] - ref_pos[1])
            best_idx = int(np.argmin(dists))
            if dists[best_idx] > tol_px:
                break  # no match in this telescope — skip star
            entry[other] = best_idx
        else:
            # All telescopes matched
            matches.append(entry)

    return matches
