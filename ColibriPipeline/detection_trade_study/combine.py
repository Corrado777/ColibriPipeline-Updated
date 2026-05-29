"""
Multi-telescope combination of per-frame detection statistics.

The "power of three" lever: rather than thresholding each telescope
independently and then requiring a post-hoc time/coordinate coincidence
(as the live simultaneous_occults.py does), we combine the three per-frame
detection statistics into a single joint statistic and threshold once.

For independent per-telescope noise, summing N z-like scores and dividing by
sqrt(N) keeps the noise unit-variance while a real, time-coincident event adds
coherently -- so a signal that is ~3 sigma in each scope becomes ~5.2 sigma
jointly, while independent false positives do not stack.  This lets each scope
run at a lower per-scope threshold at fixed joint false-alarm rate.

This module also provides an `and_reference` detector that reproduces the
current post-threshold AND scheme, purely so the harness can quantify the gain
of the joint statistic over it.
"""

import numpy as np


def align_scores(scores_by_scope, times_by_scope=None, ref_scope=None):
    """Align per-telescope per-frame score arrays onto a common frame grid.

    The three minute light curves are near-simultaneous 40 Hz series, so by
    default alignment is by frame index (truncated to the common length).  If
    per-scope GPS time vectors are supplied, each score is instead interpolated
    onto the reference scope's time grid (NaNs treated as 0 for interpolation).

    Parameters
    ----------
    scores_by_scope : dict
        scope name -> per-frame score ndarray (as returned by a Detector).
    times_by_scope : dict, optional
        scope name -> per-frame unix-time ndarray.  If given, scores are
        interpolated onto the reference scope's time grid.
    ref_scope : str, optional
        Reference scope for the common grid.  Defaults to the first key.

    Returns
    -------
    aligned : ndarray, shape (n_scopes, n_frames)
        Score rows aligned on the common grid, NaNs replaced with 0.
    grid : ndarray
        The common grid (frame indices, or reference times if interpolating).
    scopes : list of str
        Scope order corresponding to rows of `aligned`.
    """
    scopes = list(scores_by_scope.keys())
    if ref_scope is None:
        ref_scope = scopes[0]

    def _nan_to_zero(a):
        a = np.asarray(a, dtype=float)
        return np.where(np.isfinite(a), a, 0.0)

    if times_by_scope is None:
        # Index alignment: truncate to the shortest score
        n = min(len(scores_by_scope[s]) for s in scopes)
        aligned = np.stack([_nan_to_zero(scores_by_scope[s])[:n] for s in scopes], axis=0)
        return aligned, np.arange(n), scopes

    # Time-grid alignment
    grid = np.asarray(times_by_scope[ref_scope], dtype=float)
    rows = []
    for s in scopes:
        t = np.asarray(times_by_scope[s], dtype=float)
        v = _nan_to_zero(scores_by_scope[s])
        rows.append(np.interp(grid, t, v, left=0.0, right=0.0))
    return np.stack(rows, axis=0), grid, scopes


def joint_score(aligned):
    """Combine aligned per-scope scores into one joint per-frame statistic.

    joint[f] = sum_i aligned[i, f] / sqrt(n_scopes)

    The sqrt(N) normalisation keeps independent unit-variance noise at unit
    variance while a coincident signal adds coherently.

    Parameters
    ----------
    aligned : ndarray, shape (n_scopes, n_frames)

    Returns
    -------
    ndarray, shape (n_frames,)
    """
    aligned = np.asarray(aligned, dtype=float)
    n_scopes = aligned.shape[0]
    return aligned.sum(axis=0) / np.sqrt(n_scopes)


def joint_peak(scores_by_scope, times_by_scope=None, ref_scope=None):
    """Return (peak_score, peak_frame) of the joint statistic.

    Convenience wrapper around align_scores + joint_score.  `peak_frame` is an
    index into the reference scope's grid (so it is comparable to an injection
    centre expressed in that scope's frames).
    """
    aligned, grid, _ = align_scores(scores_by_scope, times_by_scope, ref_scope=ref_scope)
    js = joint_score(aligned)
    if len(js) == 0:
        return float('nan'), 0
    peak_frame = int(np.argmax(js))
    return float(js[peak_frame]), peak_frame


def and_reference_detect(peak_by_scope, threshold, coincidence_frames=8):
    """Reproduce the current post-threshold AND coincidence scheme.

    Each scope must independently exceed `threshold`, and the per-scope peak
    frames must all fall within `coincidence_frames` of one another (default
    8 frames = 0.2 s at 40 Hz, matching simultaneous_occults TIME_TOLERANCE).

    Parameters
    ----------
    peak_by_scope : dict
        scope -> (peak_score, peak_frame).
    threshold : float
        Per-scope detection threshold.
    coincidence_frames : int
        Maximum spread (max-min) of peak frames to count as coincident.

    Returns
    -------
    bool
        True if a coincident detection is registered.
    """
    scores = [v[0] for v in peak_by_scope.values()]
    frames = [v[1] for v in peak_by_scope.values()]
    if any(not np.isfinite(s) for s in scores):
        return False
    if min(scores) < threshold:
        return False
    return (max(frames) - min(frames)) <= coincidence_frames
