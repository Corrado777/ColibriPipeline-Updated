"""
Injection / recovery harness for the occultation-detection trade study.

Two experiments, sharing one injection/recovery + ROC engine:

  * Single-telescope detector trade study (mode='real')
      Inject synthetic Fresnel events into REAL positive-flux star light curves
      (real noise) and compare detectors head to head.  Answers "is the Ricker
      more sensitive than a geometric / multi-width / Fresnel-matched / Pass-SNR
      detector?".

  * Multi-telescope "power of three" study (mode='bootstrap')
      Synthesise N independent telescopes from one real star by block-bootstrapping
      its residual noise (bootstrap.make_independent_scopes), inject the SAME event
      coincidently in all of them, and compare single-telescope vs the sqrt(N)
      joint statistic vs the post-threshold AND scheme.  (The bundled sim minute
      has identical Green/Blue and a partial Red, so a raw multi-scope join would
      stack correlated noise -- hence the bootstrap.)

The recorded trial table is a pandas DataFrame so the ROC step can slice it
freely without re-running detectors.
"""

import numpy as np
import pandas as pd

from . import lightcurves as lc
from . import injection as inj
from . import combine
from . import bootstrap as bs
from . import detectors as det_mod


# Default test minute (three simultaneous telescopes, 2025-08-30 01.54.12)
_SIM_ROOT = "/home/agirmen/research_data/ColibriPipelineSimulatedDirs"
_MINUTE = "20250830_01.54.12.147_stars.npy"
DEFAULT_DATA_PATHS = {
    scope: f"{_SIM_ROOT}/{scope}/ColibriArchive/2025-08-30/{_MINUTE}"
    for scope in ("Red", "Green", "Blue")
}


def load_all(data_paths=None):
    """Load all telescopes and cross-match stars present in every scope.

    Returns
    -------
    data_by_scope : dict   scope -> {'flux','time','xy'}
    matches : list of dict  each maps scope -> star index (clean in all scopes)
    scopes : list of str
    """
    if data_paths is None:
        data_paths = DEFAULT_DATA_PATHS
    scopes = list(data_paths.keys())
    data_by_scope = {s: lc.load_minute(p) for s, p in data_paths.items()}

    xy_by_scope = {s: data_by_scope[s]['xy'] for s in scopes}
    raw_matches = lc.match_stars_across_telescopes(xy_by_scope)
    good = {s: set(lc.good_star_indices(data_by_scope[s]['flux'])) for s in scopes}
    matches = [m for m in raw_matches if all(m[s] in good[s] for s in scopes)]
    return data_by_scope, matches, scopes


def select_full_scopes(data_by_scope, scopes, min_frac=0.9):
    """Return scopes whose minute is (near) full length (see module docstring)."""
    n_max = max(data_by_scope[s]['flux'].shape[1] for s in scopes)
    return [s for s in scopes if data_by_scope[s]['flux'].shape[1] >= min_frac * n_max]


def photometric_hosts(flux_2d, min_median=500.0):
    """Indices of stars suitable as injection hosts.

    Multiplicative injection and the flux-normalised statistics are only valid
    for genuinely positive stellar flux, so we require a median flux above
    `min_median` counts (and that the curve survives the standard clean()).
    """
    hosts = []
    for i in lc.good_star_indices(flux_2d):
        if np.median(flux_2d[i]) >= min_median:
            hosts.append(i)
    return hosts


def scopes_identical(data_by_scope, scopes, n_check=5):
    """True if the listed scopes carry byte-identical flux (correlated noise)."""
    if len(scopes) < 2:
        return False
    ref = data_by_scope[scopes[0]]['flux']
    return all(np.allclose(ref, data_by_scope[s]['flux']) for s in scopes[1:])


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def _score_event(flux_by_scope, times_by_scope, detectors, ref_scope, exposure_time):
    """Run all detectors over a prebuilt per-scope flux dict.

    Returns flat columns: f'{det}|{scope}|peak', f'{det}|{scope}|frame',
    f'{det}|joint|peak', f'{det}|joint|frame'.

    The joint statistic uses FRAME-INDEX alignment, not time interpolation: the
    bundled sim minute quantises its GPS timestamps to whole seconds (61 unique
    values for 2399 frames), so np.interp on them is meaningless.  All scopes
    here share a common 40 Hz frame grid, so index alignment is exact.  (The
    `times_by_scope` argument is retained for callers that genuinely have
    heterogeneous, strictly-increasing grids.)
    """
    row = {}
    for det_name, detector in detectors.items():
        scores_by_scope = {}
        for s in flux_by_scope:
            res = detector.run(flux_by_scope[s], exposure_time)
            scores_by_scope[s] = res.score
            row[f'{det_name}|{s}|peak'] = res.peak_score
            row[f'{det_name}|{s}|frame'] = res.peak_frame
        jpeak, jframe = combine.joint_peak(scores_by_scope,
                                           times_by_scope=times_by_scope,
                                           ref_scope=ref_scope)
        row[f'{det_name}|joint|peak'] = jpeak
        row[f'{det_name}|joint|frame'] = jframe
    return row


def _nearest_frame(time_vec, t):
    return int(np.argmin(np.abs(np.asarray(time_vec, dtype=float) - t)))


# ---------------------------------------------------------------------------
# Trial drivers
# ---------------------------------------------------------------------------

def run_trials_real(data_by_scope, matches, scopes, detectors,
                    n_inject, n_null, rng, edge_margin=250, exposure_time=0.025):
    """Inject into real cross-matched star light curves across `scopes`.

    Events are injected at the same absolute GPS time in every scope (mapped to
    each scope's nearest frame), so unequal frame grids are handled correctly.
    """
    ref_scope = max(scopes, key=lambda s: data_by_scope[s]['flux'].shape[1])
    ref_time = np.asarray(data_by_scope[ref_scope]['time'], dtype=float)
    n_frames = len(ref_time)
    rows = []

    def _build_flux(match, center_time, profile):
        out = {}
        for s in scopes:
            f = np.asarray(data_by_scope[s]['flux'][match[s]], dtype=float)
            if profile is not None:
                f = inj.inject(f, _nearest_frame(data_by_scope[s]['time'], center_time),
                               profile)
            out[s] = f
        return out

    for kind, n in (('inject', n_inject), ('background', n_null)):
        for _ in range(n):
            match = matches[rng.integers(len(matches))]
            host_flux = np.asarray(data_by_scope[ref_scope]['flux'][match[ref_scope]], dtype=float)
            host_snr = lc.star_snr(host_flux)
            if kind == 'inject':
                params = inj.sample_params(rng, exposure_time)
                profile = inj.make_profile(params, exposure_time)
                center_idx = int(rng.integers(edge_margin, n_frames - edge_margin))
                center_time = ref_time[center_idx]
                meta = {'depth': inj.injected_depth(profile),
                        'objectRad': params['objectRad'], 'dist': params['dist'],
                        'center_frame': center_idx,
                        'snr': host_snr, 'duration': len(profile)}
            else:
                profile, center_time = None, None
                meta = {'depth': np.nan, 'objectRad': np.nan, 'dist': np.nan,
                        'center_frame': -1,
                        'snr': host_snr, 'duration': np.nan}
            flux_by_scope = _build_flux(match, center_time, profile)
            row = {'kind': kind, 'ref_idx': match[ref_scope], **meta}
            row.update(_score_event(flux_by_scope, None, detectors,
                                    ref_scope, exposure_time))
            rows.append(row)

    return pd.DataFrame(rows), scopes, ref_scope


def run_trials_bootstrap(source_data, host_indices, detectors, n_scopes,
                         n_inject, n_null, rng, edge_margin=250,
                         bootstrap_block=20, exposure_time=0.025):
    """Synthesise `n_scopes` independent telescopes per trial via bootstrap.

    `source_data` is one scope's {'flux','time'} (a real, full minute).  Each
    trial draws a real positive-flux host star, builds n_scopes independent
    light curves (shared baseline + independent block-bootstrapped residuals),
    injects the same event coincidently, and scores all scopes.
    """
    flux2d = source_data['flux']
    time_vec = np.asarray(source_data['time'], dtype=float)
    n_frames = len(time_vec)
    scope_names = [f"T{i + 1}" for i in range(n_scopes)]
    ref_scope = scope_names[0]
    rows = []

    for kind, n in (('inject', n_inject), ('background', n_null)):
        for _ in range(n):
            host = int(host_indices[rng.integers(len(host_indices))])
            host_snr = lc.star_snr(flux2d[host])
            scopes_flux = bs.make_independent_scopes(
                flux2d[host], n_scopes, rng, block=bootstrap_block, names=scope_names)
            if kind == 'inject':
                params = inj.sample_params(rng, exposure_time)
                profile = inj.make_profile(params, exposure_time)
                center_idx = int(rng.integers(edge_margin, n_frames - edge_margin))
                scopes_flux = {s: inj.inject(f, center_idx, profile)
                               for s, f in scopes_flux.items()}
                meta = {'depth': inj.injected_depth(profile),
                        'objectRad': params['objectRad'], 'dist': params['dist'],
                        'center_frame': center_idx,
                        'snr': host_snr, 'duration': len(profile)}
            else:
                meta = {'depth': np.nan, 'objectRad': np.nan, 'dist': np.nan,
                        'center_frame': -1,
                        'snr': host_snr, 'duration': np.nan}
            row = {'kind': kind, 'ref_idx': host, **meta}
            row.update(_score_event(scopes_flux, None, detectors,
                                    ref_scope, exposure_time))
            rows.append(row)

    return pd.DataFrame(rows), scope_names, ref_scope


# ---------------------------------------------------------------------------
# ROC / completeness
# ---------------------------------------------------------------------------

def _threshold_grid(values, n=80):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.linspace(0, 1, n)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if lo == hi:
        hi = lo + 1.0
    return np.linspace(lo, hi, n)


def compute_roc(df, scopes, ref_scope, detectors=None,
                recovery_tol=12, coincidence_frames=8):
    """Trace completeness vs false-alarm-rate for each detector & policy.

    Policies per detector:
      'single' : reference-scope peak only.
      'joint'  : sqrt(N)-normalised joint statistic peak.
      'and'    : post-threshold AND across scopes (the current scheme).

    Completeness = fraction of injections detected AND localised within
    `recovery_tol` frames of the injected centre.
    False-alarm rate = fraction of null trials firing above threshold.

    Returns roc[det][policy] = {'thresholds','far','completeness'}.
    """
    inj_df = df[df['kind'] == 'inject']
    null_df = df[df['kind'] == 'background']
    if detectors is None:
        detectors = sorted({c.split('|')[0] for c in df.columns if '|' in c})

    def _peakframe_roc(peak_col, frame_col):
        grid = _threshold_grid(np.concatenate([inj_df[peak_col].values,
                                               null_df[peak_col].values]))
        recovered = (np.abs(inj_df[frame_col].values - inj_df['center_frame'].values)
                     <= recovery_tol)
        comp, far = [], []
        for T in grid:
            comp.append(((inj_df[peak_col].values >= T) & recovered).mean()
                        if len(inj_df) else 0.0)
            far.append((null_df[peak_col].values >= T).mean() if len(null_df) else 0.0)
        return {'thresholds': grid, 'far': np.array(far), 'completeness': np.array(comp)}

    roc = {}
    for det in detectors:
        roc[det] = {}
        roc[det]['single'] = _peakframe_roc(f'{det}|{ref_scope}|peak',
                                            f'{det}|{ref_scope}|frame')
        roc[det]['joint'] = _peakframe_roc(f'{det}|joint|peak', f'{det}|joint|frame')

        # AND reference (threshold-dependent coincidence over all scopes)
        scope_peaks = {s: f'{det}|{s}|peak' for s in scopes}
        scope_frames = {s: f'{det}|{s}|frame' for s in scopes}
        grid = _threshold_grid(np.concatenate([df[scope_peaks[s]].values for s in scopes]))
        comp, far = [], []
        for T in grid:
            n_rec = 0
            for _, r in inj_df.iterrows():
                pbs = {s: (r[scope_peaks[s]], r[scope_frames[s]]) for s in scopes}
                if combine.and_reference_detect(pbs, T, coincidence_frames) and \
                   abs(r[scope_frames[ref_scope]] - r['center_frame']) <= recovery_tol:
                    n_rec += 1
            n_fa = 0
            for _, r in null_df.iterrows():
                pbs = {s: (r[scope_peaks[s]], r[scope_frames[s]]) for s in scopes}
                if combine.and_reference_detect(pbs, T, coincidence_frames):
                    n_fa += 1
            comp.append(n_rec / len(inj_df) if len(inj_df) else 0.0)
            far.append(n_fa / len(null_df) if len(null_df) else 0.0)
        roc[det]['and'] = {'thresholds': grid, 'far': np.array(far),
                          'completeness': np.array(comp)}

    return roc


def completeness_at_far(roc_entry, target_far=0.01):
    """Best completeness achievable at <= target false-alarm rate."""
    far = np.asarray(roc_entry['far'], dtype=float)
    comp = np.asarray(roc_entry['completeness'], dtype=float)
    ok = far <= target_far
    return float(np.nanmax(comp[ok])) if ok.any() else 0.0


# ---------------------------------------------------------------------------
# Brightness/SNR-aware completeness
# ---------------------------------------------------------------------------

def event_snr(df):
    """Return a Series: depth * snr * sqrt(duration) (a combined matched-filter
    event SNR proxy). NaN where not an inject row.

    Parameters
    ----------
    df : DataFrame
        Trial table as returned by run_trials_real or run_trials_bootstrap.

    Returns
    -------
    pd.Series of float, same index as df.
    """
    mask = df['kind'] == 'inject'
    result = pd.Series(np.nan, index=df.index, dtype=float)
    inj_rows = df[mask]
    result[mask] = (inj_rows['depth'].values
                    * inj_rows['snr'].values
                    * np.sqrt(inj_rows['duration'].values.astype(float)))
    return result


def _threshold_at_far(null_peaks, far_target):
    """Smallest threshold T such that fraction(null_peaks >= T) <= far_target.

    Uses the (1-far_target) quantile of finite null peaks.

    Parameters
    ----------
    null_peaks : array_like
    far_target : float

    Returns
    -------
    float
    """
    vals = np.asarray(null_peaks, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.inf
    return float(np.quantile(vals, 1.0 - far_target))


def completeness_map(df, det, ref_scope, far_target=0.01, recovery_tol=12,
                     snr_bins=None, depth_bins=None):
    """2D completeness over (stellar SNR x event depth) at fixed FAR.

    Uses the single reference-scope policy (columns f'{det}|{ref_scope}|peak'
    and f'{det}|{ref_scope}|frame') together with 'center_frame', 'snr', and
    'depth' from the trial table.

    - Threshold T = _threshold_at_far(background-row peaks, far_target).
    - An inject row is recovered if peak >= T AND |frame - center_frame| <= recovery_tol.
    - Default snr_bins = np.linspace(0, max_snr, 7); depth_bins = np.linspace(0, 1, 6).

    Parameters
    ----------
    df : DataFrame
    det : str
        Detector name (e.g. 'BoxDetector').
    ref_scope : str
    far_target : float
    recovery_tol : int
    snr_bins : array_like or None
    depth_bins : array_like or None

    Returns
    -------
    dict with keys:
        'snr_edges'    : 1-D ndarray  (len = n_snr_bins + 1)
        'depth_edges'  : 1-D ndarray  (len = n_depth_bins + 1)
        'completeness' : 2-D ndarray  (rows = snr, cols = depth), NaN for empty cells
        'counts'       : 2-D ndarray  (n injections per cell)
    """
    peak_col = f'{det}|{ref_scope}|peak'
    frame_col = f'{det}|{ref_scope}|frame'

    null_df = df[df['kind'] == 'background']
    inj_df = df[df['kind'] == 'inject'].copy()

    T = _threshold_at_far(null_df[peak_col].values, far_target)

    recovered = ((inj_df[peak_col].values >= T) &
                 (np.abs(inj_df[frame_col].values - inj_df['center_frame'].values)
                  <= recovery_tol))
    inj_df = inj_df.copy()
    inj_df['_recovered'] = recovered

    snr_vals = inj_df['snr'].values.astype(float)
    depth_vals = inj_df['depth'].values.astype(float)

    if snr_bins is None:
        max_snr = np.nanmax(snr_vals) if len(snr_vals) > 0 else 1.0
        snr_bins = np.linspace(0, max_snr, 7)
    else:
        snr_bins = np.asarray(snr_bins, dtype=float)

    if depth_bins is None:
        depth_bins = np.linspace(0, 1, 6)
    else:
        depth_bins = np.asarray(depth_bins, dtype=float)

    n_snr = len(snr_bins) - 1
    n_depth = len(depth_bins) - 1
    completeness = np.full((n_snr, n_depth), np.nan)
    counts = np.zeros((n_snr, n_depth), dtype=int)

    for i in range(n_snr):
        for j in range(n_depth):
            s_lo, s_hi = snr_bins[i], snr_bins[i + 1]
            d_lo, d_hi = depth_bins[j], depth_bins[j + 1]
            # include right edge only in last bin to avoid double-counting
            if i < n_snr - 1:
                s_mask = (snr_vals >= s_lo) & (snr_vals < s_hi)
            else:
                s_mask = (snr_vals >= s_lo) & (snr_vals <= s_hi)
            if j < n_depth - 1:
                d_mask = (depth_vals >= d_lo) & (depth_vals < d_hi)
            else:
                d_mask = (depth_vals >= d_lo) & (depth_vals <= d_hi)
            cell = s_mask & d_mask
            n_cell = int(cell.sum())
            counts[i, j] = n_cell
            if n_cell > 0:
                completeness[i, j] = float(inj_df['_recovered'].values[cell].mean())

    return {
        'snr_edges': snr_bins,
        'depth_edges': depth_bins,
        'completeness': completeness,
        'counts': counts,
    }


def completeness_vs_eventSNR(df, det, ref_scope, far_target=0.01, recovery_tol=12,
                              n_bins=12):
    """Completeness vs the combined event SNR (event_snr).

    Same threshold/recovery logic as completeness_map, binned along event_snr
    in log-spaced bins over the finite positive range.

    Parameters
    ----------
    df : DataFrame
    det : str
    ref_scope : str
    far_target : float
    recovery_tol : int
    n_bins : int

    Returns
    -------
    dict with keys:
        'centers'      : 1-D ndarray, bin centres (geometric mean of edges)
        'completeness' : 1-D ndarray, completeness per bin (NaN for empty bins)
        'counts'       : 1-D ndarray of int, n injections per bin
    """
    peak_col = f'{det}|{ref_scope}|peak'
    frame_col = f'{det}|{ref_scope}|frame'

    null_df = df[df['kind'] == 'background']
    inj_df = df[df['kind'] == 'inject'].copy()

    T = _threshold_at_far(null_df[peak_col].values, far_target)

    recovered = ((inj_df[peak_col].values >= T) &
                 (np.abs(inj_df[frame_col].values - inj_df['center_frame'].values)
                  <= recovery_tol))
    inj_df['_recovered'] = recovered

    esnr = (inj_df['depth'].values.astype(float)
            * inj_df['snr'].values.astype(float)
            * np.sqrt(inj_df['duration'].values.astype(float)))
    inj_df['_esnr'] = esnr

    finite_pos = esnr[np.isfinite(esnr) & (esnr > 0)]
    if len(finite_pos) == 0:
        return {'centers': np.array([]), 'completeness': np.array([]), 'counts': np.array([], dtype=int)}

    lo, hi = np.log10(finite_pos.min()), np.log10(finite_pos.max())
    if lo == hi:
        hi = lo + 1.0
    edges = np.logspace(lo, hi, n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean

    completeness = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (esnr >= edges[i]) & (esnr < edges[i + 1])
        else:
            mask = (esnr >= edges[i]) & (esnr <= edges[i + 1])
        n_cell = int(mask.sum())
        counts[i] = n_cell
        if n_cell > 0:
            completeness[i] = float(inj_df['_recovered'].values[mask].mean())

    return {'centers': centers, 'completeness': completeness, 'counts': counts}
