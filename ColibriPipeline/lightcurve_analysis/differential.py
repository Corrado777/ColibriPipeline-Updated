#!/usr/bin/env python3
"""
Filename : differential.py
Author   : ColibriPipeline / lightcurve_analysis (Wave 2)

Differential (relative) photometry on a stitched ``NightCatalog``.

Two methods are provided:

* ``ensemble`` (default): divide the target light curve by a nan-aware
  mean/median of nearby comparison stars, then normalize the result by its own
  median so the baseline sits near 1.0. This mirrors the
  ``add_differential_columns`` / ``normalize_rel_flux_per_star`` semantics in
  ``Ultimate_photometry_package`` (``rel = net_flux / ref_denom`` then divide by
  the per-star median), adapted to the stitched ``(n_stars, n_samples)`` matrix
  model that ``NightCatalog.get_flux_matrix`` exposes.
* ``pca``: normalize each curve by its own nan-aware mean, fit PCA on the
  reference curves only and remove the reconstructed common mode from both refs
  and target (reusing ``_pca_detrend_with_refs_only`` from
  ``Ultimate_photometry_package``), then divide the detrended target by the
  nan-aware mean of the detrended refs.

Reference selection works on representative pixel positions
(``cat.stars[['x', 'y']]``): comparison stars within ``comp_rad_pix`` of the
target, excluding the target, optionally filtered by SNR.

``Ultimate_photometry_package`` (111 KB) is imported lazily inside the functions
that need it so importing this module stays cheap.
"""

import warnings

import numpy as np

# Intra-package modules (always relative when loaded as a package).
if __package__:
    from .stitch import LightCurve, bin_lightcurve
else:  # flat execution from inside lightcurve_analysis/
    from stitch import LightCurve, bin_lightcurve  # type: ignore


def _try_import_upp():
    """Lazily import Ultimate_photometry_package (heavy, 111 KB), or None.

    Mirrors the guarded shim style used elsewhere in the package: try the
    package-relative import first, then fall back to a flat import after
    inserting the ColibriPipeline directory on sys.path. Returns ``None`` if
    the import fails for any reason (UPP has heavy top-level dependencies such
    as ``sklearn`` that may be absent on a minimal analysis box); callers fall
    back to the replicated math below.
    """
    try:
        from .. import Ultimate_photometry_package as upp  # type: ignore
        return upp
    except (ImportError, ValueError, ModuleNotFoundError):
        pass
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _pkg_parent = _Path(__file__).resolve().parent.parent
        if str(_pkg_parent) not in _sys.path:
            _sys.path.insert(0, str(_pkg_parent))
        import Ultimate_photometry_package as upp  # type: ignore
        return upp
    except (ImportError, ValueError, ModuleNotFoundError):
        return None


def _pca_detrend_with_refs_only(Fn_refs, Fn_target, max_pcomps=3):
    """Replicated from Ultimate_photometry_package, numpy-only.

    Fit a PCA on the reference normalized curves only (via centered SVD,
    equivalent to ``sklearn.decomposition.PCA``), reconstruct the common mode
    from the top components, and remove it from both refs and target,
    re-adding the baseline 1.0. NaNs are replaced by 1.0 for the fit and put
    back afterward. This avoids a hard dependency on sklearn / the heavy UPP
    top-level imports while preserving the original semantics.

    Fn_refs   : (Nref, T)
    Fn_target : (T,)
    """
    Fn_refs = np.asarray(Fn_refs, dtype=float)
    Fn_target = np.asarray(Fn_target, dtype=float)

    nref, T = Fn_refs.shape
    if nref < 2 or T < 3:
        return Fn_refs.copy(), Fn_target.copy()

    Xr = np.where(np.isfinite(Fn_refs), Fn_refs, 1.0)
    xt = np.where(np.isfinite(Fn_target), Fn_target, 1.0).reshape(1, -1)

    n_components = int(min(max_pcomps, nref, T))
    if n_components < 1:
        return Fn_refs.copy(), Fn_target.copy()

    # PCA reconstruction via centered SVD (matches sklearn PCA inverse_transform
    # of transform: project onto top-k components about the training mean).
    mean = Xr.mean(axis=0, keepdims=True)
    Xc = Xr - mean
    # economy SVD; rows are samples (refs), columns are features (time)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components]                      # (k, T)
    proj = comps.T @ comps                         # (T, T) projector

    Xr_common = (Xc @ proj) + mean
    xt_common = ((xt - mean) @ proj) + mean

    Xr_det = Xr - Xr_common + 1.0
    xt_det = (xt - xt_common + 1.0).reshape(-1)

    Xr_det = np.where(np.isfinite(Fn_refs), Xr_det, np.nan)
    xt_det = np.where(np.isfinite(Fn_target), xt_det, np.nan)
    return Xr_det, xt_det


# ---------------------------------------------------------------------------
# Reference-star selection
# ---------------------------------------------------------------------------

def select_reference_stars(cat, star_id, comp_rad_pix=100.0, min_refs=3,
                           max_refs=None, snr_min=None):
    """Select comparison stars near a target by pixel radius.

    Returns the global ``star_id`` values of stars within ``comp_rad_pix`` of
    the target (using the representative pixel positions in ``cat.stars`` /
    ``cat.match.global_xy``), excluding the target itself, optionally filtered
    by SNR.

    Parameters
    ----------
    cat : NightCatalog
    star_id : int
        Global id of the target star.
    comp_rad_pix : float
        Selection radius in pixels.
    min_refs : int
        Minimum number of references required; if fewer are found a
        ``ValueError`` is raised.
    max_refs : int, optional
        If given, keep only the ``max_refs`` nearest references.
    snr_min : float, optional
        If given, keep only references whose ``snr`` is >= this value.

    Returns
    -------
    list[int]
        Global star ids of the selected comparison stars (nearest first).
    """
    star_id = int(star_id)
    df = cat.stars

    # Map star_id -> position. The stars table may be a filtered view, so index
    # by the star_id column rather than positional row order.
    sid_arr = df['star_id'].to_numpy()
    xy = df[['x', 'y']].to_numpy(dtype=float)

    tgt_rows = np.flatnonzero(sid_arr == star_id)
    if tgt_rows.size == 0:
        raise ValueError(
            f"star_id {star_id} not present in catalog.stars")
    tx, ty = xy[tgt_rows[0]]

    # Try reusing the UPP IDL-style relative selector when the dense flux
    # matrix is small enough to be cheap; otherwise (and as the robust default)
    # fall back to a straight pixel-radius KDTree-style cut, which is what the
    # plan specifies as the fallback. The UPP selector applies its own
    # PCA/min_frac_present cuts that are too aggressive on gappy night data, so
    # a plain radius cut is the more predictable choice here.
    d = np.hypot(xy[:, 0] - tx, xy[:, 1] - ty)
    within = (d <= float(comp_rad_pix)) & (sid_arr != star_id)

    if snr_min is not None and 'snr' in df.columns:
        within &= df['snr'].to_numpy(dtype=float) >= float(snr_min)

    cand_idx = np.flatnonzero(within)
    if cand_idx.size < int(min_refs):
        raise ValueError(
            f"Only {cand_idx.size} reference star(s) found within "
            f"{comp_rad_pix} px of star {star_id} (min_refs={min_refs}). "
            f"Try a larger comp_rad_pix or relax snr_min.")

    # nearest-first ordering
    order = np.argsort(d[cand_idx])
    cand_idx = cand_idx[order]
    if max_refs is not None:
        cand_idx = cand_idx[:int(max_refs)]

    return [int(s) for s in sid_arr[cand_idx]]


# ---------------------------------------------------------------------------
# Differential light curve
# ---------------------------------------------------------------------------

def _nan_all_lightcurve(t, star_id, bin_seconds, reason):
    """Build an all-NaN differential LightCurve and emit a warning."""
    warnings.warn(reason, RuntimeWarning, stacklevel=2)
    t = np.asarray(t, dtype=float)
    return LightCurve(t=t, f=np.full(t.shape[0], np.nan),
                      kind='differential', bin_seconds=bin_seconds,
                      star_id=int(star_id))


def differential_lightcurve(cat, star_id, *, method='ensemble',
                            comp_rad_pix=100.0, denom_mode='mean',
                            max_pcomps=3, min_keep=3, ref_ids=None,
                            bin_seconds=None):
    """Compute a differential light curve for one star.

    Parameters
    ----------
    cat : NightCatalog
    star_id : int
        Global id of the target.
    method : {'ensemble', 'pca'}
        ``ensemble`` divides by the nan-aware mean/median of reference fluxes;
        ``pca`` removes the reference common mode before dividing.
    comp_rad_pix : float
        Reference-selection radius in pixels (used only if ``ref_ids`` is None).
    denom_mode : {'mean', 'median'}
        Reduction used to combine reference fluxes per sample.
    max_pcomps : int
        Maximum number of principal components for ``method='pca'``.
    min_keep : int
        Minimum number of reference stars required.
    ref_ids : sequence[int], optional
        Explicit reference star ids. If None, ``select_reference_stars`` is
        called.
    bin_seconds : float, optional
        If given, the differential curve is binned to this width (nan-aware
        mean) and the returned ``LightCurve`` carries ``bin_seconds``.

    Returns
    -------
    LightCurve
        ``kind='differential'`` (or unchanged after binning). ``.t`` is the
        common time axis (or bin centers); ``.f`` is the differential flux with
        NaN at gaps / undefined samples. On too-few-references or all-NaN
        inputs the returned curve has all-NaN flux and a warning is emitted.
    """
    star_id = int(star_id)
    if method not in ('ensemble', 'pca'):
        raise ValueError(f"method must be 'ensemble' or 'pca', got {method!r}")
    if denom_mode not in ('mean', 'median'):
        raise ValueError(
            f"denom_mode must be 'mean' or 'median', got {denom_mode!r}")

    time_axis = np.asarray(cat.time_axis, dtype=float)

    # ----- reference selection -----
    if ref_ids is None:
        try:
            ref_ids = select_reference_stars(
                cat, star_id, comp_rad_pix=comp_rad_pix, min_refs=min_keep)
        except ValueError as exc:
            return _finalize(
                _nan_all_lightcurve(time_axis, star_id, bin_seconds, str(exc)),
                bin_seconds)
    else:
        ref_ids = [int(r) for r in ref_ids if int(r) != star_id]
        if len(ref_ids) < int(min_keep):
            return _finalize(
                _nan_all_lightcurve(
                    time_axis, star_id, bin_seconds,
                    f"Only {len(ref_ids)} reference(s) supplied "
                    f"(min_keep={min_keep})."),
                bin_seconds)

    # ----- dense flux matrix: row 0 = target, rows 1.. = references -----
    flux = cat.get_flux_matrix([star_id] + list(ref_ids))  # (1+Nref, n_samples)
    target = flux[0]
    refs = flux[1:]

    if not np.isfinite(target).any():
        return _finalize(
            _nan_all_lightcurve(
                time_axis, star_id, bin_seconds,
                f"Target star {star_id} has no finite flux samples."),
            bin_seconds)
    if not np.isfinite(refs).any():
        return _finalize(
            _nan_all_lightcurve(
                time_axis, star_id, bin_seconds,
                f"No finite reference flux for star {star_id}."),
            bin_seconds)

    if method == 'ensemble':
        rel = _differential_ensemble(target, refs, denom_mode)
    else:
        rel = _differential_pca(target, refs, max_pcomps=max_pcomps,
                                min_refs_per_frame=min_keep)

    lc = LightCurve(t=time_axis, f=rel, kind='differential',
                    bin_seconds=None, star_id=star_id)
    return _finalize(lc, bin_seconds)


def _finalize(lc, bin_seconds):
    """Optionally bin the differential curve to ``bin_seconds``."""
    if bin_seconds is None or bin_seconds <= 0:
        return lc
    centers, binned = bin_lightcurve(lc.t, lc.f, bin_seconds, how='mean')
    return LightCurve(t=centers, f=binned, kind='differential',
                      bin_seconds=bin_seconds, star_id=lc.star_id)


def _differential_ensemble(target, refs, denom_mode):
    """rel = target / nan-aware mean|median(refs), normalized by its median.

    Mirrors ``add_differential_columns`` (rel_flux = net_flux / ref_denom with
    non-positive denominators masked to NaN) followed by
    ``normalize_rel_flux_per_star`` (divide by the per-star median), adapted to
    the (n_samples,) stitched vectors.
    """
    target = np.asarray(target, dtype=float)
    refs = np.asarray(refs, dtype=float)

    with warnings.catch_warnings():
        # all-NaN slices are expected at gaps; suppress the numpy warning.
        warnings.simplefilter('ignore', category=RuntimeWarning)
        if denom_mode == 'median':
            denom = np.nanmedian(refs, axis=0)
        else:
            denom = np.nanmean(refs, axis=0)

    # mask invalid denominators (zero/negative/NaN) to avoid div-by-zero
    denom = np.where(np.isfinite(denom) & (denom > 0), denom, np.nan)

    rel = np.full(target.shape[0], np.nan, dtype=float)
    good = np.isfinite(target) & np.isfinite(denom)
    rel[good] = target[good] / denom[good]

    # normalize baseline to ~1.0 by the per-star median (positive samples only)
    pos = np.isfinite(rel) & (rel > 0)
    if pos.any():
        med = np.nanmedian(rel[pos])
        if np.isfinite(med) and med > 0:
            rel = rel / med
    return rel


def _differential_pca(target, refs, max_pcomps=3, min_refs_per_frame=3):
    """PCA-detrend references, then rel = detrended_target / mean(detrended refs).

    Each curve is normalized by its own nan-aware mean (baseline ~1.0), then
    ``_pca_detrend_with_refs_only`` removes the reference common mode from both
    the refs and the target. The relative curve divides the detrended target by
    the nan-aware mean of the detrended refs over samples with at least
    ``min_refs_per_frame`` valid references.
    """
    target = np.asarray(target, dtype=float)
    refs = np.asarray(refs, dtype=float)

    # normalize each curve by its own nan-aware mean (positive samples only)
    def _norm(row):
        valid = np.isfinite(row) & (row > 0)
        if not valid.any():
            return np.full(row.shape[0], np.nan)
        m = np.nanmean(row[valid])
        if not (np.isfinite(m) and m > 0):
            return np.full(row.shape[0], np.nan)
        return row / m

    Fn_target = _norm(target)
    Fn_refs = np.vstack([_norm(r) for r in refs])

    # Prefer the real UPP implementation when it imports cleanly (sklearn
    # present); otherwise use the numpy-only replication above. Both share the
    # same semantics and signature.
    upp = _try_import_upp()
    detrend = (upp._pca_detrend_with_refs_only
               if upp is not None else _pca_detrend_with_refs_only)
    Fn_refs_det, Fn_target_det = detrend(
        Fn_refs, Fn_target, max_pcomps=max_pcomps)

    ok = np.isfinite(Fn_refs_det)
    n_ok = ok.sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        ref_mean = np.nanmean(np.where(ok, Fn_refs_det, np.nan), axis=0)
    # require enough refs per sample; mask non-positive denominators
    enough = n_ok >= int(min_refs_per_frame)
    ref_mean = np.where(enough & np.isfinite(ref_mean) & (ref_mean > 0),
                        ref_mean, np.nan)

    rel = np.full(target.shape[0], np.nan, dtype=float)
    good = np.isfinite(Fn_target_det) & np.isfinite(ref_mean)
    rel[good] = Fn_target_det[good] / ref_mean[good]

    # normalize baseline to ~1.0
    pos = np.isfinite(rel) & (rel > 0)
    if pos.any():
        med = np.nanmedian(rel[pos])
        if np.isfinite(med) and med > 0:
            rel = rel / med
    return rel
