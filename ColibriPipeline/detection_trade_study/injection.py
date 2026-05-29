"""
Inject synthetic Fresnel diffraction transmission profiles into real stellar
flux light curves for offline occultation-detection trade studies.

Depends on fresnel_physics (Qt-free Fresnel generator) from this package.
"""

import numpy as np
from . import fresnel_physics as fp


def sample_params(rng, exposure_time=0.025):
    """Return a dict of physically-sampled Fresnel parameters for genCurve.

    Samples over the physically plausible range of sub-km TNO occultation
    parameters:
      - objectRad : KBO radius, uniformly sampled 300-2000 m
      - dist      : distance to KBO, uniformly sampled 30-45 AU
      - impact    : impact parameter 0 up to ~2 object radii, in metres
      - angDi     : stellar angular diameter, uniformly sampled 0.01-0.3 mas
      - shiftAdj  : sub-frame phase uniformly sampled in [-0.5, 0.5)

    Wavelength band is fixed: startLam=4e-7 m, endLam=7e-7 m (optical).

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded random generator (e.g. numpy.random.default_rng(42)).
    exposure_time : float
        Frame exposure time in seconds (default 0.025 s = 40 Hz).

    Returns
    -------
    dict with keys: startLam, endLam, objectRad, impact, dist, angDi, shiftAdj
    """
    object_rad = rng.uniform(300.0, 2000.0)           # metres
    dist       = rng.uniform(30.0, 45.0)              # AU
    impact     = rng.uniform(0.0, 2.0 * object_rad)   # metres
    ang_di     = rng.uniform(0.01, 0.3)               # mas
    shift_adj  = rng.uniform(-0.5, 0.5)               # dimensionless

    return {
        "startLam": 4e-7,
        "endLam":   7e-7,
        "objectRad": object_rad,
        "impact":    impact,
        "dist":      dist,
        "angDi":     ang_di,
        "shiftAdj":  shift_adj,
    }


def make_profile(params, exposure_time=0.025):
    """Return a 1D multiplicative transmission profile centered on baseline 1.0.

    Calls fp.genCurve with the provided parameter dict, then normalises so
    that the out-of-event baseline equals exactly 1.0 (divides by the median
    of the outermost 20% of samples on each edge).  If genCurve returns a
    degenerate result (length 0 or all values within 1e-6 of each other) a
    length-1 array [1.0] is returned instead.

    Parameters
    ----------
    params : dict
        Parameter dict as returned by sample_params.
    exposure_time : float
        Frame exposure time in seconds (default 0.025 s).

    Returns
    -------
    numpy.ndarray
        1-D array of relative transmission values (baseline ~1.0, dip < 1.0).
    """
    raw = fp.genCurve(
        exposure_time,
        params["startLam"],
        params["endLam"],
        params["objectRad"],
        params["impact"],
        params["dist"],
        params["angDi"],
        params["shiftAdj"],
    )
    raw = np.asarray(raw, dtype=float)

    # Guard: degenerate output
    if len(raw) == 0 or np.ptp(raw) < 1e-6:
        return np.array([1.0])

    # Normalise baseline to 1.0 using edge samples (outermost 20%, min 1 pt)
    edge_n = max(1, len(raw) // 5)
    edge_samples = np.concatenate([raw[:edge_n], raw[-edge_n:]])
    baseline = np.median(edge_samples)
    if baseline == 0.0:
        return np.array([1.0])

    return raw / baseline


def inject(flux, center_frame, profile):
    """Return a copy of flux with the transmission profile multiplied in.

    The profile is centred at center_frame and straddles it symmetrically.
    The injection window is clipped to the array bounds so edge events are
    handled gracefully.  The input array is never modified in place.

    Parameters
    ----------
    flux : array-like, shape (N,)
        1-D array of stellar flux values.
    center_frame : int
        Index in flux where the centre of the occultation event is placed.
    profile : numpy.ndarray
        1-D multiplicative transmission profile (e.g. from make_profile).

    Returns
    -------
    numpy.ndarray
        Copy of flux with the profile applied.
    """
    flux = np.array(flux, dtype=float)
    profile = np.asarray(profile, dtype=float)
    n_prof = len(profile)
    half   = n_prof // 2

    # Index range in flux that receives the injection
    flux_start = center_frame - half
    flux_end   = flux_start + n_prof

    # Corresponding slice into the profile
    prof_start = 0
    prof_end   = n_prof

    # Clip to array bounds
    if flux_start < 0:
        prof_start -= flux_start   # advance into profile
        flux_start  = 0
    if flux_end > len(flux):
        prof_end  -= (flux_end - len(flux))
        flux_end   = len(flux)

    if flux_start >= flux_end or prof_start >= prof_end:
        return flux  # nothing overlaps

    flux[flux_start:flux_end] *= profile[prof_start:prof_end]
    return flux


def injected_depth(profile):
    """Return the fractional depth of the occultation event.

    Defined as 1.0 - min(profile).  A value of 0 means no dip; 1.0 means
    complete occultation.  Useful for binning recovery rate vs signal depth.

    Parameters
    ----------
    profile : array-like
        Transmission profile as returned by make_profile.

    Returns
    -------
    float
        Fractional depth in [0, 1].
    """
    return float(1.0 - np.min(profile))
