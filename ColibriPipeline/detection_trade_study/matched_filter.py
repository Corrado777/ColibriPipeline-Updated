"""
Optimal matched-filter core for the offline occultation-detection trade study.

A single ``matched_filter`` function handles all three noise-model variants
produced by preprocessing.py:

    ('scalar',   float_std)          -- ordinary normalised cross-correlation
    ('perframe', ndarray_sigma)      -- locally normalised cross-correlation
    ('psd',      psd_ndarray, fs)    -- optimal whitened matched filter (FFT)

Sign convention: occultation templates dip below the baseline.  The returned
per-frame score is constructed so that a real dip yields a POSITIVE peak.
Edges where the statistic is undefined are left NaN-filled (the 'scalar' and
'perframe' paths use scipy.signal.correlate mode='same' which avoids this;
the PSD path uses the FFT circular-correlation approach which is also full-
length with no undefined edges).
"""

import numpy as np
import scipy.signal


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def matched_filter(series, template, noise, demean=False):
    """Return a full-length per-frame detection significance, dips POSITIVE.

    Parameters
    ----------
    series : array_like, 1-D
        Preprocessed residual flux (output of a Preprocessor).
    template : array_like, 1-D
        Dip template.  Should already be zero-mean; if not, it will be
        zero-meaned internally.  A real occultation dip corresponds to
        negative values in the template.
    noise : tuple
        Tagged noise model from preprocessing.Conditioned.noise:
            ('scalar',   float_std)
            ('perframe', ndarray_sigma_same_length_as_series)
            ('psd',      onesided_power_ndarray, fs_float)

    Returns
    -------
    ndarray
        Per-frame significance of the same length as ``series``.  A real dip
        produces a positive peak.  NaN values indicate undefined positions
        (none expected for the current implementations; the field is reserved
        for future edge-padding preprocessors).

    Notes
    -----
    Scalar / perframe path
        Zero-mean and unit-L2-normalise the template.  Cross-correlate with
        ``series`` via ``scipy.signal.correlate(mode='same')`` which returns
        the full-length response with zero-padding at the edges.  Divide by
        the noise level.  Because the template is zero-mean with a negative
        dip and the series has a negative dip at the event, their cross-
        correlation is *positive* at the dip location — no sign flip needed.

    PSD (optimal whitened) path
        Zero-mean the template, zero-pad to ``len(series)``, then compute

            S = irfft( conj(rfft(template_padded)) * rfft(series) / P )

        This is the whitened cross-correlation in the frequency domain.  The
        result is normalised by ``sqrt(sum(|rfft(template_padded)|^2 / P))``
        so that S is in sigma units.  The raw irfft output is circularly
        shifted so that the centre of the template aligns with the
        corresponding frame (mode='same' semantics: shift by half the template
        length).

    Sign
        In both paths a template with a negative dip cross-correlated with a
        series that has a matching negative dip gives a *positive* response,
        so no sign flip is required.
    """
    series = np.asarray(series, dtype=float)
    template = np.asarray(template, dtype=float)
    n = len(series)

    # The preprocessor already removes the baseline (zero-mean / fractional
    # residual), so the template is matched as-is by default.  De-meaning is
    # opt-in: it must NOT be forced, or constant-shape templates (a flat box,
    # a single-frame delta) collapse to zero.
    if demean:
        template = template - template.mean()

    kind = noise[0]

    # ------------------------------------------------------------------
    # Scalar noise
    # ------------------------------------------------------------------
    if kind == 'scalar':
        sigma = float(noise[1])
        if sigma == 0.0:
            sigma = 1.0  # guard

        # Unit-L2-normalise
        t_norm = np.linalg.norm(template)
        if t_norm == 0.0:
            return np.full(n, np.nan)
        tmpl_unit = template / t_norm

        # correlate(series, template, 'same'):
        #   output[k] = sum_j series[k+j] * template[j]
        # A negative dip at position p in series aligned with the negative
        # dip in template gives a large POSITIVE dot product at lag p.
        raw = scipy.signal.correlate(series, tmpl_unit, mode='same')
        score = raw / sigma
        return score

    # ------------------------------------------------------------------
    # Per-frame noise
    # ------------------------------------------------------------------
    elif kind == 'perframe':
        sigma = np.asarray(noise[1], dtype=float)
        if sigma.shape != (n,):
            raise ValueError(
                f"perframe sigma length {len(sigma)} != series length {n}"
            )

        t_norm = np.linalg.norm(template)
        if t_norm == 0.0:
            return np.full(n, np.nan)
        tmpl_unit = template / t_norm

        raw = scipy.signal.correlate(series, tmpl_unit, mode='same')
        # Avoid divide-by-zero
        safe_sigma = np.where(sigma == 0, 1.0, sigma)
        score = raw / safe_sigma
        return score

    # ------------------------------------------------------------------
    # PSD (optimal whitened matched filter)
    # ------------------------------------------------------------------
    elif kind == 'psd':
        P = np.asarray(noise[1], dtype=float)
        # fs = noise[2]  # available but not needed once P is on rfft grid

        # Zero-pad template to length n
        t_len = len(template)
        t_padded = np.zeros(n, dtype=float)
        t_padded[:t_len] = template

        # FFT of both signals
        S_rfft = np.fft.rfft(series)        # shape: n//2+1
        T_rfft = np.fft.rfft(t_padded)     # shape: n//2+1

        # Ensure P matches the rfft length
        rfft_len = n // 2 + 1
        if len(P) != rfft_len:
            raise ValueError(
                f"PSD length {len(P)} != rfft length {rfft_len}. "
                "Whiten preprocessor should have produced matching lengths."
            )

        # Whitened cross-correlation: conj(T) * S / P
        cross = np.conj(T_rfft) * S_rfft / P

        # Normalisation factor: sqrt(sum(|T|^2 / P))
        norm_sq = np.sum(np.abs(T_rfft) ** 2 / P)
        if norm_sq <= 0.0:
            return np.full(n, np.nan)
        norm = np.sqrt(norm_sq)

        # IFFT back to time domain → circular cross-correlation
        score_circ = np.fft.irfft(cross, n=n)

        # mode='same' shift: roll so that the centre of the template aligns
        # with the corresponding series position.  irfft convention places
        # lag 0 at index 0; lag k at index k (positive) and at n-k (negative).
        # We want the peak to appear at the frame where the template centre
        # sits, i.e. shift by -(t_len//2) so the peak moves from index ~0
        # to the actual event frame.  np.roll with a negative shift moves
        # elements left (earlier indices go to later positions, so the response
        # at lag t_len//2 appears at index 0 → we roll by -(t_len//2)).
        shift = -(t_len // 2)
        score_circ = np.roll(score_circ, shift)

        score = score_circ / norm
        return score

    else:
        raise ValueError(f"Unknown noise model kind: {kind!r}")


# ---------------------------------------------------------------------------
# Peak helper
# ---------------------------------------------------------------------------

def peak(score):
    """Return (peak_frame, peak_score), NaN-safe.

    Parameters
    ----------
    score : array_like, 1-D
        Per-frame significance array as returned by ``matched_filter``.

    Returns
    -------
    peak_frame : int
        Index of the maximum (finite) value.  0 if all values are NaN.
    peak_score : float
        Maximum value.  nan if all values are NaN or the array is empty.
    """
    score = np.asarray(score, dtype=float)
    if score.size == 0 or np.all(np.isnan(score)):
        return (0, float('nan'))
    peak_frame = int(np.nanargmax(score))
    peak_score = float(np.nanmax(score))
    return (peak_frame, peak_score)
