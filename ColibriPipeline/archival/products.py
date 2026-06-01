"""The single seam through which per-night data products are written.

``write_night_product`` is the ONE place a night-product write is routed
through. For now it is a thin, behaviour-preserving indirection that reproduces
the legacy writes **byte-for-byte** (``np.save`` for ``.npy``; the existing text
format for ``.txt``). It deliberately adds no new behaviour.

Its PURPOSE is to be the future extension point for, e.g.:
  - a per-night JSON manifest enumerating every product written that night,
  - a consolidated mean-stack archive,
  - a Parquet/NPZ night catalog of star positions + light curves,
  - a nightly report bundle.

When those land, they get wired in HERE (recording metadata, fanning out to
additional sinks) without touching any call site. Until then this stays a
faithful pass-through: ``kind`` selects the legacy write path and ``**meta`` is
accepted-and-ignored so call sites can start passing context now.

Do NOT implement the manifest/catalog/bundle here yet -- only the seam.
"""

import numpy as np

__all__ = ["write_night_product"]


def write_night_product(path, data, *, kind, **meta):
    """Write a single per-night product, reproducing the legacy bytes exactly.

    Parameters
    ----------
    path : path-like
        Destination file (e.g. ``..._pos.npy`` or a light-curve ``.txt``).
    data : object
        The payload. Interpretation depends on ``kind``:
          - ``kind="npy"``  -> ``data`` is a numpy array, written via
            ``np.save(path, data)`` (identical to the legacy call).
          - ``kind="text"`` -> ``data`` is a str (already-formatted file body),
            written verbatim. Callers that build the body line-by-line should
            assemble the string and pass it here so the bytes are unchanged.
    kind : str
        Selects the legacy write path: ``"npy"`` or ``"text"``.
    **meta
        Free-form metadata describing the product (telescope, minute, detect
        threshold, product role, ...). Accepted and IGNORED today; reserved for
        the future manifest/catalog. Passing it now is forward-compatible.

    Returns
    -------
    The ``path`` that was written, for convenience.
    """
    if kind == "npy":
        # Byte-for-byte identical to the legacy ``np.save(posfile, data)``.
        np.save(path, data)
    elif kind == "text":
        # Verbatim write of an already-formatted body (no reformatting).
        with open(path, "w") as filehandle:
            filehandle.write(data)
    else:
        raise ValueError(
            f"write_night_product: unknown kind {kind!r} "
            f"(expected 'npy' or 'text')"
        )

    return path
