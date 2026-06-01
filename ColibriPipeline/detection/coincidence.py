"""Multi-telescope coincidence for the detection module.

Two combination strategies, mirroring ``DetectionConfig.coincidence_mode``:

* ``post_threshold`` -- the adaptive port of the legacy ``simultaneous_occults``
  AND scheme.  Each scope thresholds independently; surviving detections are
  grouped where they coincide in BOTH time (``time_tolerance_s``) and
  coordinates (``coord_tolerance_deg``, with RA scaled by ``cos(dec)``) across
  the telescopes that were *active* that night.  The tier of a group is the
  number of distinct active scopes that participate, so on an N-active-scope
  night the maximum tier is N.

* ``joint_statistic`` -- the pre-threshold coherent sum/sqrt(N) of aligned
  per-frame scores (ported from ``detection_trade_study.combine``).  Alignment
  is by frame index by default (correct for the bundled sim, whose timestamps
  are quantized to whole seconds), or by interpolation onto a reference scope's
  time grid when per-scope time vectors are supplied.

``active_telescopes`` resolves the env/path scheme of ``simultaneous_occults``
to discover which scopes actually produced detections for a night.

Depends only on numpy + stdlib.  Does NOT import ``detection_trade_study``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Union

import numpy as np


# Telescope name -> archive colour folder, matching simultaneous_occults.
_TELESCOPE_COLORS = {"REDBIRD": "Red", "GREENBIRD": "Green", "BLUEBIRD": "Blue"}
_ALL_SCOPES = ("REDBIRD", "GREENBIRD", "BLUEBIRD")

# Real-telescope archive roots (drive letters), matching simultaneous_occults.
_REAL_BASES = {
    "REDBIRD": Path("R:/"),
    "GREENBIRD": Path("D:/"),
    "BLUEBIRD": Path("B:/"),
}


# --------------------------------------------------------------------------- #
#  Detection record schema
# --------------------------------------------------------------------------- #

@dataclass
class Detection:
    """A single per-scope dip detection passed to ``post_threshold_match``.

    Attributes
    ----------
    time : float or datetime
        Detection time.  May be unix/relative seconds (float) or a ``datetime``;
        comparisons use a uniform seconds-since-epoch conversion so the two are
        interchangeable as long as a single night uses one convention.
    ra, dec : float
        Sky coordinates in degrees.
    path : str or None
        Optional source ``det_*.txt`` path (carried through to the match dict).
    """

    time: Union[float, datetime]
    ra: float
    dec: float
    path: Optional[str] = None


def _time_seconds(t: Union[float, datetime]) -> float:
    """Coerce a detection time to float seconds for tolerance comparisons."""
    if isinstance(t, datetime):
        return t.timestamp()
    return float(t)


# --------------------------------------------------------------------------- #
#  Date / path helpers
# --------------------------------------------------------------------------- #

def _hyphenate_date(obs_date: str) -> str:
    """Normalize 'YYYYMMDD' or 'YYYY-MM-DD' to hyphenated 'YYYY-MM-DD'."""
    s = str(obs_date).strip()
    if "-" in s:
        dt = datetime.strptime(s, "%Y-%m-%d")
    else:
        dt = datetime.strptime(s, "%Y%m%d")
    return dt.strftime("%Y-%m-%d")


def _scope_archive_dir(
    scope: str,
    obs_date: str,
    sim_root: Optional[Union[str, Path]],
    env: Optional[str],
) -> Path:
    """Resolve ``ColibriArchive/<hyphenated-date>`` for one scope.

    Replicates the env/path resolution from ``simultaneous_occults``: in 'sim'
    the base is ``<SIM_ROOT>/{Red,Green,Blue}``; in 'real' the per-scope drive
    roots (R:/ D:/ B:/).
    """
    resolved_env = (env if env is not None
                    else os.environ.get("COLIBRI_ENV", "real")).lower()
    hyphenated = _hyphenate_date(obs_date)

    if resolved_env == "sim":
        root = Path(sim_root) if sim_root is not None else Path(
            os.environ.get(
                "COLIBRI_SIM_ROOT",
                "/home/agirmen/research_data/ColibriPipelineSimulatedDirs",
            )
        )
        base = root / _TELESCOPE_COLORS[scope]
    else:
        base = _REAL_BASES[scope]

    return base / "ColibriArchive" / hyphenated


def active_telescopes(
    obs_date: str,
    sim_root: Optional[Union[str, Path]] = None,
    env: Optional[str] = None,
    *,
    all_scopes: Sequence[str] = _ALL_SCOPES,
) -> Set[str]:
    """Return the scopes that produced ANY ``det_*.txt`` for the night.

    A scope is "active" if its ``ColibriArchive/<hyphenated-date>/`` directory
    exists and contains at least one ``det_*.txt``.  Missing directories are
    silently omitted (a scope that was down that night simply isn't active).

    Parameters
    ----------
    obs_date : str
        Observation date, 'YYYYMMDD' or 'YYYY-MM-DD' (normalized internally).
    sim_root : str or Path, optional
        Override for the sim archive root (testability).  Falls back to
        ``COLIBRI_SIM_ROOT``.
    env : str, optional
        Override for ``COLIBRI_ENV`` ('sim' / 'real'), for testability.

    Returns
    -------
    set of str
        Subset of ``{'REDBIRD', 'GREENBIRD', 'BLUEBIRD'}``.
    """
    active: Set[str] = set()
    for scope in all_scopes:
        archive = _scope_archive_dir(scope, obs_date, sim_root, env)
        try:
            if archive.is_dir() and any(archive.glob("det_*.txt")):
                active.add(scope)
        except OSError:
            # Unreadable directory -> treat as inactive.
            continue
    return active


# --------------------------------------------------------------------------- #
#  Post-threshold AND coincidence
# --------------------------------------------------------------------------- #

def _coords_match(d1: Detection, d2: Detection, coord_tol: float) -> bool:
    """RA/Dec coincidence with RA scaled by cos(dec) (simultaneous_occults)."""
    for d in (d1, d2):
        if not (np.isfinite(d.ra) and np.isfinite(d.dec)):
            return False
    dra = (d1.ra - d2.ra) / np.cos(d1.dec * np.pi / 180.0)
    ddec = d1.dec - d2.dec
    return float(np.hypot(dra, ddec)) <= coord_tol


def _time_match(d1: Detection, d2: Detection, time_tol: float) -> bool:
    return abs(_time_seconds(d1.time) - _time_seconds(d2.time)) <= time_tol


def post_threshold_match(
    active: Set[str],
    detections_by_scope: Dict[str, List[Detection]],
    config,
    *,
    all_scopes: Sequence[str] = _ALL_SCOPES,
) -> List[dict]:
    """Adaptive port of the legacy AND coincidence scheme (pure / in-memory).

    Groups already-thresholded detections that coincide in BOTH time
    (``|dt| <= config.time_tolerance_s``) AND coordinates
    (``hypot((ra1-ra2)/cos(dec*pi/180), dec1-dec2) <= config.coord_tolerance_deg``)
    across the ACTIVE scopes only.  Each group's tier is the number of distinct
    active scopes participating in it (max tier == number of active scopes).

    Single-scope nights (``len(active) == 1``) have no coincidence to enforce:
    every detection passes as a tier-1 group, relying purely on the per-scope
    sigma threshold rather than cross-telescope coincidence.

    Parameters
    ----------
    active : set of str
        Scopes considered active for the night.
    detections_by_scope : dict
        scope -> list of ``Detection`` records.  Scopes absent from ``active``
        are ignored even if present here.
    config : DetectionConfig
        Supplies ``time_tolerance_s`` and ``coord_tolerance_deg``.

    Returns
    -------
    list of dict
        One dict per group: ``{'time', 'ra', 'dec', 'scopes': [...], 'tier': int,
        'paths': {scope: path}}``.  ``time/ra/dec`` are the group representative
        (first member added).
    """
    time_tol = config.time_tolerance_s
    coord_tol = config.coord_tolerance_deg

    # Flat list of (scope, detection) over active scopes only, preserving a
    # deterministic scope order then per-scope detection order.
    items = []  # list of (scope, Detection)
    for scope in all_scopes:
        if scope not in active:
            continue
        for det in detections_by_scope.get(scope, []):
            items.append((scope, det))

    n_items = len(items)

    # Union-find over individual detections; two detections merge when they
    # coincide in BOTH time and coordinates.  This mirrors the legacy pairwise
    # AND test but generalises to any number of active scopes.
    parent = list(range(n_items))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    n_active = len(active)
    # With a single active scope there is no coincidence partner; each
    # detection stands alone as its own group (handled naturally below, since
    # no unions occur).  For N>=2 we test every pair.
    if n_active >= 2:
        for i in range(n_items):
            for j in range(i + 1, n_items):
                di, dj = items[i][1], items[j][1]
                if _time_match(di, dj, time_tol) and _coords_match(di, dj, coord_tol):
                    union(i, j)

    # Collect groups in order of first appearance.
    groups: Dict[int, List[int]] = {}
    order: List[int] = []
    for i in range(n_items):
        root = find(i)
        if root not in groups:
            groups[root] = []
            order.append(root)
        groups[root].append(i)

    matches: List[dict] = []
    for root in order:
        member_idxs = groups[root]
        members = [items[k] for k in member_idxs]
        rep_scope, rep = members[0]
        scopes = []
        paths: Dict[str, Optional[str]] = {}
        for scope, det in members:
            if scope not in scopes:
                scopes.append(scope)
            # First detection per scope wins the path slot.
            if scope not in paths:
                paths[scope] = det.path
        tier = len(scopes)
        matches.append({
            "time": rep.time,
            "ra": rep.ra,
            "dec": rep.dec,
            "scopes": scopes,
            "tier": tier,
            "paths": paths,
        })

    return matches


# --------------------------------------------------------------------------- #
#  Joint-statistic detection
# --------------------------------------------------------------------------- #

def _nan_to_zero(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return np.where(np.isfinite(a), a, 0.0)


def _align_scores(scores_by_scope, times_by_scope=None, ref_scope=None):
    """Align per-scope score rows onto a common grid (ported from combine)."""
    scopes = list(scores_by_scope.keys())
    if ref_scope is None:
        ref_scope = scopes[0]

    if times_by_scope is None:
        n = min(len(scores_by_scope[s]) for s in scopes)
        aligned = np.stack([_nan_to_zero(scores_by_scope[s])[:n] for s in scopes],
                           axis=0)
        return aligned, np.arange(n), scopes

    grid = np.asarray(times_by_scope[ref_scope], dtype=float)
    rows = []
    for s in scopes:
        t = np.asarray(times_by_scope[s], dtype=float)
        v = _nan_to_zero(scores_by_scope[s])
        rows.append(np.interp(grid, t, v, left=0.0, right=0.0))
    return np.stack(rows, axis=0), grid, scopes


def _joint_score(aligned) -> np.ndarray:
    """sum_i aligned[i] / sqrt(n_scopes) (ported from combine)."""
    aligned = np.asarray(aligned, dtype=float)
    n_scopes = aligned.shape[0]
    return aligned.sum(axis=0) / np.sqrt(n_scopes)


def joint_statistic_detect(
    scores_by_scope: Dict[str, np.ndarray],
    threshold: float,
    *,
    times_by_scope: Optional[Dict[str, np.ndarray]] = None,
    ref_scope: Optional[str] = None,
) -> dict:
    """Coherent multi-scope joint statistic detection.

    Aligns the per-scope per-frame scores (by frame index by default, or onto
    ``ref_scope``'s time grid when ``times_by_scope`` is given), combines them
    as ``sum/sqrt(N)``, and thresholds the joint peak.

    Frame-index alignment is the correct default for the bundled sim, whose
    Green/Blue curves are byte-identical and whose GPS timestamps are quantized
    to whole seconds (time alignment would be degenerate).

    Returns
    -------
    dict
        ``{'peak': float, 'frame': int, 'detected': bool, 'joint': ndarray}``.
        ``frame`` indexes the common (reference) grid.
    """
    aligned, _grid, _scopes = _align_scores(scores_by_scope, times_by_scope,
                                            ref_scope=ref_scope)
    joint = _joint_score(aligned)

    if joint.size == 0:
        return {"peak": float("nan"), "frame": 0, "detected": False, "joint": joint}

    frame = int(np.argmax(joint))
    peak = float(joint[frame])
    return {
        "peak": peak,
        "frame": frame,
        "detected": bool(peak >= threshold),
        "joint": joint,
    }
