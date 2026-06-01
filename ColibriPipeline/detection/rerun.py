"""Rerun detection on archived ``*_stars.npy`` minutes.

Offline harness that replays a chosen :class:`DetectionConfig` over an already
archived night and writes ``det_*.txt`` files matching the *live* primary-detection
layout (``colibri_main_py3``), so downstream parsers (``coordsfinder.readFile`` /
``readRAdec``, ``simultaneous_occults.readRAdec``) can consume the output.

Key differences from the live writer:

* Input is the archived ``*_stars.npy`` (no RCD frames), so there are no real
  image filenames. The "filename" column is synthesised as ``<minute_stem>_<frame>``
  (last ``_<int>`` token, so ``readFile``'s frame parsing still works).
* RA/Dec come from the sibling ``*sig_pos.npy`` file when available (it shares
  star ordering with ``stars.npy``, so ``pos[i, 3:5]`` is star i's RA/Dec). These
  real coordinates are written into the RA/Dec header line (line index 6, which
  ``coordsfinder`` overwrites in the live flow and ``readRAdec`` reads). When no
  pos file (or no WCS columns) is available, a ``nan nan`` placeholder is written.
* The per-minute matched-filter width is pos-derived: the minute's field center
  RA/Dec (wrap-safe circular-mean RA, median Dec over finite pos rows) feeds
  ``canonical_width_from_radec`` so off-opposition minutes get a wider box. With
  no pos file the width falls back to the field-name/constant ``canonical_width_frames``.
* The DATE-OBS time column / header uses the ``unix_time`` axis from the
  ``stars.npy`` file, rendered ISO with a ``T`` separator (so the
  ``DATE-OBS`` line stays at index 7 and parses the same way).

Output is written into a SEPARATE directory ``<night>/<out_tag>_<detector>/`` so
the live ``det_*.txt`` are never clobbered.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import DetectionConfig, EXCLUSION_ZONE  # noqa: F401  (EXCLUSION_ZONE kept for parity)
from .coincidence import Detection, active_telescopes, post_threshold_match
from .detector import make_detector
from .width import canonical_width_frames, canonical_width_from_radec

# Seconds to save on either side of the event (mirror colibri_main_py3.EVENT_WIDTH).
EVENT_WIDTH = 1.0


# ---------------------------------------------------------------------------
# Local stars.npy loader (replicates detection_trade_study.lightcurves.load_minute
# WITHOUT importing that package).
# ---------------------------------------------------------------------------
def _load_minute(stars_npy_path):
    """Load a ``*_stars.npy`` file (shape ``(n_frames, n_stars, 4)``).

    Last axis layout ``[x, y, flux, unix_time]``. Returns a dict::

        {'flux': (n_stars, n_frames), 'time': (n_frames,), 'xy': (n_stars, 2)}
    """
    arr = np.load(stars_npy_path, allow_pickle=True)
    return {
        'flux': arr[:, :, 2].T,   # (n_stars, n_frames)
        'time': arr[:, 0, 3],     # (n_frames,)
        'xy': arr[0, :, 0:2],     # (n_stars, 2)
    }


def _unix_to_iso(unix_time):
    """Render a unix timestamp as an ISO string with a ``T`` separator.

    Matching the live header's DATE-OBS, the time portion is recoverable via
    ``line.split('T')[2]`` (the literal "DATE-OBS" contributes the first 'T').
    """
    try:
        dt = datetime.fromtimestamp(float(unix_time), tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return "NaT"
    # microseconds always present so split('.') in the live name parsing works
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')


def _minute_stem(stars_npy_path):
    """``20250830_01.54.12.147_stars.npy`` -> ``20250830_01.54.12.147``."""
    name = Path(stars_npy_path).name
    if name.endswith('_stars.npy'):
        return name[: -len('_stars.npy')]
    return Path(stars_npy_path).stem


def _load_pos_radec(stars_npy_path, stem, n_stars):
    """Load the sibling ``*sig_pos.npy`` RA/Dec for a minute, if usable.

    Globs ``<stem>_*sig_pos.npy`` in the same directory as the stars file. Returns
    the ``(n_stars, 2)`` RA/Dec slice (``pos[:, 3:5]``) only when the pos array is
    2-D with >= 5 columns AND ``len(pos) == n_stars`` (so star ordering matches
    the stars file by index). Otherwise returns None.
    """
    stars_npy_path = Path(stars_npy_path)
    cands = sorted(stars_npy_path.parent.glob(f'{stem}_*sig_pos.npy'))
    for cand in cands:
        try:
            pos = np.load(cand, allow_pickle=True)
        except Exception:
            continue
        pos = np.asarray(pos, dtype=float)
        if pos.ndim == 2 and pos.shape[1] >= 5 and pos.shape[0] == n_stars:
            return pos[:, 3:5]
    return None


def _field_center_from_radec(pos_radec):
    """Compute a field-center (RA, Dec) from per-star RA/Dec (finite rows only).

    RA via a wrap-safe circular mean (``atan2(mean sin, mean cos)``) normalized to
    [0, 360); Dec via the median. Returns ``(ra, dec)`` floats, or ``(nan, nan)``
    when no finite rows exist.
    """
    radec = np.asarray(pos_radec, dtype=float)
    finite = np.isfinite(radec[:, 0]) & np.isfinite(radec[:, 1])
    if not np.any(finite):
        return float('nan'), float('nan')
    ra = radec[finite, 0]
    dec = radec[finite, 1]
    ra_rad = np.radians(ra)
    mean_sin = np.mean(np.sin(ra_rad))
    mean_cos = np.mean(np.cos(ra_rad))
    center_ra = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0
    center_dec = float(np.median(dec))
    return float(center_ra), center_dec


def _field_from_stem(stem):
    """Derive a categorical field name from the minute stem prefix, if any.

    Live minute names are ``<field?><YYYYMMDD>_<HH.MM.SS.mmm>``. If a non-digit
    prefix precedes the date, treat it as the field name; otherwise return None.
    """
    head = stem.split('_', 1)[0]
    # strip a trailing 8-digit date if present
    if len(head) > 8 and head[-8:].isdigit() and not head[:-8].isdigit():
        return head[:-8]
    return None


# ---------------------------------------------------------------------------
# Per-minute rerun
# ---------------------------------------------------------------------------
def rerun_minute(stars_npy_path, config, *, field_name=None, obs_date=None,
                 out_dir, telescope):
    """Rerun detection over one archived minute; write ``det_*.txt`` per detection.

    Parameters
    ----------
    stars_npy_path : str or Path
        Path to a ``*_stars.npy`` file.
    config : DetectionConfig
    field_name : str, optional
        Categorical field name (for canonical-width auto-compute). May be None.
    obs_date : optional
        Observation date passed to ``canonical_width_frames`` (ephemeris epoch).
    out_dir : str or Path
        Directory the ``det_*.txt`` files are written into (created if needed).
    telescope : str
        Telescope label (used in the filename + header).

    Returns
    -------
    list of dict
        One entry per detection:
        ``{'star', 'frame', 'time', 'significance', 'x', 'y', 'path'}``.
    """
    stars_npy_path = Path(stars_npy_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_minute(stars_npy_path)
    flux = data['flux']            # (n_stars, n_frames)
    times = np.asarray(data['time'], dtype=float)  # (n_frames,)
    xy = data['xy']                # (n_stars, 2)
    n_frames = flux.shape[1]

    stem = _minute_stem(stars_npy_path)
    if field_name is None:
        field_name = _field_from_stem(stem)

    n_stars = flux.shape[0]

    # Real RA/Dec from the sibling pos file (shares star ordering with stars.npy).
    pos_radec = _load_pos_radec(stars_npy_path, stem, n_stars)

    # Build the detector ONCE per minute. Width is pos-derived (field-center
    # opposition angle) when RA/Dec are available, else the field-name fallback.
    if pos_radec is not None:
        center_ra, center_dec = _field_center_from_radec(pos_radec)
        width = canonical_width_from_radec(center_ra, center_dec, obs_date,
                                           config.exposure_time, config)
    else:
        width = canonical_width_frames(field_name, obs_date,
                                       config.exposure_time, config)
    detector = make_detector(config, width)

    save_chunk = int(round(EVENT_WIDTH / config.exposure_time))

    detections = []
    for star_idx in range(flux.shape[0]):
        (frameNum, lc_arr, conv_padded, lc_std, lc_mean,
         bkg_std, bkg_mean, minVal, significance) = detector.detect(flux[star_idx])

        if not (isinstance(frameNum, (int, np.integer)) and frameNum > 0):
            continue

        # The detection frame is in the TRIMMED light-curve frame. We don't have
        # the trim offset cheaply; map it back to the original by aligning the
        # trimmed length against the original (leading-zero offset).
        n_lc = len(lc_arr)
        lead = _leading_zero_offset(flux[star_idx], n_lc)
        orig_frame = int(frameNum) + lead
        orig_frame = max(0, min(n_frames - 1, orig_frame))

        ts_unix = float(times[orig_frame]) if orig_frame < len(times) else float('nan')
        ts_iso = _unix_to_iso(ts_unix)
        star_x = float(xy[star_idx, 0])
        star_y = float(xy[star_idx, 1])

        if pos_radec is not None:
            star_ra = float(pos_radec[star_idx, 0])
            star_dec = float(pos_radec[star_idx, 1])
        else:
            star_ra = float('nan')
            star_dec = float('nan')

        # filename components (mirror live naming).
        date = ts_iso.split('T')[0]
        if 'T' in ts_iso and '.' in ts_iso:
            time_part = ts_iso.split('T')[1].split('.')[0].replace(':', '')
            mstime = ts_iso.split('T')[1].split('.')[1]
        else:
            time_part = '000000'
            mstime = '000000'

        savefile = out_dir / (
            f'det_{date}_{time_part}_{mstime}_star{star_idx}_{telescope}.txt'
        )

        _write_det_file(
            savefile,
            event_frame=orig_frame,
            stem=stem,
            star_x=star_x, star_y=star_y,
            star_ra=star_ra, star_dec=star_dec,
            ts_iso=ts_iso,
            telescope=telescope, field_name=field_name,
            significance=float(significance),
            lc_std=float(lc_std), lc_mean=float(lc_mean),
            bkg_std=float(bkg_std), bkg_mean=float(bkg_mean),
            minVal=float(minVal),
            lc_arr=np.asarray(lc_arr, dtype=float),
            conv_padded=np.asarray(conv_padded, dtype=float),
            times=times, lead=lead, save_chunk=save_chunk,
        )

        detections.append({
            'star': int(star_idx),
            'frame': int(orig_frame),
            'time': ts_unix,
            'significance': float(significance),
            'x': star_x,
            'y': star_y,
            'ra': star_ra,
            'dec': star_dec,
            'path': str(savefile),
        })

    return detections


def _leading_zero_offset(flux_1d, trimmed_len):
    """Number of leading zeros trimmed by ``np.trim_zeros`` for this curve.

    ``detect`` runs ``np.trim_zeros``; to map a trimmed-frame index back to the
    original frame index we add the count of leading zeros.
    """
    f = np.asarray(flux_1d, dtype=float)
    if trimmed_len <= 0 or trimmed_len >= len(f):
        return 0
    lead = 0
    for v in f:
        if v == 0.0:
            lead += 1
        else:
            break
    return lead


def _write_det_file(savefile, *, event_frame, stem, star_x, star_y,
                    star_ra, star_dec, ts_iso,
                    telescope, field_name, significance, lc_std, lc_mean,
                    bkg_std, bkg_mean, minVal, lc_arr, conv_padded, times,
                    lead, save_chunk):
    """Write a single ``det_*.txt`` matching the live header + data layout.

    Header line indices (verified against colibri_main_py3 lines 607-624 and the
    downstream parsers):

        0-3  : '#'
        4    : '#    Event File: <synthetic>'
        5    : '#    Star Coords: <x> <y>'
        6    : '#    RA Dec Coords: <ra> <dec>' (from pos.npy; nan nan if absent; readRAdec reads idx 6)
        7    : '#    DATE-OBS: <iso>'          (readFile event time, idx 7)
        8    : '#    Telescope: <tel>'
        9    : '#    Field: <field>'
        10   : '#    significance: <sig>'      (readFile event_type, idx 10)
        11-14: lightcurve std/mean, conv bkg std/mean
        15   : '#    Convolution minimal value: <minVal>'
        16-17: '#'
        18   : '#filename     time      flux     conv_flux'
        19+  : data rows
    """
    field_str = field_name if field_name is not None else 'unknown'

    # synthetic Event File: must end in _<frame>.<ext> so readFile idx-4 parses it.
    event_file = f'{stem}_{int(event_frame)}.rcd'

    # window of frames to save around the event (mirror live save_chunk logic).
    n = len(lc_arr)
    # event_frame is original-frame; the lc/conv arrays are trimmed-frame.
    ev_trim = int(event_frame) - lead
    ev_trim = max(0, min(n - 1, ev_trim))

    if ev_trim - save_chunk <= 0:
        lo, hi = 0, min(n, ev_trim + save_chunk)
    elif ev_trim + save_chunk >= n:
        lo, hi = max(0, ev_trim - save_chunk), n
    else:
        lo, hi = ev_trim - save_chunk, ev_trim + save_chunk

    with open(savefile, 'w') as fh:
        fh.write('#\n#\n#\n#\n')
        fh.write('#    Event File: %s\n' % event_file)              # idx 4
        fh.write('#    Star Coords: %f %f\n' % (star_x, star_y))    # idx 5
        fh.write('#    RA Dec Coords: %f %f\n' % (star_ra, star_dec))  # idx 6
        fh.write('#    DATE-OBS: %s\n' % ts_iso)                    # idx 7
        fh.write('#    Telescope: %s\n' % telescope)                # idx 8
        fh.write('#    Field: %s\n' % field_str)                    # idx 9
        fh.write('#    significance: %.3f\n' % significance)        # idx 10
        fh.write('#    Raw lightcurve std: %.4f\n' % lc_std)        # idx 11
        fh.write('#    Raw lightcurve mean: %.4f\n' % lc_mean)      # idx 12
        fh.write('#    Convolution background std: %.4f\n' % bkg_std)   # idx 13
        fh.write('#    Convolution background mean: %.4f\n' % bkg_mean)  # idx 14
        fh.write('#    Convolution minimal value: %.4f\n' % minVal)  # idx 15
        fh.write('#\n#\n')                                          # idx 16,17
        fh.write('#filename     time      flux     conv_flux\n')    # idx 18

        for i in range(lo, hi):
            frame_idx = i + lead
            fname = f'{stem}_{frame_idx}.rcd'
            if frame_idx < len(times):
                t_unix = float(times[frame_idx])
            else:
                t_unix = float('nan')
            # The live writer stores only the seconds component of the timestamp;
            # mirror that so downstream `time` columns are comparable.
            iso = _unix_to_iso(t_unix)
            if 'T' in iso:
                try:
                    sec_col = float(iso.split('T')[1].split(':')[2])
                except (IndexError, ValueError):
                    sec_col = t_unix
            else:
                sec_col = t_unix
            fh.write('%s %f  %f  %f\n' % (fname, sec_col, lc_arr[i], conv_padded[i]))


# ---------------------------------------------------------------------------
# Per-night rerun
# ---------------------------------------------------------------------------
def rerun_night(telescope, date, config, *, out_tag='rerun',
                archive_root=None, sim_root=None):
    """Rerun detection over a whole archived night.

    Resolves the night dir via ``lightcurve_analysis.paths.resolve_archive_root``,
    globs ``*_stars.npy``, and writes output into a SEPARATE
    ``<night>/<out_tag>_<detector>/`` directory so live ``det_*.txt`` are never
    clobbered. Also writes ``rerun_summary.csv`` there.

    Returns
    -------
    dict
        ``{'out_dir', 'n_minutes', 'n_detections', 'detections': [...]}``.
    """
    from lightcurve_analysis.paths import resolve_archive_root, night_dir

    root = resolve_archive_root(archive_root=archive_root,
                                telescope=telescope, sim_root=sim_root)
    ndir = night_dir(root, date)

    out_dir = ndir / f'{out_tag}_{config.detector}'
    out_dir.mkdir(parents=True, exist_ok=True)

    stars_files = sorted(ndir.glob('*_stars.npy'))

    all_detections = []
    for spath in stars_files:
        stem = _minute_stem(spath)
        field_name = _field_from_stem(stem)
        dets = rerun_minute(
            spath, config,
            field_name=field_name, obs_date=date,
            out_dir=out_dir, telescope=telescope,
        )
        for d in dets:
            d['minute'] = stem
        all_detections.extend(dets)

    # summary CSV
    summary_path = out_dir / 'rerun_summary.csv'
    with open(summary_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['star', 'minute', 'frame', 'time', 'significance'])
        for d in all_detections:
            writer.writerow([d['star'], d.get('minute', ''), d['frame'],
                             d['time'], d['significance']])

    return {
        'out_dir': str(out_dir),
        'n_minutes': len(stars_files),
        'n_detections': len(all_detections),
        'detections': all_detections,
    }


# ---------------------------------------------------------------------------
# Matched generation across telescopes
# ---------------------------------------------------------------------------
_TELESCOPE_COLORS = {"REDBIRD": "Red", "GREENBIRD": "Green", "BLUEBIRD": "Blue"}
_COLOR_TO_SCOPE = {v.upper(): k for k, v in _TELESCOPE_COLORS.items()}

# Mirror simultaneous_occults' directory naming for matched dirs.
_BARE_FORMAT = '%Y-%m-%d_%H%M%S_%f'


def _normalize_scope(label):
    """Normalize a telescope label to REDBIRD/GREENBIRD/BLUEBIRD."""
    key = str(label).strip().upper()
    if key in _TELESCOPE_COLORS:
        return key
    if key in _COLOR_TO_SCOPE:
        return _COLOR_TO_SCOPE[key]
    return key


def rerun_match(date, config, *, telescopes=None, out_tag='rerun',
                archive_root=None, sim_root=None):
    """Rerun detection on each active telescope and cross-match the results.

    Reruns every active (or requested) telescope for ``date`` via
    :func:`rerun_night`, builds :class:`detection.coincidence.Detection` records
    from the in-memory detections (now carrying real RA/Dec from pos.npy), and
    runs ``post_threshold_match`` to find time+coord coincidences across scopes.

    Matched groups (tier >= 2, or all when only one scope is active) are written
    under a SHARED matched location ``<Green-night>/<out_tag>_<detector>/matched/``
    as ``<BARE timestamp>-Tier<N>/`` directories, each holding copies of the
    participating ``det_*.txt`` files. A ``matched_summary.csv`` is also written.

    Parameters
    ----------
    date : str
        Night date (YYYY-MM-DD or YYYYMMDD).
    config : DetectionConfig
    telescopes : list, optional
        Explicit scopes to rerun/match (normalized to canonical names). If None,
        uses ``sorted(active_telescopes(date, sim_root=sim_root))``.
    out_tag : str
    archive_root, sim_root : optional path overrides.

    Returns
    -------
    dict
        ``{'matched_dir', 'n_active', 'active', 'n_matches', 'matches': [...]}``.
    """
    from lightcurve_analysis.paths import resolve_archive_root, night_dir

    if telescopes is not None:
        active = [_normalize_scope(t) for t in telescopes]
    else:
        active = sorted(active_telescopes(date, sim_root=sim_root))

    detections_by_scope = {}
    for scope in active:
        res = rerun_night(scope, date, config, out_tag=out_tag,
                          archive_root=archive_root, sim_root=sim_root)
        detections_by_scope[scope] = [
            Detection(time=d['time'], ra=d['ra'], dec=d['dec'], path=d['path'])
            for d in res['detections']
        ]

    matches = post_threshold_match(set(active), detections_by_scope, config)

    # Shared matched location anchored on GREENBIRD's night dir.
    green_root = resolve_archive_root(archive_root=archive_root,
                                      telescope='GREENBIRD', sim_root=sim_root)
    matched_dir = night_dir(green_root, date) / f'{out_tag}_{config.detector}' / 'matched'
    matched_dir.mkdir(parents=True, exist_ok=True)

    n_active = len(active)
    written = []
    for match in matches:
        tier = match['tier']
        # Require a real cross-scope coincidence on multi-scope nights.
        if tier < 2 and n_active >= 2:
            continue

        rep_time = match['time']
        if not isinstance(rep_time, datetime):
            try:
                rep_time = datetime.fromtimestamp(float(rep_time))
            except (ValueError, OverflowError, OSError):
                rep_time = datetime.fromtimestamp(0)
        dir_name = rep_time.strftime(_BARE_FORMAT)

        match_dir = matched_dir / f'{dir_name}-Tier{tier}'
        match_dir.mkdir(parents=True, exist_ok=True)

        for scope in match['scopes']:
            det_path = match['paths'].get(scope)
            if det_path is not None and Path(det_path).exists():
                shutil.copy(det_path, match_dir)

        written.append({
            'timestamp': dir_name,
            'tier': tier,
            'scopes': match['scopes'],
            'dir': str(match_dir),
        })

    # Matched summary CSV.
    summary_path = matched_dir / 'matched_summary.csv'
    with open(summary_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['timestamp', 'tier', 'scopes'])
        for w in written:
            writer.writerow([w['timestamp'], w['tier'], ';'.join(w['scopes'])])

    return {
        'matched_dir': str(matched_dir),
        'n_active': n_active,
        'active': list(active),
        'n_matches': len(written),
        'matches': written,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser():
    p = argparse.ArgumentParser(
        prog='python -m detection.rerun',
        description='Rerun dip detection over an archived Colibri night.',
    )
    p.add_argument('--telescope', required=True,
                   help='Telescope label (REDBIRD/GREENBIRD/BLUEBIRD or color).')
    p.add_argument('--date', required=True, help='Night date (YYYY-MM-DD).')
    p.add_argument('--detector', default='box', choices=['box', 'ricker'],
                   help='Detector to use (default: box).')
    p.add_argument('--sigma', type=float, default=None,
                   help='Significance threshold (default: DetectionConfig default).')
    p.add_argument('--canonical-width', type=int, default=None,
                   help='Explicit box width in frames (overrides auto-compute).')
    p.add_argument('--opposition-angle', type=float, default=None,
                   help='Explicit opposition angle in degrees.')
    p.add_argument('--coincidence-mode', default=None,
                   choices=['post_threshold', 'joint_statistic'],
                   help='Coincidence mode (default: DetectionConfig default).')
    p.add_argument('--out-tag', default='rerun',
                   help='Prefix for the output dir (<out-tag>_<detector>).')
    p.add_argument('--archive-root', default=None,
                   help='Explicit archive root (overrides env resolution).')
    p.add_argument('--sim-root', default=None,
                   help='Sim root override (else COLIBRI_SIM_ROOT).')
    p.add_argument('--match', action='store_true',
                   help='Cross-match detections across active telescopes and '
                        'write matched/<ts>-TierN dirs.')
    p.add_argument('--telescopes', default=None,
                   help='Comma-separated scopes to match (default: active set).')
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    kwargs = {'detector': args.detector}
    if args.sigma is not None:
        kwargs['sigma_threshold'] = args.sigma
    if args.canonical_width is not None:
        kwargs['canonical_width'] = args.canonical_width
    if args.opposition_angle is not None:
        kwargs['opposition_angle_deg'] = args.opposition_angle
    if args.coincidence_mode is not None:
        kwargs['coincidence_mode'] = args.coincidence_mode

    config = DetectionConfig(**kwargs)

    if args.match:
        telescopes = None
        if args.telescopes:
            telescopes = [t for t in args.telescopes.split(',') if t.strip()]
        match_result = rerun_match(
            args.date, config,
            telescopes=telescopes,
            out_tag=args.out_tag,
            archive_root=args.archive_root,
            sim_root=args.sim_root,
        )
        print('Rerun matched summary')
        print('  date         :', args.date)
        print('  detector     :', config.detector)
        print('  n active     :', match_result['n_active'])
        print('  active       :', match_result['active'])
        print('  n matches    :', match_result['n_matches'])
        print('  matched dir  :', match_result['matched_dir'])
        return 0

    result = rerun_night(
        args.telescope, args.date, config,
        out_tag=args.out_tag,
        archive_root=args.archive_root,
        sim_root=args.sim_root,
    )

    print('Rerun detection summary')
    print('  telescope    :', args.telescope)
    print('  date         :', args.date)
    print('  detector     :', config.detector)
    print('  sigma        :', config.sigma_threshold)
    print('  minutes      :', result['n_minutes'])
    print('  detections   :', result['n_detections'])
    print('  out dir      :', result['out_dir'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
