"""Unit tests for detection.coincidence."""

from __future__ import annotations

import numpy as np
import pytest

from detection.config import DetectionConfig
from detection.coincidence import (
    Detection,
    active_telescopes,
    post_threshold_match,
    joint_statistic_detect,
)


# --------------------------------------------------------------------------- #
#  active_telescopes
# --------------------------------------------------------------------------- #

def _touch_det(archive_root, color, hyphenated_date, name="det_x.txt"):
    d = archive_root / color / "ColibriArchive" / hyphenated_date
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text("dummy\n")


def test_active_telescopes_subset(tmp_path):
    date = "2025-08-30"
    # Red + Green get detections; Blue's dir exists but has no det_*.txt.
    _touch_det(tmp_path, "Red", date, "det_red.txt")
    _touch_det(tmp_path, "Green", date, "det_green.txt")
    (tmp_path / "Blue" / "ColibriArchive" / date).mkdir(parents=True, exist_ok=True)

    active = active_telescopes(date, sim_root=tmp_path, env="sim")
    assert active == {"REDBIRD", "GREENBIRD"}


def test_active_telescopes_all_three_and_date_normalization(tmp_path):
    date = "2025-08-30"
    for color in ("Red", "Green", "Blue"):
        _touch_det(tmp_path, color, date, f"det_{color}.txt")

    # Pass the un-hyphenated form; it must normalize internally.
    active = active_telescopes("20250830", sim_root=tmp_path, env="sim")
    assert active == {"REDBIRD", "GREENBIRD", "BLUEBIRD"}


def test_active_telescopes_single_scope(tmp_path):
    date = "2025-08-30"
    _touch_det(tmp_path, "Green", date, "det_green.txt")
    active = active_telescopes(date, sim_root=tmp_path, env="sim")
    assert active == {"GREENBIRD"}


def test_active_telescopes_missing_dirs(tmp_path):
    # Nothing on disk -> empty set, robust to missing dirs.
    active = active_telescopes("2025-08-30", sim_root=tmp_path, env="sim")
    assert active == set()


# --------------------------------------------------------------------------- #
#  post_threshold_match
# --------------------------------------------------------------------------- #

def _cfg():
    return DetectionConfig()  # time_tolerance_s=0.2, coord_tolerance_deg=0.002


def test_three_scopes_all_coincident_tier3():
    cfg = _cfg()
    active = {"REDBIRD", "GREENBIRD", "BLUEBIRD"}
    dets = {
        "REDBIRD": [Detection(time=100.00, ra=150.0, dec=20.0, path="r")],
        "GREENBIRD": [Detection(time=100.05, ra=150.0005, dec=20.0, path="g")],
        "BLUEBIRD": [Detection(time=99.98, ra=149.9995, dec=20.0001, path="b")],
    }
    matches = post_threshold_match(active, dets, cfg)
    assert len(matches) == 1
    m = matches[0]
    assert m["tier"] == 3
    assert set(m["scopes"]) == {"REDBIRD", "GREENBIRD", "BLUEBIRD"}
    assert set(m["paths"]) == {"REDBIRD", "GREENBIRD", "BLUEBIRD"}


def test_three_scopes_one_off_coord_tier2_plus_tier1():
    cfg = _cfg()
    active = {"REDBIRD", "GREENBIRD", "BLUEBIRD"}
    # Red & Green coincide; Blue same time but far in coordinates (> coord tol).
    dets = {
        "REDBIRD": [Detection(time=100.0, ra=150.0, dec=20.0, path="r")],
        "GREENBIRD": [Detection(time=100.05, ra=150.0005, dec=20.0, path="g")],
        "BLUEBIRD": [Detection(time=100.0, ra=151.0, dec=20.0, path="b")],
    }
    matches = post_threshold_match(active, dets, cfg)
    tiers = sorted(m["tier"] for m in matches)
    assert tiers == [1, 2]

    by_tier = {m["tier"]: m for m in matches}
    assert set(by_tier[2]["scopes"]) == {"REDBIRD", "GREENBIRD"}
    assert by_tier[1]["scopes"] == ["BLUEBIRD"]


def test_two_active_both_coincident_tier2():
    cfg = _cfg()
    active = {"REDBIRD", "GREENBIRD"}
    dets = {
        "REDBIRD": [Detection(time=50.0, ra=10.0, dec=-5.0, path="r")],
        "GREENBIRD": [Detection(time=50.1, ra=10.0005, dec=-5.0, path="g")],
    }
    matches = post_threshold_match(active, dets, cfg)
    assert len(matches) == 1
    assert matches[0]["tier"] == 2
    assert set(matches[0]["scopes"]) == {"REDBIRD", "GREENBIRD"}


def test_single_active_every_detection_tier1():
    cfg = _cfg()
    active = {"GREENBIRD"}
    dets = {
        "GREENBIRD": [
            Detection(time=10.0, ra=1.0, dec=2.0, path="a"),
            Detection(time=20.0, ra=3.0, dec=4.0, path="b"),
            # Even two close-together detections stay separate -- single-scope
            # night has no coincidence to enforce.
            Detection(time=10.05, ra=1.0, dec=2.0, path="c"),
        ]
    }
    matches = post_threshold_match(active, dets, cfg)
    assert len(matches) == 3
    assert all(m["tier"] == 1 for m in matches)


def test_coord_fail_does_not_merge_even_if_time_matches():
    cfg = _cfg()
    active = {"REDBIRD", "GREENBIRD"}
    dets = {
        "REDBIRD": [Detection(time=100.0, ra=150.0, dec=20.0, path="r")],
        # Same time, but RA off by 1 deg (>> coord tol).
        "GREENBIRD": [Detection(time=100.0, ra=151.0, dec=20.0, path="g")],
    }
    matches = post_threshold_match(active, dets, cfg)
    assert sorted(m["tier"] for m in matches) == [1, 1]


def test_time_fail_does_not_merge_even_if_coords_match():
    cfg = _cfg()
    active = {"REDBIRD", "GREENBIRD"}
    dets = {
        "REDBIRD": [Detection(time=100.0, ra=150.0, dec=20.0, path="r")],
        # Same coords, but 5 s apart (>> time tol).
        "GREENBIRD": [Detection(time=105.0, ra=150.0, dec=20.0, path="g")],
    }
    matches = post_threshold_match(active, dets, cfg)
    assert sorted(m["tier"] for m in matches) == [1, 1]


# --------------------------------------------------------------------------- #
#  joint_statistic_detect
# --------------------------------------------------------------------------- #

def test_joint_three_identical_peaks_scale_by_sqrt3():
    n, k, s = 100, 37, 4.0
    base = np.zeros(n)
    base[k] = s
    scores = {sc: base.copy() for sc in ("REDBIRD", "GREENBIRD", "BLUEBIRD")}

    # Threshold below s*sqrt(3) -> detected.
    res = joint_statistic_detect(scores, threshold=s * np.sqrt(3) - 0.5)
    assert res["frame"] == k
    assert res["peak"] == pytest.approx(s * np.sqrt(3))
    assert res["detected"] is True

    # Threshold above -> not detected.
    res2 = joint_statistic_detect(scores, threshold=s * np.sqrt(3) + 0.5)
    assert res2["detected"] is False


def test_joint_noise_stays_unit_variance():
    rng = np.random.default_rng(0)
    n = 20000
    scores = {sc: rng.standard_normal(n)
              for sc in ("REDBIRD", "GREENBIRD", "BLUEBIRD")}
    res = joint_statistic_detect(scores, threshold=100.0)  # threshold irrelevant
    joint_std = float(np.std(res["joint"]))
    # Independent unit-variance noise -> joint std ~ 1.
    assert joint_std == pytest.approx(1.0, abs=0.05)


def test_joint_unequal_lengths_truncate_to_min():
    scores = {
        "A": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "B": np.array([1.0, 2.0, 3.0]),
        "C": np.array([1.0, 2.0, 3.0, 4.0]),
    }
    res = joint_statistic_detect(scores, threshold=0.0)
    assert res["joint"].shape == (3,)
