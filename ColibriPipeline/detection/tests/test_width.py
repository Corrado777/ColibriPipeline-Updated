"""Tests for detection.width.canonical_width_frames."""

from detection.config import DetectionConfig
from detection.width import canonical_width_frames


def test_opposition_yields_width_at_opposition():
    cfg = DetectionConfig(opposition_angle_deg=0.0)
    w = canonical_width_frames("field1", "2025-08-30", cfg.exposure_time, cfg)
    assert w == cfg.width_at_opposition  # ~6


def test_larger_angle_yields_wider_box():
    cfg0 = DetectionConfig(opposition_angle_deg=0.0)
    cfg60 = DetectionConfig(opposition_angle_deg=60.0)
    w0 = canonical_width_frames(None, "2025-08-30", cfg0.exposure_time, cfg0)
    w60 = canonical_width_frames(None, "2025-08-30", cfg60.exposure_time, cfg60)
    assert w60 > w0


def test_explicit_canonical_width_override():
    cfg = DetectionConfig(canonical_width=11)
    w = canonical_width_frames("field1", "2025-08-30", cfg.exposure_time, cfg)
    assert w == 11


def test_unknown_field_falls_back():
    cfg = DetectionConfig()  # no overrides
    w = canonical_width_frames("does_not_exist", "2025-08-30", cfg.exposure_time, cfg)
    assert w == cfg.width_at_opposition


def test_always_positive():
    cfg = DetectionConfig(opposition_angle_deg=89.9)
    w = canonical_width_frames(None, "2025-08-30", cfg.exposure_time, cfg)
    assert w >= 1
