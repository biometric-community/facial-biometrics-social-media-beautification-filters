"""Filters and metrics, checked on controlled inputs."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from beautyfr.filters.sn_approx import FILTER_NAMES, apply_filter
from beautyfr.metrics import (
    auc_trapezoid,
    best_accuracy_threshold,
    roc_curve_points,
    ssim_gray,
    summarize_verification,
    tpr_at_fpr,
)


@pytest.fixture
def face():
    rng = np.random.RandomState(0)
    small = rng.randint(0, 255, (14, 14, 3), dtype=np.uint8)
    return cv2.resize(small, (112, 112), interpolation=cv2.INTER_CUBIC)


@pytest.mark.parametrize("name", FILTER_NAMES)
def test_filters_keep_the_image_shape(face, name):
    out = apply_filter(face, name)
    assert out.shape == face.shape and out.dtype == np.uint8


def test_original_is_unchanged(face):
    assert np.array_equal(apply_filter(face, "original"), face)


@pytest.mark.parametrize("name", ["uploaded", "thinner_face", "mellow_glow", "belle"])
def test_other_filters_change_the_image(face, name):
    assert not np.array_equal(apply_filter(face, name), face)


def test_unknown_filter_is_rejected(face):
    with pytest.raises(ValueError):
        apply_filter(face, "sepia")


def test_thinner_face_squeezes_most_at_mid_face():
    """Its docstring and the filter it imitates target the cheeks and jaw, mid-image."""
    ramp = np.tile(np.linspace(0, 255, 112, dtype=np.float32), (112, 1))
    image = np.stack([ramp] * 3, axis=2).astype(np.uint8)
    warped = apply_filter(image, "thinner_face").astype(np.float32)
    shift = np.abs(warped[:, :20, 0] - ramp[:, :20]).mean(axis=1)
    assert shift[56] > shift[2]


def windowed_ssim(a: np.ndarray, b: np.ndarray, window: int = 8) -> float:
    """Wang et al. SSIM from local statistics, averaged over sliding windows."""
    x, y = a.astype(np.float64).mean(axis=2), b.astype(np.float64).mean(axis=2)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    pool = lambda t: cv2.blur(t, (window, window))[window:-window, window:-window]
    mu_x, mu_y = pool(x), pool(y)
    var_x, var_y, cov = pool(x * x) - mu_x**2, pool(y * y) - mu_y**2, pool(x * y) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(ssim.mean())


def test_ssim_uses_local_windows():
    """Eq. (1): SSIM compares local structure, which a whole-image statistic hides."""
    rng = np.random.RandomState(1)
    ramp = np.tile(np.linspace(30, 220, 112), (112, 1))
    image = np.clip(ramp + rng.normal(0, 25, (112, 112)), 0, 255)[..., None].repeat(3, axis=2).astype(np.uint8)
    blurred = cv2.GaussianBlur(image, (7, 7), 2.0)
    assert ssim_gray(image, blurred) == pytest.approx(windowed_ssim(image, blurred), abs=0.05)


def test_identical_images_have_unit_ssim(face):
    assert ssim_gray(face, face) == pytest.approx(1.0)


def test_best_accuracy_finds_the_separating_threshold():
    acc, threshold = best_accuracy_threshold(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0]))
    assert acc == 1.0
    assert 0.2 < threshold < 0.8


def test_tpr_at_fpr_matches_a_worked_example():
    scores = np.concatenate([np.linspace(0, 1, 1001), [0.5, 2.0]])
    labels = np.concatenate([np.zeros(1001, dtype=int), np.ones(2, dtype=int)])
    assert tpr_at_fpr(scores, labels, 0.01) == 0.5


def test_auc_of_separable_scores_is_one():
    scores, labels = np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])
    assert auc_trapezoid(*roc_curve_points(scores, labels)) == pytest.approx(1.0)


def test_summary_reports_every_metric():
    summary = summarize_verification(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0]))
    assert summary["accuracy"] == 1.0 and summary["n_pairs"] == 4
