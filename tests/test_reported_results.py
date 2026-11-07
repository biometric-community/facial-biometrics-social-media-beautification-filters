"""REPORT.md, the committed output of the full run, against the published results."""

from __future__ import annotations

import re

import pytest


def row(report: str, name: str) -> tuple[float, float, float]:
    match = re.search(rf"\| {name} \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", report)
    assert match, f"{name} row missing from REPORT.md"
    return tuple(float(g) for g in match.groups())


@pytest.mark.parametrize("name", ["uploaded", "thinner_face", "mellow_glow", "belle"])
def test_filter_aggressivity_matches_table_3(report, targets, name):
    assert row(report, name)[2] * 100 == pytest.approx(targets["ssim_pct"][name], abs=targets["ssim_tolerance"])


def test_original_accuracy_matches_table_4(report, targets):
    assert row(report, "original")[0] == pytest.approx(targets["original_accuracy"], abs=targets["tolerance_pp"])


@pytest.mark.parametrize("name", ["uploaded", "thinner_face", "mellow_glow", "belle"])
def test_filters_barely_change_face_verification(report, targets, name):
    """Abstract: less aggressive filters do not adversely affect face recognition."""
    assert row(report, "original")[0] - row(report, name)[0] <= targets["max_filter_drop_pp"]
