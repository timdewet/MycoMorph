"""Tests for the hmax / radial_symmetry detectors and merged-pair splitting."""

from __future__ import annotations

import numpy as np

from mycomorph.core.foci import DetectorOpts, Focus
from mycomorph.core.foci.decompose import split_merged_foci
from mycomorph.core.foci.detectors import (
    ALL_KEYS,
    CLASSICAL_BASELINE_KEYS,
    DIM_SIGNAL_KEYS,
    REGISTRY,
)


def _render_spots(
    shape: tuple[int, int],
    spots: list[tuple[float, float, float, float]],
    background: float = 100.0,
    noise_sigma: float = 2.0,
    seed: int = 7,
) -> np.ndarray:
    """Gaussian spots (y, x, amplitude, sigma) on a noisy flat background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float64)
    img = np.full(shape, background, dtype=np.float64)
    for y, x, amp, sig in spots:
        img += amp * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * sig * sig))
    img += rng.normal(0.0, noise_sigma, size=shape)
    return img


SPOTS = [
    (20.3, 30.7, 60.0, 1.4),
    (50.5, 80.2, 45.0, 1.5),
    (75.8, 25.4, 80.0, 1.3),
    (90.1, 90.6, 55.0, 1.6),
]


def _distances_to_truth(foci, truth):
    """Per-truth-spot distance to the nearest detection (inf if none)."""
    out = []
    for y, x, _amp, _sig in truth:
        if not foci:
            out.append(np.inf)
            continue
        out.append(min(np.hypot(f.y - y, f.x - x) for f in foci))
    return out


# ── registry wiring ──────────────────────────────────────────────────────────

def test_new_keys_registered():
    assert "hmax" in REGISTRY
    assert "radial_symmetry" in REGISTRY
    assert "hmax" in CLASSICAL_BASELINE_KEYS
    assert "radial_symmetry" in DIM_SIGNAL_KEYS
    assert "hmax" in ALL_KEYS
    assert "radial_symmetry" in ALL_KEYS
    # Factory contract: instantiable, correct .name, has fit/detect.
    for key in ("hmax", "radial_symmetry"):
        det = REGISTRY[key]()
        assert det.name == key
        assert det.fit(None) is None
        assert callable(det.detect)


# ── h-maxima ─────────────────────────────────────────────────────────────────

def test_hmax_detects_known_spots():
    img = _render_spots((120, 120), SPOTS)
    det = REGISTRY["hmax"]()
    foci = det.detect(img, None, DetectorOpts())
    dists = _distances_to_truth(foci, SPOTS)
    assert max(dists) < 1.2, f"missed a spot: distances {dists}"
    # Permissive-by-design: a handful of noise peaks clearing the h/SNR
    # gates is expected (downstream feature filtering removes them) —
    # this bound catches pathological blowups only.
    assert len(foci) <= 3 * len(SPOTS), f"too many false positives: {len(foci)}"


def test_hmax_empty_image():
    det = REGISTRY["hmax"]()
    assert det.detect(np.zeros((64, 64)), None, DetectorOpts()) == []


# ── radial symmetry ──────────────────────────────────────────────────────────

def test_radial_symmetry_detects_and_localises():
    img = _render_spots((120, 120), SPOTS)
    det = REGISTRY["radial_symmetry"]()
    foci = det.detect(img, None, DetectorOpts())
    dists = _distances_to_truth(foci, SPOTS)
    assert max(dists) < 1.2, f"missed a spot: distances {dists}"
    # The point of the detector: sub-pixel localisation on dim spots.
    assert float(np.mean(dists)) < 0.4, f"poor localisation: {dists}"
    assert len(foci) <= len(SPOTS) + 4, f"too many false positives: {len(foci)}"


def test_radial_symmetry_empty_image():
    det = REGISTRY["radial_symmetry"]()
    assert det.detect(np.zeros((64, 64)), None, DetectorOpts()) == []


def test_radial_symmetry_assigns_cell_labels(labeled_mask_3cells):
    # One bright spot inside cell 2 (rows 20..30, cols 20..40).
    img = _render_spots((50, 60), [(25.0, 30.0, 80.0, 1.4)])
    det = REGISTRY["radial_symmetry"]()
    foci = det.detect(img, labeled_mask_3cells, DetectorOpts())
    match = [f for f in foci if np.hypot(f.y - 25.0, f.x - 30.0) < 1.2]
    assert match and match[0].cell_label == 2


# ── merged-pair splitting ────────────────────────────────────────────────────

def _pair_image():
    """Two same-PSF spots 4 px apart — merges into one detection under
    local-maxima spacing at default opts."""
    pair = [(40.0, 38.5, 80.0, 1.3), (40.0, 42.5, 70.0, 1.3)]
    return _render_spots((80, 80), pair), pair


def test_split_merged_recovers_pair():
    img, pair = _pair_image()
    parent = Focus(y=40.0, x=40.4, sigma=2.4, intensity=90.0, snr=10.0)
    opts = DetectorOpts(snr_min=2.0, split_merged=True)
    out = split_merged_foci(img, [parent], opts)
    assert len(out) == 2, "merged pair was not split"
    dists = _distances_to_truth(out, pair)
    assert max(dists) < 0.8, f"split children off-target: {dists}"


def test_split_leaves_isolated_spot_alone():
    spot = (40.2, 40.6, 80.0, 1.4)
    img = _render_spots((80, 80), [spot])
    parent = Focus(y=spot[0], x=spot[1], sigma=1.4, intensity=80.0, snr=10.0)
    opts = DetectorOpts(snr_min=2.0, split_merged=True)
    out = split_merged_foci(img, [parent], opts)
    assert len(out) == 1
    assert np.hypot(out[0].y - spot[0], out[0].x - spot[1]) < 1e-9


def test_split_respects_already_resolved_neighbours():
    img, pair = _pair_image()
    parents = [
        Focus(y=pair[0][0], x=pair[0][1], sigma=1.3, intensity=80.0, snr=10.0),
        Focus(y=pair[1][0], x=pair[1][1], sigma=1.3, intensity=70.0, snr=10.0),
    ]
    opts = DetectorOpts(snr_min=2.0, split_merged=True)
    out = split_merged_foci(img, parents, opts)
    # Both spots already have their own detection — nothing may double up.
    assert len(out) == 2
    for parent, child in zip(parents, out):
        assert np.hypot(child.y - parent.y, child.x - parent.x) < 1e-9


def test_split_noop_on_empty_list():
    img, _ = _pair_image()
    assert split_merged_foci(img, [], DetectorOpts()) == []


def test_detector_plus_split_resolves_doublet():
    """End-to-end composition: each detector's output through the
    splitter yields exactly the two true foci in the pair region —
    including when refinement initially half-resolves the doublet
    (post-refinement duplicate suppression must collapse it first).
    """
    opts = DetectorOpts(snr_min=2.0, split_merged=True)
    for seed in (3, 11, 42):
        pair = [(40.0, 38.5, 80.0, 1.3), (40.0, 42.5, 70.0, 1.3)]
        img = _render_spots((80, 80), pair, seed=seed)
        for key in ("wavelet", "hmax", "dog", "radial_symmetry"):
            foci = REGISTRY[key]().detect(img, None, opts)
            split = split_merged_foci(img, foci, opts)
            near = sorted(
                round(f.x, 1) for f in split
                if abs(f.y - 40.0) < 3.0 and 35.0 < f.x < 46.0
            )
            assert near == [38.5, 42.5], (
                f"{key} seed {seed}: pair region resolved to {near}"
            )
