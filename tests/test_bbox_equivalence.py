"""Bounding-box per-cell code gives exactly what the whole-mask scans gave.

Classify and Features used to compare the whole labelled mask against one
label per cell (``labeled_mask == lbl``, ``np.where``, ``np.sum``), which is
O(cells x pixels) on dense fields. They now take every cell's area and
bounding box from one pass (``mask_index.label_index``) and read only that
box. The ``_ref_*`` functions below are the pre-change implementations,
copied verbatim from commit ae994f0; outputs must match them bit for bit.

The synthetic fields cover what dense real fields contain: cells touching
each other and every border, cells inside the edge margin, tiny debris,
oversized clumps, a label split into two pieces, labels hidden entirely by
overlap, and gaps in the label numbering.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import binary_dilation
from skimage.transform import resize

from mycomorph.core import cell_quality_classifier as cqc
from mycomorph.core.extract import api as extract_api
from mycomorph.core.extract import crops as crops_mod
from mycomorph.core.foci._types import DetectorOpts, Focus
from mycomorph.core.foci.detectors._common import label_at, refine_gaussian_peak
from mycomorph.core.foci.detectors.bacteroidal import BacteroidalDetector
from mycomorph.core.mask_index import label_bbox, label_index


# ── Pre-change reference implementations (verbatim, commit ae994f0) ──────────

def _ref_extract_cell_crop(image_channels, labeled_mask, cell_label, pad=10,
                           phase_channel=None):
    CROP_SIZE = cqc.CROP_SIZE
    h, w = labeled_mask.shape
    cell_pixels = labeled_mask == cell_label

    if not cell_pixels.any():
        return None, None

    ys, xs = np.where(cell_pixels)
    y_min = max(ys.min() - pad, 0)
    y_max = min(ys.max() + pad + 1, h)
    x_min = max(xs.min() - pad, 0)
    x_max = min(xs.max() + pad + 1, w)

    # Crop imaging channel(s)
    if phase_channel is not None:
        # Phase-only mode: single imaging channel → generalisable model
        channel_crops = image_channels[phase_channel:phase_channel + 1,
                                       y_min:y_max, x_min:x_max].astype(np.float32)
    else:
        # All-channels mode
        channel_crops = image_channels[:, y_min:y_max, x_min:x_max].astype(np.float32)

    # Crop mask (binary for this cell only)
    mask_crop = cell_pixels[y_min:y_max, x_min:x_max].astype(np.float32)
    mask_crop = mask_crop[np.newaxis, :, :]  # (1, h_crop, w_crop)

    combined = np.concatenate([channel_crops, mask_crop], axis=0)  # (N_ch+1, h, w)

    # Pad to square
    _, ch, cw = combined.shape
    side = max(ch, cw)
    padded = np.zeros((combined.shape[0], side, side), dtype=np.float32)
    y_off = (side - ch) // 2
    x_off = (side - cw) // 2
    padded[:, y_off:y_off + ch, x_off:x_off + cw] = combined

    # Resize to CROP_SIZE × CROP_SIZE
    resized = np.zeros((padded.shape[0], CROP_SIZE, CROP_SIZE), dtype=np.float32)
    for c in range(padded.shape[0]):
        resized[c] = resize(padded[c], (CROP_SIZE, CROP_SIZE),
                            order=1, preserve_range=True, anti_aliasing=True)

    # Normalise each channel independently to [0, 1]
    for c in range(resized.shape[0] - 1):  # Skip mask channel
        ch_data = resized[c]
        cmin, cmax = ch_data.min(), ch_data.max()
        if cmax > cmin:
            resized[c] = (ch_data - cmin) / (cmax - cmin)
        else:
            resized[c] = 0.0

    # Mask channel: already 0/1, just re-threshold after resize
    resized[-1] = (resized[-1] > 0.5).astype(np.float32)

    return resized, (y_min, y_max, x_min, x_max)


def _ref_extract_all_crops(image_channels, labeled_mask, pad=10, phase_channel=None):
    cell_labels = np.unique(labeled_mask)
    cell_labels = cell_labels[cell_labels > 0]

    crops = {}
    bboxes = {}

    for label in cell_labels:
        crop, bbox = _ref_extract_cell_crop(image_channels, labeled_mask, label, pad,
                                            phase_channel=phase_channel)
        if crop is not None:
            crops[label] = crop
            bboxes[label] = bbox

    return crops, bboxes


def _ref_detect_edge_cells(labeled_mask, margin=cqc.EDGE_MARGIN):
    h, w = labeled_mask.shape
    edge_labels = set()

    for label in np.unique(labeled_mask):
        if label == 0:
            continue
        ys, xs = np.where(labeled_mask == label)
        if (ys.min() < margin or ys.max() >= h - margin or
                xs.min() < margin or xs.max() >= w - margin):
            edge_labels.add(label)

    return edge_labels


def _ref_detect_debris_by_area(labeled_mask, pixels_per_um=13.8767,
                               min_area_um2=cqc.MIN_AREA_UM2):
    px_area_per_um2 = pixels_per_um ** 2
    min_area_px = min_area_um2 * px_area_per_um2
    debris_labels = set()

    for label in np.unique(labeled_mask):
        if label == 0:
            continue
        area = np.sum(labeled_mask == label)
        if area < min_area_px:
            debris_labels.add(label)

    return debris_labels


def _ref_detect_large_clumps(labeled_mask, pixels_per_um=13.8767,
                             max_area_um2=cqc.MAX_AREA_UM2):
    px_area_per_um2 = pixels_per_um ** 2
    max_area_px = max_area_um2 * px_area_per_um2
    large_labels = set()

    for label in np.unique(labeled_mask):
        if label == 0:
            continue
        area = np.sum(labeled_mask == label)
        if area > max_area_px:
            large_labels.add(label)

    return large_labels


def _ref_classify_and_filter_mask(
    labeled_mask,
    image_channels,
    phase_channel,
    model_path=None,
    pixels_per_um=13.8767,
    keep_classes=("good",),
    confidence_threshold=0.5,
    use_rules=True,
    device=None,
    verbose=True,
    decision_threshold=None,
):
    CLASS_NAMES = cqc.CLASS_NAMES
    CROP_SIZE = cqc.CROP_SIZE
    _GOOD_INDEX, _BAD_INDEX = cqc._GOOD_INDEX, cqc._BAD_INDEX

    if decision_threshold is not None:
        t = cqc._valid_decision_threshold(decision_threshold)
        if t is None:
            raise ValueError(
                f"decision_threshold must be a P(bad) cutoff strictly between "
                f"0 and 1, got {decision_threshold!r}"
            )
        decision_threshold = t

    filtered = labeled_mask.copy()
    all_labels = set(np.unique(labeled_mask)) - {0}
    n_total = len(all_labels)

    report = {
        "total_cells": n_total,
        "removed_edge": 0,
        "removed_debris": 0,
        "removed_cnn": 0,
        "kept": 0,
        "details": {},  # label -> (class_name, confidence)
        "decision_threshold": decision_threshold,
    }

    if n_total == 0:
        return filtered, report

    # ── Step 1: Rule-based filtering ──────────────────────────────────────
    removed = set()

    if use_rules:
        edge_cells = _ref_detect_edge_cells(labeled_mask, margin=cqc.EDGE_MARGIN)
        for lbl in edge_cells:
            filtered[filtered == lbl] = 0
            report["details"][int(lbl)] = ("edge_cell", 1.0)
        removed |= edge_cells
        report["removed_edge"] = len(edge_cells)

        debris_cells = _ref_detect_debris_by_area(
            labeled_mask, pixels_per_um=pixels_per_um, min_area_um2=cqc.MIN_AREA_UM2
        )
        # Don't double-count cells already removed as edge
        debris_only = debris_cells - edge_cells
        for lbl in debris_only:
            filtered[filtered == lbl] = 0
            report["details"][int(lbl)] = ("debris", 1.0)
        removed |= debris_only
        report["removed_debris"] = len(debris_only)

    remaining = all_labels - removed

    # ── Step 2: CNN classification ────────────────────────────────────────
    if model_path is not None and remaining:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # 3 channels: phase + mask + area (area is derived from mask)
        in_channels = 3
        clf, clf_lock = cqc._cached_classifier(model_path, in_channels, device=device)

        # Extract crops for remaining cells (phase + mask + true area)
        labels_list = sorted(remaining)
        crops_list = []
        for lbl in labels_list:
            crop, _ = _ref_extract_cell_crop(image_channels, labeled_mask, lbl,
                                             phase_channel=phase_channel)
            if crop is not None:
                # True pixel area from original mask (shape-independent)
                true_area = int(np.sum(labeled_mask == lbl))
                crop = cqc._append_area_channel_single(crop, true_area_px=true_area)
                crops_list.append(crop)
            else:
                crops_list.append(
                    np.zeros((in_channels, CROP_SIZE, CROP_SIZE), dtype=np.float32)
                )

        crops_array = np.stack(crops_list, axis=0)
        with clf_lock:
            class_indices, probabilities = clf.predict_batch(crops_array)

        keep_indices = {CLASS_NAMES.index(c) for c in keep_classes if c in CLASS_NAMES}
        cnn_removed = 0

        for lbl, cls_idx, probs in zip(labels_list, class_indices, probabilities):
            if decision_threshold is not None:
                # Tuned rule: the P(bad) cutoff alone decides the class.
                cls_idx = (_GOOD_INDEX if float(probs[_BAD_INDEX]) < decision_threshold
                           else _BAD_INDEX)
                reject = cls_idx not in keep_indices
            else:
                # Remove if not in keep_classes OR below confidence threshold
                reject = (cls_idx not in keep_indices
                          or float(probs[cls_idx]) < confidence_threshold)
            cls_name = CLASS_NAMES[cls_idx]
            confidence = float(probs[cls_idx])
            report["details"][int(lbl)] = (cls_name, confidence)

            if reject:
                filtered[filtered == lbl] = 0
                cnn_removed += 1

        report["removed_cnn"] = cnn_removed

    report["kept"] = len(set(np.unique(filtered)) - {0})

    if verbose:
        print(f"    Cell QC: {n_total} total → {report['kept']} kept "
              f"(edge={report['removed_edge']}, debris={report['removed_debris']}, "
              f"cnn={report['removed_cnn']})")

    return filtered, report


def _ref_features_crop(image_channels, labeled_mask, cell_label, selected_channels, opts):
    h, w = labeled_mask.shape
    cell_pixels = labeled_mask == cell_label
    if not cell_pixels.any():
        return None, 0

    ys, xs = np.where(cell_pixels)
    pad = opts.pad
    y_min = max(int(ys.min()) - pad, 0)
    y_max = min(int(ys.max()) + pad + 1, h)
    x_min = max(int(xs.min()) - pad, 0)
    x_max = min(int(xs.max()) + pad + 1, w)

    mask_crop = cell_pixels[y_min:y_max, x_min:x_max]

    if opts.mask_background:
        dilated = binary_dilation(mask_crop, iterations=opts.dilate_px)
    else:
        dilated = np.ones_like(mask_crop)

    channel_planes: list[np.ndarray] = []
    for ch in selected_channels:
        plane = image_channels[ch, y_min:y_max, x_min:x_max].astype(np.float32)
        if opts.mask_background:
            bg = float(np.median(plane[~dilated])) if (~dilated).any() else 0.0
            plane = np.where(dilated, plane, bg).astype(np.float32)
        channel_planes.append(plane)

    if opts.include_mask_channel:
        channel_planes.append(mask_crop.astype(np.float32))

    combined = np.stack(channel_planes, axis=0)

    # Pad to square then resize to the requested side.
    _, ch_h, ch_w = combined.shape
    side = max(ch_h, ch_w)
    padded = np.zeros((combined.shape[0], side, side), dtype=np.float32)
    y_off = (side - ch_h) // 2
    x_off = (side - ch_w) // 2
    padded[:, y_off:y_off + ch_h, x_off:x_off + ch_w] = combined

    n_out = padded.shape[0]
    resized = np.zeros((n_out, opts.crop_size, opts.crop_size), dtype=np.float32)
    for c in range(n_out):
        resized[c] = resize(
            padded[c],
            (opts.crop_size, opts.crop_size),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        )

    if opts.normalise_per_crop:
        # Per-image-channel min-max into [0, 1]; the mask channel is rebound.
        n_image = len(selected_channels)
        for c in range(n_image):
            cmin, cmax = resized[c].min(), resized[c].max()
            if cmax > cmin:
                resized[c] = (resized[c] - cmin) / (cmax - cmin)
            else:
                resized[c] = 0.0
        if opts.include_mask_channel:
            resized[-1] = (resized[-1] > 0.5).astype(np.float32)
    elif opts.include_mask_channel:
        resized[-1] = (resized[-1] > 0.5).astype(np.float32)

    return resized, int(cell_pixels.sum())


def _ref_bacteroidal_detect(image, labeled_mask=None, opts=None):
    from skimage.feature import peak_local_max

    opts = opts or DetectorOpts()
    if labeled_mask is None:
        # Without a mask we have no bacteroidal background; degrade to
        # a single global median.
        labeled_mask = np.ones_like(image, dtype=np.int32)

    out: list[Focus] = []
    n_cells = int(labeled_mask.max())
    for cell_id in range(1, n_cells + 1):
        cell_pixels = labeled_mask == cell_id
        if not cell_pixels.any():
            continue
        cell_values = image[cell_pixels].astype(np.float64)
        cell_median = float(np.median(cell_values))
        cell_std = float(cell_values.std())
        if cell_std < 1e-9:
            continue

        # Local bbox crop, background-subtracted in-cell, zeroed elsewhere.
        ys, xs = np.where(cell_pixels)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        mask_crop = cell_pixels[y0:y1, x0:x1]
        sub = image[y0:y1, x0:x1].astype(np.float64) - cell_median
        sub = np.where(mask_crop, sub, 0.0)

        # Detect peaks above threshold (in cell-std units).
        min_amp = opts.snr_min * cell_std
        peaks = peak_local_max(
            sub,
            min_distance=max(1, int(round(opts.min_sigma * 2))),
            threshold_abs=min_amp,
            exclude_border=False,
        )
        for peak in peaks:
            y_peak = float(peak[0] + y0)
            x_peak = float(peak[1] + x0)
            if opts.refine:
                y_sub, x_sub, sigma_fit, amp = refine_gaussian_peak(
                    image, y_peak, x_peak,
                    half_window=opts.refine_window,
                    init_sigma=opts.min_sigma * 1.2,
                )
            else:
                y_sub, x_sub = y_peak, x_peak
                sigma_fit = opts.min_sigma * 1.2
                amp = float(sub[int(peak[0]), int(peak[1])])
            snr = (amp - cell_median) / cell_std if cell_std > 0 else 0.0
            if snr < opts.snr_min:
                continue
            out.append(Focus(
                y=y_sub,
                x=x_sub,
                sigma=sigma_fit,
                intensity=amp,
                snr=snr,
                cell_label=label_at(labeled_mask, y_sub, x_sub),
            ))
    return out


# ── Synthetic fields ──────────────────────────────────────────────────────────

SHAPE = (150, 190)
PIXELS_PER_UM = 4.0   # min debris 0.3 µm² = 4.8 px, max cell 25 µm² = 400 px


def _synthetic_mask(seed, dtype=np.int32, shape=SHAPE):
    rng = np.random.default_rng(seed)
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    mask = np.zeros(shape, dtype=np.int64)
    label = 0
    # Random rods, some centred off the image so they are cut by the border.
    # Later cells overwrite earlier ones: touching cells, cells split in
    # two or hidden completely (which leaves a gap in the labels).
    for _ in range(70):
        label += int(rng.integers(1, 4))          # gaps in the numbering
        cy, cx = rng.uniform(-4, h + 4), rng.uniform(-4, w + 4)
        a, b = rng.uniform(2.0, 13.0), rng.uniform(0.6, 4.0)
        th = rng.uniform(0, np.pi)
        dy, dx = yy - cy, xx - cx
        u = (dy * np.cos(th) + dx * np.sin(th)) / a
        v = (-dy * np.sin(th) + dx * np.cos(th)) / b
        mask[u * u + v * v <= 1.0] = label
    # Placed cells for the edge cases. Each gets a fresh label.
    def put(rs, cs):
        nonlocal label
        label += 1
        mask[rs, cs] = label
    put(slice(0, 3), slice(0, 4))                 # top-left corner
    put(slice(h - 2, h), slice(w - 5, w))         # bottom-right corner
    put(slice(1, 5), slice(60, 66))               # inside the 2 px margin
    put(slice(2, 6), slice(80, 86))               # just clear of the margin
    put(slice(70, 74), slice(w - 3, w - 1))       # inside margin on the right
    put(slice(75, 76), slice(95, 96))             # 1 px debris
    put(slice(90, 92), slice(40, 42))             # 4 px debris (< 4.8)
    put(slice(100, 125), slice(140, 165))         # 625 px clump
    label += 1                                    # one label in two pieces
    mask[30:34, 20:26] = label
    mask[50:53, 30:36] = label
    assert label < 256, "keep labels in uint8 range"
    return mask.astype(dtype)


def _synthetic_image(seed, n_channels=2, shape=SHAPE):
    rng = np.random.default_rng(seed + 1000)
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    planes = [
        (1000 + 300 * np.sin(yy / (7 + c)) * np.cos(xx / (11 + c))
         + rng.normal(0, 40, shape)).clip(0, 65535)
        for c in range(n_channels)
    ]
    return np.stack(planes).astype(np.uint16)


INT_DTYPES = [np.uint8, np.uint16, np.int32, np.int64, np.uint32]


def _same_scalars(a, b):
    """Equal, and with the same Python/NumPy types (sets or sequences)."""
    a_sorted, b_sorted = sorted(a), sorted(b)
    assert a_sorted == b_sorted
    assert [type(x) for x in a_sorted] == [type(x) for x in b_sorted]


# ── mask_index ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", INT_DTYPES + [np.float64])
def test_label_index_matches_whole_mask_scans(dtype):
    mask = _synthetic_mask(1, dtype)
    labels, areas, slices = label_index(mask)

    ref = np.unique(mask)
    ref = ref[ref != 0]
    assert labels.dtype == ref.dtype
    np.testing.assert_array_equal(labels, ref)
    assert len(labels) < mask.max()              # the field really has gaps
    for lbl, area, sl in zip(labels, areas, slices):
        cell = mask == lbl
        assert area == cell.sum()
        ys, xs = np.where(cell)
        assert sl == (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        assert sl == label_bbox(mask, lbl)


def test_label_index_odd_masks():
    # Negative and very sparse labels take the slow path but agree.
    mask = _synthetic_mask(2, np.int64)
    mask[mask == mask.max()] = -7
    mask[mask == np.unique(mask)[-1]] = 10**12
    labels, areas, _ = label_index(mask)
    ref, counts = np.unique(mask, return_counts=True)
    np.testing.assert_array_equal(labels, ref[ref != 0])
    np.testing.assert_array_equal(areas, counts[ref != 0])

    empty = np.zeros((5, 5), np.uint16)
    labels, areas, slices = label_index(empty)
    assert labels.size == 0 and areas.size == 0 and slices == []
    assert label_bbox(empty, 1) is None


# ── Rule-based filters ────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("margin", [0, 2, 5])
def test_detect_edge_cells_matches(dtype, margin):
    mask = _synthetic_mask(3, dtype)
    new = cqc.detect_edge_cells(mask, margin=margin)
    _same_scalars(new, _ref_detect_edge_cells(mask, margin=margin))
    if margin == 2:
        assert len(new) >= 5


@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("ppu", [PIXELS_PER_UM, 13.8767])
def test_area_filters_match(dtype, ppu):
    mask = _synthetic_mask(4, dtype)
    debris = cqc.detect_debris_by_area(mask, pixels_per_um=ppu)
    clumps = cqc.detect_large_clumps(mask, pixels_per_um=ppu)
    _same_scalars(debris, _ref_detect_debris_by_area(mask, pixels_per_um=ppu))
    _same_scalars(clumps, _ref_detect_large_clumps(mask, pixels_per_um=ppu))
    if ppu == PIXELS_PER_UM:
        assert debris and clumps


# ── Classifier crops ──────────────────────────────────────────────────────────

def _assert_crop_equal(new, ref):
    new_crop, new_bbox = new
    ref_crop, ref_bbox = ref
    if ref_crop is None:
        assert new_crop is None and new_bbox is None
        return
    assert new_crop.dtype == ref_crop.dtype
    np.testing.assert_array_equal(new_crop, ref_crop)
    assert new_bbox == ref_bbox
    assert [type(v) for v in new_bbox] == [type(v) for v in ref_bbox]


@pytest.mark.parametrize("dtype", [np.uint16, np.int32])
@pytest.mark.parametrize("pad", [0, 10, 30])
@pytest.mark.parametrize("phase_channel", [None, 0, 1])
def test_extract_cell_crop_matches(dtype, pad, phase_channel):
    mask = _synthetic_mask(5, dtype)
    image = _synthetic_image(5)
    labels, _, slices = label_index(mask)
    for lbl, sl in zip(labels, slices):
        ref = _ref_extract_cell_crop(image, mask, lbl, pad, phase_channel=phase_channel)
        _assert_crop_equal(
            cqc.extract_cell_crop(image, mask, lbl, pad, phase_channel=phase_channel,
                                  cell_slice=sl),
            ref,
        )
        _assert_crop_equal(
            cqc.extract_cell_crop(image, mask, lbl, pad, phase_channel=phase_channel),
            ref,
        )
    # Absent labels: gaps, past the last label, and the background.
    gaps = sorted(set(range(1, int(mask.max()) + 3)) - set(labels.tolist()))
    for lbl in gaps[:5] + [0]:
        _assert_crop_equal(
            cqc.extract_cell_crop(image, mask, lbl, pad, phase_channel=phase_channel),
            _ref_extract_cell_crop(image, mask, lbl, pad, phase_channel=phase_channel),
        )


def test_extract_all_crops_matches():
    mask = _synthetic_mask(6, np.int32)
    image = _synthetic_image(6)
    crops, bboxes = cqc.extract_all_crops(image, mask, phase_channel=0)
    ref_crops, ref_bboxes = _ref_extract_all_crops(image, mask, phase_channel=0)
    _same_scalars(crops.keys(), ref_crops.keys())
    for lbl in ref_crops:
        _assert_crop_equal((crops[lbl], bboxes[lbl]), (ref_crops[lbl], ref_bboxes[lbl]))


# ── classify_and_filter_mask ──────────────────────────────────────────────────

class _HashClassifier:
    """Stand-in CNN: P(bad) is a hash of the crop's bytes.

    Any difference in a crop changes its prediction, and every batch it was
    given is kept for comparison.
    """

    def __init__(self):
        self.batches = []

    def predict_batch(self, crops, batch_size=64):
        self.batches.append(crops.copy())
        p_bad = np.array(
            [int.from_bytes(hashlib.sha256(c.tobytes()).digest()[:4], "little") / 2**32
             for c in crops],
            dtype=np.float32,
        )
        probs = np.stack([1.0 - p_bad, p_bad], axis=1).astype(np.float32)
        return probs.argmax(axis=1), probs


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.int64])
@pytest.mark.parametrize("kwargs", [
    dict(use_rules=True),
    dict(use_rules=False),
    dict(use_rules=True, confidence_threshold=0.75),
    dict(use_rules=True, decision_threshold=0.4),
    dict(use_rules=True, pixels_per_um=13.8767),
], ids=["rules", "no-rules", "conf-0.75", "tuned-0.4", "real-scale"])
@pytest.mark.parametrize("with_model", [False, True], ids=["rules-only", "cnn"])
def test_classify_and_filter_mask_matches(monkeypatch, tmp_path, dtype, kwargs, with_model):
    mask = _synthetic_mask(7, dtype)
    image = _synthetic_image(7)
    model_path = None
    classifiers = []
    if with_model:
        model_path = tmp_path / "fake.pth"
        model_path.write_bytes(b"")

        def fake_cached(path, in_channels, device=None):
            import threading
            classifiers.append(_HashClassifier())
            return classifiers[-1], threading.Lock()

        monkeypatch.setattr(cqc, "_cached_classifier", fake_cached)

    kwargs = {"pixels_per_um": PIXELS_PER_UM, **kwargs}
    filtered, report = cqc.classify_and_filter_mask(
        mask, image, phase_channel=0, model_path=model_path, verbose=False, **kwargs)
    ref_filtered, ref_report = _ref_classify_and_filter_mask(
        mask, image, phase_channel=0, model_path=model_path, verbose=False, **kwargs)

    assert filtered.dtype == ref_filtered.dtype
    np.testing.assert_array_equal(filtered, ref_filtered)
    assert report == ref_report
    assert ref_report["kept"] > 0
    if kwargs.get("use_rules"):
        assert ref_report["removed_edge"] > 0
    if with_model:
        new_batch, ref_batch = (c.batches[0] for c in classifiers)
        np.testing.assert_array_equal(new_batch, ref_batch)
        assert ref_report["removed_cnn"] > 0


def test_classify_and_filter_mask_empty_and_all_removed():
    image = _synthetic_image(8)
    empty = np.zeros(SHAPE, np.int32)
    assert (cqc.classify_and_filter_mask(empty, image, 0, verbose=False)[1]
            == _ref_classify_and_filter_mask(empty, image, 0, verbose=False)[1])
    # Only edge cells and debris: everything is removed by the rules.
    mask = np.zeros(SHAPE, np.int32)
    mask[0:4, 10:20] = 3
    mask[60:61, 60:62] = 9
    new_f, new_r = cqc.classify_and_filter_mask(mask, image, 0, verbose=False)
    ref_f, ref_r = _ref_classify_and_filter_mask(mask, image, 0, verbose=False)
    np.testing.assert_array_equal(new_f, ref_f)
    assert new_r == ref_r and new_r["kept"] == 0


# ── Features crops ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("opts", [
    crops_mod.CropOpts(),
    crops_mod.CropOpts(crop_size=96, pad=10),             # the RibF run's options
    crops_mod.CropOpts(crop_size=48, pad=0, dilate_px=1),
    crops_mod.CropOpts(crop_size=64, pad=30),
    crops_mod.CropOpts(mask_background=False),
    crops_mod.CropOpts(include_mask_channel=False),
    crops_mod.CropOpts(normalise_per_crop=False),
])
@pytest.mark.parametrize("selected", [[0, 1], [1]])
def test_features_crop_matches(opts, selected):
    mask = _synthetic_mask(9, np.int32)
    image = _synthetic_image(9)
    labels, _, slices = label_index(mask)
    for lbl, sl in zip(labels, slices):
        ref_crop, ref_area = _ref_features_crop(image, mask, int(lbl), selected, opts)
        for kw in ({"cell_slice": sl}, {}):
            crop, area = crops_mod.extract_cell_crop(image, mask, int(lbl), selected, opts, **kw)
            assert crop.dtype == ref_crop.dtype
            np.testing.assert_array_equal(crop, ref_crop)
            assert area == ref_area and type(area) is type(ref_area)
    assert crops_mod.extract_cell_crop(image, mask, 10_000, selected, opts) == (None, 0)


def test_process_fov_crops_match():
    mask = _synthetic_mask(10, np.int32)
    image = _synthetic_image(10)
    fov = np.concatenate([image, mask[None].astype(np.uint16)])
    opts = crops_mod.CropOpts(crop_size=48)
    df, crops_arr, meta = extract_api._process_fov(
        fov, 0,
        image_channel_names=["Phase", "Fluor"],
        pixels_per_um=PIXELS_PER_UM,
        run_id="run", well="A01", source_czi="", plate_acq_dt=None, fov_acq_time=None,
        intensity_channels=[0, 1],
        crop_opts=opts, crop_channels=[0, 1],
        condition_fields=crops_mod.derive_condition_fields("cond__rep__drug"),
        tiff_filename="A01.tif",
        midline_features_flag=False,
    )
    labeled = extract_api._label_mask(fov[-1])
    assert len(meta) == len(df) == crops_arr.shape[0]
    for crop, m in zip(crops_arr, meta):
        ref_crop, ref_area = _ref_features_crop(image, labeled, m["cell_label"], [0, 1], opts)
        np.testing.assert_array_equal(crop, ref_crop)
        assert m["area_px"] == ref_area


# ── Labelling tool ────────────────────────────────────────────────────────────

def test_label_cells_extract_all_cells_matches(tmp_path):
    import tifffile

    from mycomorph.core.hyperstack import get_labeled_mask_from_fov
    from mycomorph.core.label_cells import extract_all_cells

    fovs = []
    for seed in (12, 13):
        binary = (_synthetic_mask(seed, np.int32) > 0).astype(np.uint16) * 255
        fovs.append(np.concatenate([_synthetic_image(seed), binary[None]]))
    data = np.stack(fovs)
    path = tmp_path / "A01.tif"
    tifffile.imwrite(str(path), data, imagej=True, metadata={
        "axes": "ZCYX", "channels": 3, "slices": 2, "hyperstack": True, "mode": "composite",
    })

    cells = extract_all_cells([path], phase_channel=0, mask_channel=2)

    expected = []
    for fov_idx, fov in enumerate(data):
        labeled, n = get_labeled_mask_from_fov(fov, 2)
        img = np.delete(fov, 2, axis=0)
        for lbl in range(1, n + 1):
            crop, _ = _ref_extract_cell_crop(img, labeled, lbl)
            if crop is not None:
                expected.append((f"A01_fov{fov_idx:03d}_cell{lbl:04d}", crop,
                                 int(np.sum(labeled == lbl))))
    assert [c["cell_id"] for c in cells] == [e[0] for e in expected]
    for cell, (_, crop, area) in zip(cells, expected):
        np.testing.assert_array_equal(cell["crop"], crop)
        assert cell["area_px"] == area


# ── Bacteroidal foci detector ─────────────────────────────────────────────────

@pytest.mark.parametrize("refine", [True, False])
@pytest.mark.parametrize("with_mask", [True, False])
def test_bacteroidal_matches(refine, with_mask):
    mask = _synthetic_mask(11, np.int32)
    rng = np.random.default_rng(11)
    image = rng.normal(200.0, 10.0, SHAPE)
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]]
    for _ in range(40):
        cy, cx = rng.uniform(0, SHAPE[0]), rng.uniform(0, SHAPE[1])
        image += rng.uniform(80, 400) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 3.0)
    image = image.astype(np.float32)
    labeled = mask if with_mask else None
    opts = DetectorOpts(refine=refine)
    new = BacteroidalDetector().detect(image, labeled, opts)
    ref = _ref_bacteroidal_detect(image, labeled, opts)
    assert new == ref
    assert len(ref) > 0
