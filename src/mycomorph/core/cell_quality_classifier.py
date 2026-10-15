#!/usr/bin/env python3
"""
Cell Quality Classifier
=======================
CNN-based classification of segmented cells into quality categories.
Used as a post-segmentation filter to remove bad cells from masks.

Categories:
    0 = good         — well-segmented, in-focus single cell
    1 = out_of_focus  — blurry cell (low edge contrast in phase)
    2 = clumped       — multiple overlapping cells segmented as one
    3 = edge_cell     — cell touching the FOV boundary (incomplete)
    4 = debris        — non-cell object that passed min_size filter

Usage in pipeline:
    from cell_quality_classifier import classify_and_filter_mask
    filtered_mask = classify_and_filter_mask(
        labeled_mask, channels, phase_channel=2, model_path="model.pth"
    )

Requirements:
    pip install torch torchvision numpy scikit-image scipy
"""

import threading

import numpy as np
from pathlib import Path

# The block below is frozen: every saved .pth (the bundled presets and every
# model trained with ModelPrep, mycomorph.modelprep) depends on it, and
# tests/test_classifier_model_def.py pins its SHA-256. ModelPrep now imports
# this block instead of keeping a copy. The block's own header still names that
# former copy; it is kept verbatim so the hash stays the one the archived
# ModelPrep, and the Fig. S2 models trained with it, used.

# ── BEGIN SHARED MODEL DEFINITION ─────────────────────────────────────────────
# Keep byte-identical with:
#   MycoMorph:  src/mycomorph/core/cell_quality_classifier.py
#   ModelPrep:  src/modelprep/core/cell_quality_classifier.py
#
# Anything inside this block is reproduced verbatim across both repos. A drift-
# guard test in each repo hashes the block and fails on mismatch. If you need
# to change the architecture, the area-channel logic, or the CellQualityClassifier
# loader, update BOTH copies in the same commit and refresh the golden hash.
# ─────────────────────────────────────────────────────────────────────────────

# Binary classification: good (0) vs bad (1)
CLASS_NAMES = ["good", "bad"]
NUM_CLASSES = len(CLASS_NAMES)
CROP_SIZE = 64  # Pixels. Resized crop fed to CNN.

# Legacy 5-class names, used to remap old labeled datasets to binary
LEGACY_CLASS_NAMES = ["good", "out_of_focus", "clumped", "edge_cell", "debris"]

# ── Area channel helpers ──────────────────────────────────────────────────────
#
# The area channel encodes absolute cell size as a log-normalised value in
# [0, 1], computed from the true pixel area in the original (un-resized) mask.
# This is shape-independent: a long rod and a round coccus with the same pixel
# count get exactly the same area value.
#
# Normalisation range:  log(MIN_AREA_PX) … log(MAX_AREA_PX)
#   At 13.87 px/µm → 192.5 px²/µm²
#     MIN_AREA_PX = 10   (~0.05 µm² — tiny debris)
#     MAX_AREA_PX = 10000 (~52 µm² — large clump)

_AREA_LOG_MIN = np.log(10.0)
_AREA_LOG_MAX = np.log(10000.0)
_AREA_LOG_RANGE = _AREA_LOG_MAX - _AREA_LOG_MIN


def _normalise_area(area_px):
    """Map raw pixel area → [0, 1] via log scale."""
    return float(np.clip(
        (np.log(max(area_px, 1)) - _AREA_LOG_MIN) / _AREA_LOG_RANGE,
        0.0, 1.0,
    ))


def _append_area_channel_single(crop, true_area_px=None):
    """
    Append a normalised area channel to a single crop.

    Args:
        crop:          (C, H, W) float32 — last channel is the binary mask
        true_area_px:  pixel count of the cell in the original (un-resized) mask.
                       If None, approximates from the resized mask (less accurate).

    Returns:
        (C+1, H, W) float32 with the area channel appended
    """
    if true_area_px is not None:
        area_val = _normalise_area(true_area_px)
    else:
        # Fallback: use resized mask pixel count (shape-dependent approximation)
        area_val = _normalise_area(float(crop[-1].sum()))

    mask = crop[-1:]                            # (1, H, W) — just for shape
    area_ch = np.full_like(mask, area_val)      # (1, H, W)
    return np.concatenate([crop, area_ch], axis=0)


def _append_area_channel_batch(crops, true_areas_px=None):
    """
    Append a normalised area channel to a batch of crops.

    Args:
        crops:          (N, C, H, W) float32 — last channel is the binary mask
        true_areas_px:  (N,) array of pixel counts from original masks.
                        If None, approximates from resized masks.

    Returns:
        (N, C+1, H, W) float32
    """
    n = crops.shape[0]
    mask = crops[:, -1:, :, :]                  # (N, 1, H, W)

    if true_areas_px is not None:
        vals = np.array([_normalise_area(a) for a in true_areas_px],
                        dtype=np.float32).reshape(n, 1, 1, 1)
    else:
        # Fallback: resized mask pixel count
        raw = mask.sum(axis=(2, 3), keepdims=True)  # (N, 1, 1, 1)
        vals = np.clip(
            (np.log(np.maximum(raw, 1)) - _AREA_LOG_MIN) / _AREA_LOG_RANGE,
            0.0, 1.0,
        ).astype(np.float32)

    area_ch = np.broadcast_to(vals, mask.shape).copy()  # (N, 1, H, W)
    return np.concatenate([crops, area_ch], axis=1)


# ── CNN model ─────────────────────────────────────────────────────────────────

def _build_model(in_channels, num_classes=NUM_CLASSES):
    """
    Build a lightweight CNN for cell quality classification.

    Architecture:
        4 conv blocks (Conv2d → BatchNorm → ReLU → MaxPool2×2)
        Global Average Pooling → Dropout → FC → num_classes

    With CROP_SIZE=64, after 4 max-pools: 64→32→16→8→4, then GAP → 256-d.
    Total ~400K parameters — trainable on a few hundred labelled cells.
    """
    import torch
    import torch.nn as nn

    class ConvBlock(nn.Module):
        def __init__(self, c_in, c_out, kernel=3, pad=1):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel, padding=pad, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        def forward(self, x):
            return self.block(x)

    class CellQualityCNN(nn.Module):
        def __init__(self, c_in, n_cls):
            super().__init__()
            self.features = nn.Sequential(
                ConvBlock(c_in, 32),
                ConvBlock(32, 64),
                ConvBlock(64, 128),
                ConvBlock(128, 256),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(256, n_cls),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    return CellQualityCNN(in_channels, num_classes)


# ── Classifier wrapper ────────────────────────────────────────────────────────

class CellQualityClassifier:
    """
    Wrapper that loads a trained CNN and classifies cell crops.

    Usage:
        clf = CellQualityClassifier("model.pth", in_channels=3)
        predictions = clf.predict_batch(crops_array)
    """

    def __init__(self, model_path, in_channels, device=None):
        import torch

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model = _build_model(in_channels, NUM_CLASSES)

        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict_single(self, crop):
        """
        Classify a single cell crop.

        Args:
            crop: (C, CROP_SIZE, CROP_SIZE) float32 numpy array

        Returns:
            class_idx: int
            class_name: str
            probabilities: (NUM_CLASSES,) array
        """
        import torch

        x = torch.from_numpy(crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        class_idx = int(probs.argmax())
        return class_idx, CLASS_NAMES[class_idx], probs

    def predict_batch(self, crops, batch_size=64):
        """
        Classify a batch of cell crops.

        Args:
            crops: (N, C, CROP_SIZE, CROP_SIZE) float32 numpy array
            batch_size: processing batch size

        Returns:
            class_indices: (N,) int array
            probabilities: (N, NUM_CLASSES) array
        """
        import torch

        all_probs = []

        for i in range(0, len(crops), batch_size):
            batch = torch.from_numpy(crops[i:i + batch_size]).to(self.device)
            with torch.no_grad():
                logits = self.model(batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

        all_probs = np.concatenate(all_probs, axis=0)
        class_indices = all_probs.argmax(axis=1)

        return class_indices, all_probs


# ── END SHARED MODEL DEFINITION ───────────────────────────────────────────────
# Below: MycoMorph-only inference helpers (crop extraction, rule-based
# filters, classifier cache and the mask filter). ModelPrep doesn't have
# these; it trains on pre-extracted HDF5 crops.

# Default thresholds for rule-based pre-filtering
EDGE_MARGIN = 2        # Pixels from border to count as edge cell
MIN_AREA_UM2 = 0.3     # Minimum cell area in µm² (debris filter)
MAX_AREA_UM2 = 25.0    # Maximum single-cell area in µm² (clump indicator)


# ── Cell crop extraction ──────────────────────────────────────────────────────

def extract_cell_crop(image_channels, labeled_mask, cell_label, pad=10,
                      phase_channel=None):
    """
    Extract a square crop of a single cell from multi-channel image data.

    The crop is taken from the bounding box of the cell (from the labeled mask)
    plus padding, padded to square, then resized to CROP_SIZE × CROP_SIZE.

    If phase_channel is specified, only that channel is used (+ mask), giving a
    2-channel crop that is independent of the number of fluorescence channels.
    This makes the model generalisable across different acquisition setups.

    Args:
        image_channels: (C, Y, X) numpy array — all imaging channels
        labeled_mask:   (Y, X) integer array — 0=background, N=cell label
        cell_label:     integer — which cell to extract
        pad:            pixels of padding around bounding box
        phase_channel:  if set, extract only this channel (+ mask → 2-ch crop).
                        If None, extract all channels (+ mask → C+1-ch crop).

    Returns:
        crop: (2, CROP_SIZE, CROP_SIZE) or (C+1, CROP_SIZE, CROP_SIZE) float32
              Intensities are individually normalised to [0, 1] per channel.
        bbox: (y_min, y_max, x_min, x_max) of the padded bounding box
    """
    from skimage.transform import resize

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


def extract_all_crops(image_channels, labeled_mask, pad=10, phase_channel=None):
    """
    Extract crops for all cells in a labeled mask.

    Args:
        image_channels: (C, Y, X) array
        labeled_mask:   (Y, X) integer array
        pad:            padding around bounding box
        phase_channel:  if set, extract only this channel (+ mask → 2-ch crop)

    Returns:
        crops:  dict mapping cell_label -> crop array
        bboxes: dict mapping cell_label -> (y_min, y_max, x_min, x_max)
    """
    cell_labels = np.unique(labeled_mask)
    cell_labels = cell_labels[cell_labels > 0]

    crops = {}
    bboxes = {}

    for label in cell_labels:
        crop, bbox = extract_cell_crop(image_channels, labeled_mask, label, pad,
                                       phase_channel=phase_channel)
        if crop is not None:
            crops[label] = crop
            bboxes[label] = bbox

    return crops, bboxes


# ── Rule-based pre-filters ────────────────────────────────────────────────────

def detect_edge_cells(labeled_mask, margin=EDGE_MARGIN):
    """
    Find cells whose bounding box touches the image border.

    Returns:
        set of cell labels that are edge cells
    """
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


def detect_debris_by_area(labeled_mask, pixels_per_um=13.8767, min_area_um2=MIN_AREA_UM2):
    """
    Find cells below a minimum area threshold (likely debris).

    Returns:
        set of cell labels classified as debris
    """
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


def detect_large_clumps(labeled_mask, pixels_per_um=13.8767, max_area_um2=MAX_AREA_UM2):
    """
    Flag cells above a maximum single-cell area as potential clumps.
    These are candidates for CNN review, not automatic rejection.

    Returns:
        set of cell labels that are suspiciously large
    """
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


# Constructing a CellQualityClassifier means a torch.load + load_state_dict
# + device transfer. Both callers pay that repeatedly for the same weights:
# `classify_filter_tiff` calls the filter once per FOV, and the live preview
# re-runs classification on every option change. Cache instances on the model
# file's identity (path + mtime + size), so retraining to the same path
# invalidates the entry without any explicit cache-busting.
_CLASSIFIER_CACHE: dict[tuple, tuple] = {}
_CLASSIFIER_CACHE_LOCK = threading.Lock()


def _cached_classifier(model_path, in_channels, device=None):
    """Return ``(classifier, inference_lock)`` for these weights.

    The lock guards forward passes: cached instances are shared between
    the pipeline's worker thread and the preview's, and a torch module is
    only reliably re-entrant when nothing else is mid-forward on it.
    """
    model_path = Path(model_path)
    try:
        stat = model_path.stat()
        key = (
            str(model_path.resolve()), stat.st_mtime_ns, stat.st_size,
            int(in_channels), str(device),
        )
    except OSError:
        # Can't fingerprint it — build an uncached instance rather than
        # risk handing back weights that no longer match the file.
        return CellQualityClassifier(str(model_path), in_channels, device=device), threading.Lock()

    with _CLASSIFIER_CACHE_LOCK:
        entry = _CLASSIFIER_CACHE.get(key)
        if entry is None:
            entry = (
                CellQualityClassifier(str(model_path), in_channels, device=device),
                threading.Lock(),
            )
            _CLASSIFIER_CACHE[key] = entry
        return entry


# ── Tuned decision threshold (from ModelPrep) ─────────────────────────────────
#
# ModelPrep writes ``decision_threshold`` to the ``training_config.json`` next
# to the ``.pth``. It is the Youden-J cutoff on the validation ROC of
# P(bad) (class index 1), so it is applied as "keep the cell iff P(bad) < t".
# That is a different rule from the default argmax + P(good) confidence
# threshold, and the two numbers are not interchangeable.

_BAD_INDEX = CLASS_NAMES.index("bad")
_GOOD_INDEX = CLASS_NAMES.index("good")


def _valid_decision_threshold(value):
    """Return ``value`` as a float if it is a usable P(bad) cutoff, else None."""
    try:
        t = float(value)
    except (TypeError, ValueError):
        return None
    # roc_curve can put the Youden-J optimum at its +inf sentinel; that (and
    # anything outside (0, 1)) would keep or reject every cell.
    if not np.isfinite(t) or not 0.0 < t < 1.0:
        return None
    return t


def read_decision_threshold(model_path):
    """Return the tuned P(bad) cutoff stored beside ``model_path``, or None.

    Looks for ``decision_threshold`` in ``training_config.json`` in the same
    folder as the ``.pth`` (as ModelPrep writes it). Returns None when the file
    or key is missing, unreadable, or not a value strictly between 0 and 1.
    """
    import json

    if model_path is None:
        return None
    cfg_path = Path(model_path).parent / "training_config.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(cfg, dict):
        return None
    return _valid_decision_threshold(cfg.get("decision_threshold"))


def effective_p_bad_cutoff(confidence_threshold, decision_threshold=None):
    """P(bad) value above which ``classify_and_filter_mask`` rejects a cell.

    Puts both rules on the P(bad) axis that the ROC / PR curves use:

    - tuned: reject iff P(bad) >= ``decision_threshold``;
    - default: keep iff argmax is "good" and P(good) >= ``confidence_threshold``,
      i.e. P(good) >= max(confidence_threshold, 0.5), so the cutoff is
      1 - max(confidence_threshold, 0.5). Slider values below 0.5 therefore
      change nothing: argmax already rejects every cell with P(bad) > 0.5.
    """
    if decision_threshold is not None:
        return float(decision_threshold)
    return 1.0 - max(float(confidence_threshold), 0.5)


# ── Main filtering API ────────────────────────────────────────────────────────

def classify_and_filter_mask(
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
    """
    Classify each cell in a labeled mask and return a filtered version
    containing only cells that pass quality control.

    This function applies two levels of filtering:

    1. Rule-based (fast, no model needed):
       - Edge cells: bounding box touches image border
       - Debris: area below minimum threshold

    2. CNN-based (requires trained model):
       - Classifies remaining cells as good / bad
       - Keeps only cells whose class is in keep_classes

    If no model_path is provided, only rule-based filtering is applied.

    The CNN's class is decided in one of two ways:

    - Default (``decision_threshold=None``): argmax of the softmax, and the
      cell is also rejected if that class's probability is below
      ``confidence_threshold``.
    - Tuned (``decision_threshold=t``): the cell is "good" iff P(bad) < t,
      otherwise "bad". ``confidence_threshold`` is ignored. ``t`` is the
      P(bad) cutoff ModelPrep stores as ``decision_threshold`` (see
      ``read_decision_threshold``).

    Args:
        labeled_mask:   (Y, X) integer array from Cellpose (0=bg, N=cell)
        image_channels: (C, Y, X) array — all imaging channels
        phase_channel:  index of the phase contrast channel
        model_path:     path to trained .pth model (None = rules only)
        pixels_per_um:  pixel scale for area calculations
        keep_classes:   tuple of class names to keep (default: only "good")
        confidence_threshold: minimum softmax probability to trust prediction
                        (default rule only)
        use_rules:      whether to apply rule-based pre-filtering
        device:         torch device (None = auto)
        verbose:        print filtering statistics
        decision_threshold: P(bad) cutoff in (0, 1); None = default rule

    Returns:
        filtered_mask:  (Y, X) integer array with rejected cells set to 0
        report:         dict with filtering statistics
    """
    if decision_threshold is not None:
        t = _valid_decision_threshold(decision_threshold)
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
        edge_cells = detect_edge_cells(labeled_mask, margin=EDGE_MARGIN)
        for lbl in edge_cells:
            filtered[filtered == lbl] = 0
            report["details"][int(lbl)] = ("edge_cell", 1.0)
        removed |= edge_cells
        report["removed_edge"] = len(edge_cells)

        debris_cells = detect_debris_by_area(
            labeled_mask, pixels_per_um=pixels_per_um, min_area_um2=MIN_AREA_UM2
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
        clf, clf_lock = _cached_classifier(model_path, in_channels, device=device)

        # Extract crops for remaining cells (phase + mask + true area)
        labels_list = sorted(remaining)
        crops_list = []
        for lbl in labels_list:
            crop, _ = extract_cell_crop(image_channels, labeled_mask, lbl,
                                        phase_channel=phase_channel)
            if crop is not None:
                # True pixel area from original mask (shape-independent)
                true_area = int(np.sum(labeled_mask == lbl))
                crop = _append_area_channel_single(crop, true_area_px=true_area)
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
