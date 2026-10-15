#!/usr/bin/env python3
"""
Cell Quality Classifier — ModelPrep (training side)
===================================================
CNN model definition + training-side data loaders.

This file is the **training half** of the classifier surface that MycoMorph
uses for inference. The model-architecture section below is mirrored
byte-for-byte in MycoMorph so a ``.pth`` trained here loads cleanly in the
pipeline there.

Categories (binary):
    0 = good   — well-segmented, in-focus single cell to KEEP
    1 = bad    — out-of-focus / clumped / edge / debris — REJECT
"""

import numpy as np
from pathlib import Path


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
# Below: ModelPrep-only training-side data I/O. The TIFF crop extraction and
# rule-based filters that live in the MycoMorph copy do NOT belong here — this
# package consumes pre-extracted HDF5 crops (see crops_h5.py).


# ── Training-side: legacy .npy/labels.csv dataset loader ──────────────────────
#
# Used by train_classifier.train() when given a legacy directory layout
# (``crops/*.npy`` + ``labels.csv``). The HDF5 path (the new default) bypasses
# this and calls ``train_classifier.train(crops=..., labels=...)`` directly.

def save_crop_dataset(crops_dict, labels_dict, output_dir):
    """
    Save extracted crops and labels to disk in the legacy training format.

    Directory structure:
        output_dir/
        ├── crops/<identifier>.npy
        ├── labels.csv     (filename, label_idx, label_name)
        └── metadata.json  (crop_size, n_channels, class_names)
    """
    import json
    import csv

    output_dir = Path(output_dir)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    n_channels = None

    for ident, crop in crops_dict.items():
        fname = f"{ident}.npy"
        np.save(crops_dir / fname, crop)
        label_idx = labels_dict.get(ident, -1)
        label_name = CLASS_NAMES[label_idx] if 0 <= label_idx < NUM_CLASSES else "unlabeled"
        rows.append([fname, label_idx, label_name])
        if n_channels is None:
            n_channels = crop.shape[0]

    with open(output_dir / "labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label_idx", "label_name"])
        writer.writerows(rows)

    meta = {
        "crop_size": CROP_SIZE,
        "n_channels": n_channels,
        "class_names": CLASS_NAMES,
        "n_samples": len(rows),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_crop_dataset(dataset_dir, phase_only=False, phase_channel_in_crop=0):
    """
    Load a saved crop dataset (legacy .npy + labels.csv layout).

    metadata.json is optional — if absent, n_channels is inferred from the
    first .npy file in crops/ and class_names defaults to CLASS_NAMES.

    Returns:
        crops:  (N, C, H, W) float32 array
        labels: (N,) int array
        meta:   dict with dataset metadata
    """
    import json
    import csv

    dataset_dir = Path(dataset_dir)

    # ── Load or infer metadata ─────────────────────────────────────────────
    meta_path = dataset_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        crops_dir = dataset_dir / "crops"
        n_channels = None
        if crops_dir.exists():
            npy_files = sorted(crops_dir.glob("*.npy"))
            if npy_files:
                sample = np.load(npy_files[0])
                n_channels = sample.shape[0]
        meta = {
            "crop_size": CROP_SIZE,
            "n_channels": n_channels,
            "class_names": CLASS_NAMES,
            "n_samples": None,
        }
        print(
            f"  Note: metadata.json not found in {dataset_dir}.\n"
            f"  Inferred n_channels={n_channels} from crops directory."
        )

    # ── Load crops and labels ──────────────────────────────────────────────
    labels_csv = dataset_dir / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"labels.csv not found in {dataset_dir}.\n"
            f"Run `modelprep-cli label ...` first to generate labeled training data."
        )

    crops_list = []
    labels_list = []
    areas_list = []

    with open(labels_csv) as f:
        reader = csv.DictReader(f)
        has_area_col = "area_px" in (reader.fieldnames or [])
        for row in reader:
            label_idx = int(row["label_idx"])
            if label_idx < 0:
                continue
            fname = row["filename"]
            if not fname.endswith(".npy"):
                fname = fname + ".npy"
            crop_path = dataset_dir / "crops" / fname
            if not crop_path.exists():
                print(f"  Warning: crop file not found, skipping: {crop_path.name}")
                continue
            crop = np.load(crop_path)
            crops_list.append(crop)
            labels_list.append(label_idx)
            if has_area_col and row.get("area_px", "").strip():
                areas_list.append(int(row["area_px"]))
            else:
                areas_list.append(None)

    if not crops_list:
        raise ValueError(
            f"No labeled crops found in {dataset_dir}."
        )

    crops = np.stack(crops_list, axis=0).astype(np.float32)
    labels = np.array(labels_list, dtype=np.int64)
    true_areas = (np.array(areas_list, dtype=np.float32)
                  if all(a is not None for a in areas_list) else None)
    if true_areas is None and any(a is not None for a in areas_list):
        true_areas = np.array(
            [a if a is not None else 0 for a in areas_list], dtype=np.float32
        )

    # ── Remap legacy 5-class labels to binary (good=0, bad=1) ─────────────
    stored_classes = meta.get("class_names", CLASS_NAMES)
    if len(stored_classes) > 2 or labels.max() > 1:
        n_remapped = int((labels > 0).sum())
        labels = (labels > 0).astype(np.int64)
        meta["class_names"] = CLASS_NAMES
        print(f"  Remapped legacy {len(stored_classes)}-class labels → binary "
              f"(good/bad). {n_remapped} cells mapped to 'bad'.")

    # ── Phase-only slicing ─────────────────────────────────────────────────
    if phase_only and crops.shape[1] > 2:
        orig_ch = crops.shape[1]
        crops = crops[:, [phase_channel_in_crop, -1], :, :]
        print(f"  Phase-only mode: sliced crops from {orig_ch} → 2 channels")

    # ── Append area channel ───────────────────────────────────────────────
    crops = _append_area_channel_batch(crops, true_areas_px=true_areas)
    meta["n_channels"] = crops.shape[1]
    src = "true pixel areas from labels.csv" if true_areas is not None else "mask fallback"
    print(f"  Added area channel → {crops.shape[1]} total channels  [{src}]")

    meta["n_samples"] = len(labels)

    return crops, labels, meta
