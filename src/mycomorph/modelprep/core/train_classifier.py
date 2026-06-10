#!/usr/bin/env python3
"""
Train Cell Quality Classifier — ModelPrep
=========================================
Trains the CNN on labeled cell crops. Two input modes:

  1. **Arrays** (preferred for HDF5 pipeline). Pass ``crops`` and ``labels``
     directly; ``data_dir`` is ignored. Used by ``modelprep-cli train``.
  2. **Legacy directory**. Pass ``data_dir`` pointing at a
     ``crops/*.npy + labels.csv`` layout. Kept for migrations only.

Features:
    - Stratified three-way train/validation/test split, with graceful
      fall-back to a two-way split on tiny datasets and clear errors when a
      class is too rare to split at all
    - Channel-aware augmentation: intensity jitter/noise touches only the
      phase channel(s); the binary mask is re-binarised after geometric
      transforms and the constant area channel is preserved
    - Class-weighted loss to handle imbalanced labels
    - Early stopping with best-model checkpointing (selected on validation)
    - Decision threshold chosen on validation, final metrics reported on the
      held-out test set (so the headline number isn't model-selection-biased)
    - Fully seeded RNG (Python/NumPy/torch + DataLoader) for reproducible .pth
    - Training curves, confusion matrix, ROC/PR curves saved as PNGs + JSON
"""

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

# Module logger. ``train()`` tees this to the console AND a per-run train.log in
# the output directory, so every run keeps an on-disk record alongside its .pth.
log = logging.getLogger("modelprep.train")


def _setup_logging(output_dir: Path):
    log.setLevel(logging.INFO)
    log.propagate = False
    if not any(getattr(h, "_modelprep_console", False) for h in log.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(message)s"))
        ch._modelprep_console = True  # type: ignore[attr-defined]
        log.addHandler(ch)
    for h in list(log.handlers):
        if isinstance(h, logging.FileHandler):
            log.removeHandler(h)
            h.close()
    fh = logging.FileHandler(Path(output_dir) / "train.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)
    return fh


def _close_logging(fh) -> None:
    if fh is not None:
        log.removeHandler(fh)
        fh.close()


# ── Channel-aware augmentation ────────────────────────────────────────────────
#
# Crop channel layout (both the HDF5 and legacy paths end here):
#     [phase ..., mask, area]   — area is always the LAST channel (appended by
#     _append_area_channel_*), mask is second-to-last.
#
# Intensity augmentation (per-channel normalise + Gaussian noise) must only
# touch the phase channel(s): the mask is binary and the area channel is a
# constant plane, so noising them creates features the model never sees at
# inference time (train/inference skew). Geometric transforms are applied to
# phase+mask together (so they stay aligned), the mask is re-binarised
# afterwards (affine interpolation makes it fractional), and the constant area
# plane is regenerated rather than transformed (translate would 0-fill it).

def _phase_count(c: int) -> int:
    """Number of leading phase channels given total channel count ``c``."""
    if c >= 3:
        return c - 2          # [phase..., mask, area]
    if c == 2:
        return 1              # [phase, mask]
    return c                  # degenerate; treat all as phase


def _normalise_phase_inplace(x, n_phase):
    """Per-channel min-max normalise the first ``n_phase`` channels of x to [0, 1]."""
    for c in range(n_phase):
        cmin = x[c].min()
        cmax = x[c].max()
        if cmax > cmin:
            x[c] = (x[c] - cmin) / (cmax - cmin)


class _PhaseAwareTrainTransform:
    """Geometric aug on phase+mask, intensity aug on phase only, mask
    re-binarised, area channel preserved."""

    def __init__(self, max_noise_std: float = 0.03):
        import torchvision.transforms.v2 as T
        self.max_noise_std = max_noise_std
        self._geom = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply([T.RandomRotation(degrees=(90, 90))], p=0.5),
            T.RandomApply([
                T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
            ], p=0.3),
        ])

    def __call__(self, x):
        import torch as t
        c = x.shape[0]
        if c < 3:
            # No protected area channel; normalise phase, jitter it, leave the rest.
            x = x.clone()
            n_phase = _phase_count(c)
            x = self._geom(x)
            _normalise_phase_inplace(x, n_phase)
            std = t.rand(1).item() * self.max_noise_std
            if std > 0 and n_phase > 0:
                x[:n_phase] = (x[:n_phase] + t.randn_like(x[:n_phase]) * std).clamp(0, 1)
            return x

        area_val = float(x[-1].mean())
        pm = x[:-1].clone()                       # [phase..., mask]
        pm = self._geom(pm)                       # aligned geometric transform
        pm[-1] = (pm[-1] > 0.5).to(pm.dtype)      # re-binarise mask after interp
        n_phase = pm.shape[0] - 1
        _normalise_phase_inplace(pm, n_phase)
        std = t.rand(1).item() * self.max_noise_std
        if std > 0 and n_phase > 0:
            pm[:n_phase] = (pm[:n_phase] + t.randn_like(pm[:n_phase]) * std).clamp(0, 1)
        area = t.full((1, pm.shape[1], pm.shape[2]), area_val, dtype=pm.dtype)
        return t.cat([pm, area], dim=0)


class _PhaseAwareValTransform:
    """Per-channel normalise phase only; mask + area pass through unchanged."""

    def __call__(self, x):
        x = x.clone()
        _normalise_phase_inplace(x, _phase_count(x.shape[0]))
        return x


def make_augmentation_transforms():
    """Return ``(train_transform, val_transform)`` — channel-aware (see above)."""
    return _PhaseAwareTrainTransform(max_noise_std=0.03), _PhaseAwareValTransform()


class CellCropDataset:
    """PyTorch-compatible dataset wrapping numpy arrays + optional transforms."""

    def __init__(self, crops, labels, transform=None):
        import torch
        self.crops = torch.from_numpy(crops)
        self.labels = torch.from_numpy(labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        crop = self.crops[idx].clone()
        label = self.labels[idx]
        if self.transform is not None:
            crop = self.transform(crop)
        return crop, label


def compute_class_weights(labels, num_classes):
    """Inverse-frequency class weights for imbalanced datasets."""
    import torch
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.from_numpy(weights)


def _resolve_device(device):
    if device is None:
        from .device import detect_device
        return detect_device()
    return device


def _set_seed(seed: int) -> None:
    """Seed every RNG that affects a training run so the produced .pth is
    reproducible (split, weight init, augmentation, DataLoader shuffle)."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _make_splits(labels, val_fraction, test_fraction, seed):
    """Stratified train/val/test indices.

    Returns ``(train_idx, val_idx, test_idx, note)``. Falls back to a two-way
    split (empty ``test_idx``) when the dataset is too small for a held-out
    test set. Raises ``ValueError`` (caller-friendly) when a class is too rare
    to stratify at all.
    """
    from sklearn.model_selection import train_test_split

    labels = np.asarray(labels)
    n = len(labels)
    idx = np.arange(n)
    _classes, counts = np.unique(labels, return_counts=True)
    min_count = int(counts.min()) if len(counts) else 0
    want_test = bool(test_fraction) and test_fraction > 0

    if want_test and min_count >= 4:
        try:
            trainval, test = train_test_split(
                idx, test_size=test_fraction, stratify=labels, random_state=seed,
            )
            rel_val = val_fraction / (1.0 - test_fraction)
            train, val = train_test_split(
                trainval, test_size=rel_val, stratify=labels[trainval],
                random_state=seed,
            )
            note = (f"3-way stratified split "
                    f"(test={test_fraction:.0%}, val={val_fraction:.0%})")
            return train, val, test, note
        except ValueError:
            pass  # fall through to 2-way

    train, val = train_test_split(
        idx, test_size=val_fraction, stratify=labels, random_state=seed,
    )
    if want_test:
        note = ("dataset too small for a held-out test set — using a 2-way "
                "train/val split; reported metrics are VALIDATION (the model "
                "was selected on them, so read them as optimistic).")
    else:
        note = f"2-way stratified split (val={val_fraction:.0%})"
    return train, val, np.array([], dtype=int), note


def _evaluate(model, loader, device):
    """Run ``model`` over ``loader`` → ``(true, preds, probs)`` numpy arrays."""
    import torch
    model.eval()
    trues, preds, probs = [], [], []
    with torch.no_grad():
        for batch_crops, batch_labels in loader:
            batch_crops = batch_crops.to(device)
            logits = model(batch_crops)
            p = torch.softmax(logits, dim=1).cpu().numpy()
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(batch_labels.numpy())
            probs.append(p)
    true = np.array(trues)
    pred = np.array(preds)
    prob = np.concatenate(probs, axis=0) if probs else np.zeros((0, 2), dtype=np.float32)
    return true, pred, prob


def _binary_curves(true, scores_pos):
    """ROC/PR curves + Youden-J optimal threshold for the positive (good=… no,
    pos_label=1) class. Returns None if only one class is present."""
    true = np.asarray(true)
    if len(np.unique(true)) < 2:
        return None
    from sklearn.metrics import (
        roc_curve, auc, precision_recall_curve, average_precision_score,
    )
    fpr, tpr, roc_th = roc_curve(true, scores_pos, pos_label=1)
    roc_auc = float(auc(fpr, tpr))
    best = int(np.argmax(tpr - fpr))
    thr = float(roc_th[best])
    precision, recall, pr_th = precision_recall_curve(true, scores_pos, pos_label=1)
    ap = float(average_precision_score(true, scores_pos, pos_label=1))
    baseline = float((true == 1).mean())
    return {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
                "thresholds": roc_th.tolist(), "auc": roc_auc, "best_idx": best},
        "pr": {"precision": precision.tolist(), "recall": recall.tolist(),
               "thresholds": pr_th.tolist(), "ap": ap, "baseline": baseline},
        "optimal_threshold": thr,
    }


def train(
    *,
    data_dir=None,
    crops=None,
    labels=None,
    output_dir,
    epochs=80,
    batch_size=32,
    lr=1e-3,
    val_fraction=0.2,
    test_fraction=0.15,
    patience=15,
    device=None,
    seed=42,
    freeze_backbone=False,
    phase_only=True,
    phase_channel_in_crop=0,
    pretrained_path=None,
    provenance=None,
    progress_cb=None,
):
    """
    Train the cell quality CNN.

    Either pass ``crops`` + ``labels`` (preferred — comes straight out of
    ``LabelsStore.to_training_dataset()``) or ``data_dir`` for the legacy
    ``crops/*.npy + labels.csv`` layout.

    The split is three-way (train/val/test): the model is checkpointed on
    validation accuracy, the decision threshold is chosen on validation, and
    the reported metrics (accuracy, AUC, AP, confusion) are computed on the
    held-out test set. On small datasets it falls back to a two-way split and
    says so (``eval_split`` in the returned summary). ``test_fraction=0``
    forces the old two-way behaviour.

    ``provenance`` (optional dict — e.g. source_h5, used_uids hash, label
    counts) is recorded verbatim in ``training_config.json``.

    Returns a summary dict (see the bottom of this function for keys).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from .cell_quality_classifier import (
        CLASS_NAMES,
        NUM_CLASSES,
        _build_model,
        load_crop_dataset,
    )

    if crops is None and data_dir is None:
        raise ValueError("train(): pass either `crops`+`labels` or `data_dir`.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _log_fh = _setup_logging(output_dir)

    _set_seed(seed)
    device = _resolve_device(device)
    _print_device(device)
    # On Apple MPS, let unsupported ops silently fall back to CPU instead of
    # aborting the whole run.
    if device == "mps":
        import os as _os
        _os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # ── Load data ─────────────────────────────────────────────────────────
    if crops is None:
        log.info("Loading dataset from disk...")
        crops, labels, _meta = load_crop_dataset(
            data_dir, phase_only=phase_only,
            phase_channel_in_crop=phase_channel_in_crop,
        )
    else:
        crops = np.asarray(crops, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64)

    n_samples = len(labels)
    in_channels = crops.shape[1]

    class_counts = {CLASS_NAMES[i]: int((labels == i).sum())
                    for i in range(NUM_CLASSES)}
    log.info(f"  {n_samples} labeled cells, {in_channels} channels per crop")
    for name, count in class_counts.items():
        log.info(f"    {name}: {count}")

    # ── Guard rails: need both classes, and ≥2 of the rarer one ────────────
    present = np.unique(labels)
    if len(present) < 2:
        only = CLASS_NAMES[int(present[0])] if len(present) == 1 else "a single class"
        raise ValueError(
            f"All {n_samples} labelled cells are '{only}'. Label at least a few "
            f"cells of the other class before training a binary classifier."
        )
    min_count = min(class_counts.values())
    if min_count < 2:
        raise ValueError(
            f"Too few labels in the rarer class to split "
            f"(good={class_counts.get('good', 0)}, bad={class_counts.get('bad', 0)}). "
            f"Label at least 2 (ideally many more) of each class before training."
        )
    if n_samples < 20:
        log.info("WARNING: very few labeled cells — results will be unreliable.")

    # ── Train/val/test split (stratified, three-way when feasible) ─────────
    train_idx, val_idx, test_idx, split_note = _make_splits(
        labels, val_fraction, test_fraction, seed,
    )
    log.info(f"  Split: train={len(train_idx)}  val={len(val_idx)}  "
          f"test={len(test_idx)}  · {split_note}")
    if progress_cb is not None:
        progress_cb(0.0, f"Split: {len(train_idx)}/{len(val_idx)}/{len(test_idx)} "
                         f"(train/val/test)")

    train_transform, val_transform = make_augmentation_transforms()
    train_ds = CellCropDataset(crops[train_idx], labels[train_idx], train_transform)
    val_ds = CellCropDataset(crops[val_idx], labels[val_idx], val_transform)
    loader_gen = torch.Generator()
    loader_gen.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=False, generator=loader_gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)
    test_loader = None
    if len(test_idx) > 0:
        test_ds = CellCropDataset(crops[test_idx], labels[test_idx], val_transform)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=0)

    # ── Model, loss, optimizer ────────────────────────────────────────────
    model = _build_model(in_channels, NUM_CLASSES).to(device)

    if pretrained_path is not None:
        try:
            state = torch.load(str(pretrained_path), map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model_sd = model.state_dict()
            loadable = {k: v for k, v in state.items()
                        if k in model_sd and v.shape == model_sd[k].shape}
            reinit = sorted(k for k, v in state.items()
                            if k in model_sd and v.shape != model_sd[k].shape)
            missing = sorted(k for k in model_sd if k not in state)
            model.load_state_dict(loadable, strict=False)
            log.info(f"  Loaded {len(loadable)}/{len(model_sd)} layers from {pretrained_path}")
            if reinit:
                log.info(f"    re-initialised (shape mismatch — e.g. head/first conv): {reinit}")
            if missing:
                log.info(f"    left at fresh init (absent in checkpoint): {missing}")
        except Exception as e:
            log.info(f"  WARNING: could not load pretrained weights ({e}); training from scratch")

    if freeze_backbone:
        frozen = 0
        for p in model.features.parameters():
            p.requires_grad = False
            frozen += 1
        log.info(f"  Backbone frozen ({frozen} tensors) — training the classifier head only.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Model: {n_params:,} trainable parameters")

    class_weights = compute_class_weights(labels[train_idx], NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    log.info(f"\nTraining for up to {epochs} epochs (patience={patience})...\n")

    epoch = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch_crops, batch_labels in train_loader:
            batch_crops = batch_crops.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_crops)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_labels)
            train_correct += (logits.argmax(1) == batch_labels).sum().item()
            train_total += len(batch_labels)
        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_crops, batch_labels in val_loader:
                batch_crops = batch_crops.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_crops)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item() * len(batch_labels)
                preds = logits.argmax(1)
                val_correct += (preds == batch_labels).sum().item()
                val_total += len(batch_labels)
        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        log.info(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.3f} | "
            f"LR={lr_now:.2e}"
        )
        if progress_cb is not None:
            progress_cb(
                (epoch + 1) / epochs,
                f"Epoch {epoch+1}/{epochs}: train_acc={train_acc:.3f}  val_acc={val_acc:.3f}",
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            log.info(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
            if progress_cb is not None:
                progress_cb(
                    1.0,
                    f"Early stopped at epoch {epoch + 1}. Best val acc {best_val_acc:.3f}.",
                )
            break
    else:
        if progress_cb is not None:
            progress_cb(1.0, f"Done · best val acc {best_val_acc:.3f} at epoch {best_epoch}")

    # ── Restore best (validation-selected) weights ─────────────────────────
    log.info(f"\nBest validation accuracy: {best_val_acc:.3f} at epoch {best_epoch}")
    model.load_state_dict(
        torch.load(output_dir / "best_model.pth", map_location=device, weights_only=True)
    )
    torch.save(model.state_dict(), output_dir / "final_model.pth")

    # ── Threshold on validation, final metrics on held-out test ────────────
    val_true, _val_preds, val_probs = _evaluate(model, val_loader, device)
    val_curves = (_binary_curves(val_true, val_probs[:, 1])
                  if NUM_CLASSES == 2 and val_probs.shape[1] >= 2 else None)
    decision_threshold = val_curves["optimal_threshold"] if val_curves else 0.5

    if test_loader is not None:
        eval_split = "test"
        eval_true, eval_preds, eval_probs = _evaluate(model, test_loader, device)
    else:
        eval_split = "val"
        eval_true, eval_preds, eval_probs = val_true, _val_preds, val_probs

    eval_acc = float((eval_preds == eval_true).mean()) if len(eval_true) else float("nan")
    eval_curves = (_binary_curves(eval_true, eval_probs[:, 1])
                   if NUM_CLASSES == 2 and eval_probs.shape[1] >= 2 else None)
    roc_auc = eval_curves["roc"]["auc"] if eval_curves else float("nan")
    ap = eval_curves["pr"]["ap"] if eval_curves else float("nan")

    from sklearn.metrics import classification_report, confusion_matrix

    present_classes = sorted(set(eval_true.tolist()) | set(eval_preds.tolist()))
    target_names = [CLASS_NAMES[i] for i in present_classes]
    log.info("\n" + "=" * 60)
    log.info(f"CLASSIFICATION REPORT ({eval_split} set, n={len(eval_true)})")
    log.info("=" * 60)
    log.info(classification_report(eval_true, eval_preds,
                                labels=present_classes,
                                target_names=target_names,
                                zero_division=0))

    confusion = None
    if len(present) == 2:
        cm = confusion_matrix(eval_true, eval_preds, labels=[0, 1])
        tn, fp, fn, tp = (int(v) for v in cm.ravel())
        confusion = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    if eval_curves is not None:
        log.info(f"  {eval_split}: ROC AUC={roc_auc:.3f}  AP={ap:.3f}  "
              f"(decision threshold from val={decision_threshold:.3f})")

    # ── Persist config + metrics ───────────────────────────────────────────
    config = {
        "in_channels": in_channels,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "eval_split": eval_split,
        "split_note": split_note,
        "class_counts": class_counts,
        "best_val_acc": float(best_val_acc),
        "best_epoch": best_epoch,
        "eval_acc": eval_acc,
        "roc_auc": roc_auc,
        "average_precision": ap,
        "decision_threshold": float(decision_threshold),
        "confusion": confusion,
        "total_epochs": epoch + 1,
        "lr": lr,
        "batch_size": batch_size,
        "patience": patience,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "seed": int(seed),
        "freeze_backbone": bool(freeze_backbone),
        "trainer": "modelprep",
        "trainer_version": _get_version(),
        "provenance": provenance or {},
    }
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    metrics = {
        "num_classes": int(NUM_CLASSES),
        "class_names": list(CLASS_NAMES),
        "eval_split": eval_split,
        "decision_threshold": float(decision_threshold),
        "val_optimal_threshold": (val_curves["optimal_threshold"] if val_curves else None),
        "binary": eval_curves,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ── Plots ──────────────────────────────────────────────────────────────
    try:
        _plot_artefacts(
            output_dir, history, eval_true, eval_preds,
            target_names, present_classes, eval_curves, eval_split,
        )
    except ImportError:
        log.info("(matplotlib not available — skipped plot generation)")

    log.info(f"\nModel saved: {output_dir / 'best_model.pth'}")
    log.info(f"Config saved: {output_dir / 'training_config.json'}")
    _close_logging(_log_fh)

    return {
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "eval_split": eval_split,
        "split_note": split_note,
        "eval_acc": eval_acc,
        "roc_auc": roc_auc,
        "average_precision": ap,
        "decision_threshold": float(decision_threshold),
        "confusion": confusion,
        "class_counts": class_counts,
        "seed": int(seed),
        "model_path": str(output_dir / "best_model.pth"),
    }


def _print_device(device):
    import torch
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
        log.info(f"Device: CUDA GPU ({gpu_name})")
    elif device == "mps":
        log.info("Device: Apple Silicon GPU (MPS)")
    else:
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if has_cuda or has_mps:
            log.info("Device: CPU  (GPU available — pass --device cuda or --device mps)")
        else:
            log.info("Device: CPU  (no GPU detected)")


def _get_version() -> str:
    try:
        from modelprep import __version__
        return __version__
    except Exception:
        return "unknown"


def _plot_artefacts(
    output_dir, history, eval_true, eval_preds,
    target_names, present_classes, curves, eval_split,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Validation")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training Loss")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Validation")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Training Accuracy")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()

    cm = confusion_matrix(eval_true, eval_preds, labels=present_classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(len(target_names)))
    ax.set_yticks(range(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticklabels(target_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({eval_split})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    if curves is None:
        return
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    roc = curves["roc"]; pr = curves["pr"]
    fpr, tpr = np.array(roc["fpr"]), np.array(roc["tpr"])
    ax_roc.plot(fpr, tpr, color="#2166ac", lw=2, label=f"AUC = {roc['auc']:.3f}")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    bi = roc["best_idx"]
    ax_roc.plot(fpr[bi], tpr[bi], "g^", ms=10,
                label=f"Optimal threshold={curves['optimal_threshold']:.2f}")
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curve ({eval_split})")
    ax_roc.legend(loc="lower right", fontsize=8); ax_roc.grid(True, alpha=0.3)

    ax_pr.plot(pr["recall"], pr["precision"], color="#b2182b", lw=2,
               label=f"AP = {pr['ap']:.3f}")
    ax_pr.axhline(pr["baseline"], color="k", ls="--", lw=1, alpha=0.5,
                  label=f"Baseline ({pr['baseline']:.2f})")
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend(loc="lower left", fontsize=8); ax_pr.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "roc_pr_curves.png", dpi=150)
    plt.close()
