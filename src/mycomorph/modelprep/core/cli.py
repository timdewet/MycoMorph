"""ModelPrep CLI — ``modelprep-cli``.

Subcommands:
  ``info``          — describe an ``all_crops.h5`` (cells, channels, schema)
  ``models``        — list bundled classifier presets
  ``verify-model``  — confirm a ``.pth`` loads + runs (ready for MycoPrep)
  ``label``         — interactive labeller (matplotlib grid)
  ``review``        — review + correct an existing model's predictions
  ``train``         — train a binary cell-quality CNN from labelled HDF5 crops
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer

from .crops_h5 import HDF5CropsReader
from .labels_store import LabelsStore, default_labels_dir_for


app = typer.Typer(
    add_completion=False,
    help="ModelPrep — labeller + trainer for the cell-quality classifier. "
         "Consumes MycoPrep's HDF5 crops (all_crops.h5).",
)


# ── Preset model registry ────────────────────────────────────────────────────

def _bundled_models_dir() -> Path:
    return Path(__file__).resolve().parent / "models"


def list_preset_models() -> dict[str, Path]:
    """Return {preset_name: absolute_path} for every ``.pth`` in the bundled
    models directory."""
    models_dir = _bundled_models_dir()
    if not models_dir.exists():
        return {}
    presets = {}
    for pth in sorted(models_dir.glob("*.pth")):
        name = pth.stem
        if name.endswith("_best_model"):
            name = name[: -len("_best_model")]
        presets[name] = pth
    return presets


def resolve_classifier_preset(name: str) -> Path:
    """Return the absolute path of a bundled preset by name."""
    presets = list_preset_models()
    if name in presets:
        return presets[name]
    # Allow passing a direct path too
    p = Path(name)
    if p.exists():
        return p
    raise FileNotFoundError(
        f"Unknown preset {name!r}. Bundled presets: {sorted(presets) or '(none)'}"
    )


# ── Default outputs ──────────────────────────────────────────────────────────

def _default_output_dir() -> Path:
    return Path.home() / ".modelprep" / "models" / f"trained_{int(time.time())}"


# ── Commands ─────────────────────────────────────────────────────────────────

@app.command()
def info(
    h5: Path = typer.Option(
        ..., "--h5", "-i", exists=True, file_okay=True, dir_okay=False,
        readable=True, help="Path to an all_crops.h5 file.",
    ),
) -> None:
    """Describe an ``all_crops.h5`` file (cells, channels, schema)."""
    with HDF5CropsReader(h5) as reader:
        typer.echo(reader.summary())
        typer.echo(f"  phase channel: idx={reader.phase_channel_idx}  "
                   f"name={reader.channel_names[reader.phase_channel_idx]!r}")
        typer.echo(f"  mask channel:  idx={reader.mask_channel_idx}  "
                   f"name={reader.channel_names[reader.mask_channel_idx]!r}")
        typer.echo(f"  native crop:   {reader.native_crop_size}×{reader.native_crop_size}")
        typer.echo(f"  classifier projection: 3 channels × "
                   f"{reader.classifier_size}×{reader.classifier_size}")


@app.command()
def models() -> None:
    """List bundled classifier presets."""
    presets = list_preset_models()
    if not presets:
        typer.echo("(no bundled presets)")
        return
    for name, path in presets.items():
        typer.echo(f"  {name:12s}  {path}")


@app.command(name="verify-model")
def verify_model(
    model: str = typer.Option(
        ..., "--model", "-m",
        help="Path to a trained .pth, or a bundled preset name (e.g. 'mtb').",
    ),
    in_channels: int = typer.Option(
        3, "--in-channels", help="Input channels the model expects (default 3).",
    ),
) -> None:
    """Verify a ``.pth`` loads + runs through the shared classifier loader.

    Catches a broken hand-off before you point MycoPrep at the model: loads it
    exactly the way MycoPrep will, runs a synthetic batch, and checks the
    output shape / class names.
    """
    import json

    import numpy as np

    from .cell_quality_classifier import (
        CLASS_NAMES, NUM_CLASSES, CROP_SIZE, CellQualityClassifier,
    )

    # Resolve a bundled preset name or a direct .pth path.
    try:
        model_path = resolve_classifier_preset(model)
    except FileNotFoundError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=2)

    try:
        clf = CellQualityClassifier(str(model_path), in_channels=in_channels, device="cpu")
        dummy = np.zeros((2, in_channels, CROP_SIZE, CROP_SIZE), dtype=np.float32)
        idxs, probs = clf.predict_batch(dummy)
        if probs.shape != (2, NUM_CLASSES):
            raise ValueError(f"unexpected output shape {probs.shape}, expected (2, {NUM_CLASSES})")
    except Exception as e:  # noqa: BLE001
        typer.echo(
            f"FAILED: {model_path.name} did not load/run as a {in_channels}-channel "
            f"cell-quality classifier: {e}",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(
        f"OK: {model_path.name} loads as a {in_channels}-channel classifier "
        f"({NUM_CLASSES} classes {CLASS_NAMES}); synthetic batch → probs {probs.shape}. "
        f"Ready to drop into MycoPrep's Custom-model picker."
    )
    cfg = model_path.parent / "training_config.json"
    if cfg.exists():
        try:
            c = json.loads(cfg.read_text())
            typer.echo(
                f"   trained: eval_split={c.get('eval_split')} "
                f"eval_acc={c.get('eval_acc')} auc={c.get('roc_auc')} "
                f"threshold={c.get('decision_threshold')}"
            )
        except Exception:  # noqa: BLE001
            pass


@app.command()
def doctor() -> None:
    """Environment self-check — versions, compute device, Qt plugin, presets.

    Run this first if the GUI won't launch: it prints the
    ``QT_QPA_PLATFORM_PLUGIN_PATH`` fix when the Qt platform plugin can't be
    found (the most common first-run failure).
    """
    import hashlib
    import importlib
    import json as _json
    import os as _os
    import platform
    import sys

    import modelprep

    typer.echo(f"ModelPrep {modelprep.__version__}")
    typer.echo(f"Python    {platform.python_version()}  ({sys.platform})")

    for mod in ("numpy", "scipy", "sklearn", "skimage", "h5py", "pandas",
                "matplotlib", "typer", "qtawesome", "psutil"):
        try:
            m = importlib.import_module(mod)
            typer.echo(f"  {mod:12s} {getattr(m, '__version__', '?')}")
        except Exception as e:  # noqa: BLE001
            typer.echo(f"  {mod:12s} MISSING ({e})")

    try:
        import torch
        from .device import detect_device, device_label
        typer.echo(f"  {'torch':12s} {torch.__version__}  →  device: "
                   f"{device_label()} ({detect_device()})")
    except Exception as e:  # noqa: BLE001
        typer.echo(f"  {'torch':12s} MISSING ({e})")

    # PyQt6 + Qt platform plugin (the usual first-run snag).
    try:
        import PyQt6
        from PyQt6.QtCore import QT_VERSION_STR
        typer.echo(f"  {'PyQt6/Qt':12s} {QT_VERSION_STR}")
        envp = _os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
        wheel_plat = Path(PyQt6.__file__).parent / "Qt6" / "plugins" / "platforms"
        if envp:
            typer.echo(f"    QT_QPA_PLATFORM_PLUGIN_PATH={envp}")
        elif wheel_plat.exists():
            typer.echo(f"    Qt platform plugins: {wheel_plat}")
            typer.echo("    If the GUI aborts with \"Could not find the Qt platform "
                       "plugin\", run:")
            typer.echo(f"      export QT_QPA_PLATFORM_PLUGIN_PATH={wheel_plat}")
    except Exception as e:  # noqa: BLE001
        typer.echo(f"  {'PyQt6':12s} MISSING ({e})")

    presets = list_preset_models()
    typer.echo(f"  presets: {sorted(presets) or '(none)'}")
    for name, p in presets.items():
        sidecar = p.with_suffix(".json")
        if sidecar.exists():
            try:
                meta = _json.loads(sidecar.read_text())
                sha = hashlib.sha256(p.read_bytes()).hexdigest()
                status = "OK" if sha == meta.get("sha256") else "SHA MISMATCH"
                typer.echo(f"    {name}: integrity {status}")
            except Exception:  # noqa: BLE001
                pass


@app.command()
def label(
    h5: Path = typer.Option(
        ..., "--h5", "-i", exists=True, file_okay=True, dir_okay=False,
        readable=True, help="Path to an all_crops.h5 file.",
    ),
    labels_dir: Optional[Path] = typer.Option(
        None, "--labels-dir", help="Where labels.csv lives. "
        "Default: <h5 parent>/modelprep_labels/",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Preset name (e.g. 'mtb') or path to a .pth. If omitted, cells "
             "are shown in random order rather than ranked by uncertainty.",
    ),
    n_pool: int = typer.Option(
        1000, "--n-pool", "-n", min=1,
        help="Number of unseen cells to load for this labelling session.",
    ),
) -> None:
    """Launch the interactive labeller."""
    from .label_cells import run_grid_labeling

    model_path = None
    if model is not None:
        try:
            model_path = resolve_classifier_preset(model)
        except FileNotFoundError as e:
            typer.echo(f"ERROR: {e}", err=True)
            raise typer.Exit(code=2)

    run_grid_labeling(
        h5_path=h5,
        labels_dir=labels_dir,
        model_path=model_path,
        n_pool=n_pool,
    )


@app.command()
def review(
    h5: Path = typer.Option(..., "--h5", "-i", exists=True),
    model: str = typer.Option(
        ..., "--model", "-m",
        help="Preset name (e.g. 'mtb') or path to a .pth file.",
    ),
    labels_dir: Optional[Path] = typer.Option(None, "--labels-dir"),
    mode: str = typer.Option(
        "uncertain", "--mode",
        help="'uncertain' = only show low-confidence cells; 'all' = show every "
             "cell sorted by confidence.",
    ),
    confidence_threshold: float = typer.Option(
        0.85, "--confidence-threshold",
        help="In 'uncertain' mode, only show cells below this confidence.",
    ),
    include_labeled: bool = typer.Option(
        False, "--include-labeled/--unlabeled-only",
        help="By default skip cells that already have a label.",
    ),
) -> None:
    """Review and correct a trained classifier's predictions."""
    from .review_classifications import run_review_grid

    try:
        model_path = resolve_classifier_preset(model)
    except FileNotFoundError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=2)

    if mode not in ("uncertain", "all"):
        typer.echo(f"ERROR: --mode must be 'uncertain' or 'all', got {mode!r}", err=True)
        raise typer.Exit(code=2)

    run_review_grid(
        h5_path=h5,
        model_path=model_path,
        labels_dir=labels_dir,
        mode=mode,
        confidence_threshold=confidence_threshold,
        only_unlabeled=not include_labeled,
    )


@app.command()
def train(
    h5: Path = typer.Option(..., "--h5", "-i", exists=True),
    labels_dir: Optional[Path] = typer.Option(
        None, "--labels-dir",
        help="Directory holding labels.csv. Default: <h5 parent>/modelprep_labels/",
    ),
    output: Path = typer.Option(
        None, "--output", "-o",
        help="Where to write best_model.pth + training_config.json. "
             "Default: ~/.modelprep/models/trained_<ts>/",
    ),
    epochs: int = typer.Option(80, "--epochs", min=1),
    batch_size: int = typer.Option(32, "--batch-size", min=1),
    lr: float = typer.Option(1e-3, "--lr", min=1e-7),
    val_fraction: float = typer.Option(0.2, "--val-fraction", min=0.01, max=0.9),
    test_fraction: float = typer.Option(
        0.15, "--test-fraction", min=0.0, max=0.8,
        help="Held-out test fraction for unbiased final metrics. 0 disables "
             "(2-way train/val split). Auto-falls back to 2-way on tiny datasets.",
    ),
    patience: int = typer.Option(15, "--patience", min=1),
    seed: int = typer.Option(42, "--seed", help="RNG seed for reproducible runs."),
    freeze_backbone: bool = typer.Option(
        False, "--freeze-backbone/--no-freeze-backbone",
        help="Train only the classifier head (freeze conv backbone). Useful "
             "when fine-tuning from a preset on a small label set.",
    ),
    pretrained: Optional[str] = typer.Option(
        None, "--pretrained",
        help="Preset name or path to a .pth to fine-tune from.",
    ),
    device: Optional[str] = typer.Option(
        None, "--device",
        help="Torch device override: 'cuda', 'mps', 'cpu'. Default: auto-detect.",
    ),
) -> None:
    """Train the binary cell-quality CNN from labelled HDF5 crops."""
    import numpy as np

    from .cell_quality_classifier import CLASS_NAMES, NUM_CLASSES
    from .labels_store import LABELS_SCHEMA_VERSION, cell_uids_sha256
    from .train_classifier import train as run_train

    h5_path = Path(h5)
    labels_dir_path = Path(labels_dir) if labels_dir else default_labels_dir_for(h5_path)
    if not labels_dir_path.exists():
        typer.echo(
            f"ERROR: labels dir not found: {labels_dir_path}\n"
            f"Run `modelprep-cli label --h5 {h5_path}` to label some cells first.",
            err=True,
        )
        raise typer.Exit(code=2)

    pretrained_path: Optional[Path] = None
    if pretrained is not None:
        try:
            pretrained_path = resolve_classifier_preset(pretrained)
        except FileNotFoundError as e:
            typer.echo(f"ERROR: {e}", err=True)
            raise typer.Exit(code=2)

    if output is None:
        output = _default_output_dir()

    reader = HDF5CropsReader(h5_path)
    store = LabelsStore(labels_dir_path)
    try:
        crops, labels, used_uids = store.to_training_dataset(reader)
    except ValueError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=2)
    finally:
        reader.close()

    typer.echo(f"Training on {len(used_uids)} labelled cells "
               f"(crops shape={crops.shape}, labels shape={labels.shape})")
    typer.echo(f"Output dir: {output}")
    output.mkdir(parents=True, exist_ok=True)

    class_counts = {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(NUM_CLASSES)}
    provenance = {
        "source_h5": h5_path.name,
        "labels_dir": str(labels_dir_path),
        "n_labelled": int(len(used_uids)),
        "used_uids_sha256": cell_uids_sha256(used_uids),
        "class_counts": class_counts,
        "labels_schema_version": LABELS_SCHEMA_VERSION,
    }

    summary = run_train(
        crops=crops,
        labels=labels,
        output_dir=output,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        patience=patience,
        seed=seed,
        freeze_backbone=freeze_backbone,
        device=device,
        pretrained_path=str(pretrained_path) if pretrained_path else None,
        provenance=provenance,
    )

    typer.echo("")
    typer.echo(f"  best_val_acc:    {summary['best_val_acc']:.3f} "
               f"at epoch {summary['best_epoch']}")
    split = summary.get("eval_split", "val")
    acc = summary.get("eval_acc")
    if acc is not None and acc == acc:  # not NaN
        typer.echo(f"  {split}_accuracy:  {acc:.3f}  (held-out)"
                   if split == "test" else f"  {split}_accuracy:  {acc:.3f}")
    auc = summary.get("roc_auc")
    if auc is not None and auc == auc:
        typer.echo(f"  {split} ROC AUC:  {auc:.3f}   AP: {summary.get('average_precision', float('nan')):.3f}")
    typer.echo(f"  threshold:       {summary.get('decision_threshold', 0.5):.3f}  (chosen on validation)")
    if summary.get("confusion"):
        c = summary["confusion"]
        typer.echo(f"  confusion ({split}): tn={c['tn']} fp={c['fp']} fn={c['fn']} tp={c['tp']}")
    if "no held-out test" in summary.get("split_note", ""):
        typer.echo("  NOTE: dataset too small for a held-out test set — metrics are "
                   "validation (optimistic). Label more cells for an honest estimate.")
    typer.echo(f"  model:           {summary['model_path']}")
    typer.echo(f"  config:          {output / 'training_config.json'}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
