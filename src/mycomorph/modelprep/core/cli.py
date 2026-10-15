"""ModelPrep CLI — ``modelprep-cli``.

Subcommands:
  ``info``          — describe an ``all_crops.h5`` (cells, channels, schema)
  ``models``        — list bundled classifier presets
  ``verify-model``  — confirm a ``.pth`` loads + runs (ready for MycoMorph)
  ``label``         — interactive labeller (matplotlib grid)
  ``review``        — review + correct an existing model's predictions
  ``train``         — train a binary cell-quality CNN from labelled HDF5 crops
  ``feature-model`` — train the feature-based comparison model
  ``evaluate``      — score a saved CNN or feature model on the current hold-out
  ``learning-curve``— train on growing subsets, score each on the hold-out
  ``consistency``   — summarise saved label-consistency checks
  ``export``        — copy a trained CNN to a visible folder for MycoMorph
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
         "Consumes MycoMorph's HDF5 crops (all_crops.h5).",
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


# ── Training defaults ────────────────────────────────────────────────────────
# Fine-tuning from a model uses the GUI's defaults (the Fig. S2 runs used these);
# training from scratch needs more epochs and a larger learning rate.
FINE_TUNE_DEFAULTS = {"epochs": 40, "lr": 1e-4}
SCRATCH_DEFAULTS = {"epochs": 80, "lr": 1e-3}

# CLI option name (without "--") for each cell_filter field.
_FILTER_OPTIONS = {
    "gene": "gene", "condition": "condition", "condition_label": "condition-label",
    "reporter": "reporter", "drug": "drug", "tiff_file": "tiff",
}


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
    values: bool = typer.Option(
        False, "--values",
        help="List every distinct gene / condition / … value with its cell "
             "count (the values `label --gene/--condition/…` accept).",
    ),
) -> None:
    """Describe an ``all_crops.h5`` file (cells, channels, schema, metadata)."""
    from .cell_filter import FILTER_FIELDS, distinct_values

    with HDF5CropsReader(h5) as reader:
        typer.echo(reader.summary())
        typer.echo(f"  phase channel: idx={reader.phase_channel_idx}  "
                   f"name={reader.channel_names[reader.phase_channel_idx]!r}")
        typer.echo(f"  mask channel:  idx={reader.mask_channel_idx}  "
                   f"name={reader.channel_names[reader.mask_channel_idx]!r}")
        typer.echo(f"  native crop:   {reader.native_crop_size}×{reader.native_crop_size}")
        typer.echo(f"  classifier projection: 3 channels × "
                   f"{reader.classifier_size}×{reader.classifier_size}")
        typer.echo("  metadata (filter with `label --<option> VALUE`):")
        for key, (dataset, _name) in FILTER_FIELDS.items():
            pairs = distinct_values(reader, key)
            option = "--" + _FILTER_OPTIONS[key]
            if not values:
                shown = ", ".join(repr(v) for v, _ in pairs[:6])
                more = f", … (+{len(pairs) - 6}; use --values)" if len(pairs) > 6 else ""
                typer.echo(f"    {option:18s} {dataset}: {len(pairs)} value(s): {shown}{more}")
                continue
            typer.echo(f"    {option:18s} {dataset}: {len(pairs)} value(s)")
            for v, n in pairs:
                typer.echo(f"        {v!r:48s} {n:>8,} cells")

    from .crop_provenance import crop_options, crop_warnings

    opts = crop_options(h5)
    mb = opts.get("mask_background")
    typer.echo(f"  mask_background: {'unknown' if mb is None else mb}"
               f"{'  (from ' + opts['source'] + ')' if 'source' in opts else ''}")
    for w in crop_warnings(h5, opts):
        typer.echo(f"  WARNING: {w}")


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

    Catches a broken hand-off before you point MycoMorph at the model: loads it
    exactly the way MycoMorph will, runs a synthetic batch, and checks the
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
        f"Ready to drop into MycoMorph's Custom-model picker."
    )
    cfg = model_path.parent / "training_config.json"
    if cfg.exists():
        try:
            c = json.loads(cfg.read_text(encoding="utf-8"))
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
                "pyarrow", "matplotlib", "typer", "qtawesome", "psutil"):
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
                meta = _json.loads(sidecar.read_text(encoding="utf-8"))
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
    gene: list[str] = typer.Option(
        [], "--gene", help="Only cells of this gene (repeatable; values are "
        "OR'd). Exact match — see `modelprep-cli info --values`.",
    ),
    condition: list[str] = typer.Option(
        [], "--condition", help="Only cells with this condition, e.g. "
        "'ATc+MMC-' (repeatable). Combined with other filters by AND.",
    ),
    condition_label: list[str] = typer.Option(
        [], "--condition-label", help="Only cells with this MycoMorph "
        "condition_label (repeatable).",
    ),
    reporter: list[str] = typer.Option(
        [], "--reporter", help="Only cells with this reporter (repeatable).",
    ),
    drug: list[str] = typer.Option(
        [], "--drug", help="Only cells with this drug (repeatable).",
    ),
    tiff: list[str] = typer.Option(
        [], "--tiff", help="Only cells from this per-well TIFF file name "
        "(repeatable).",
    ),
    seed: int = typer.Option(
        42, "--seed", help="Seed for the random order (used without a model). "
        "Same file + filter + seed → same sample.",
    ),
) -> None:
    """Launch the interactive labeller.

    Without a model, cells come in a seeded random order; with one, least
    certain first. --gene/--condition/… restrict the pool to matching cells
    (values OR'd within an option, AND'd across options). The filter, seed and
    ordering are recorded with the session in modelprep_labels/manifest.json.
    """
    from .cell_filter import CellFilter, distinct_values
    from .label_cells import run_grid_labeling

    model_path = None
    if model is not None:
        try:
            model_path = resolve_classifier_preset(model)
        except FileNotFoundError as e:
            typer.echo(f"ERROR: {e}", err=True)
            raise typer.Exit(code=2)

    cell_filter = CellFilter.from_mapping({
        "gene": gene, "condition": condition, "condition_label": condition_label,
        "reporter": reporter, "drug": drug, "tiff_file": tiff,
    })
    if not cell_filter.is_empty():
        # Catch typos before opening the labeller: every value must exist.
        with HDF5CropsReader(h5) as reader:
            unknown = cell_filter.unknown_values(reader)
            for key, missing in unknown.items():
                available = ", ".join(repr(v) for v, _ in distinct_values(reader, key))
                typer.echo(f"ERROR: no cells have --{_FILTER_OPTIONS[key]} "
                           f"{', '.join(map(repr, missing))}. Available: {available}",
                           err=True)
            if unknown:
                raise typer.Exit(code=2)

    run_grid_labeling(
        h5_path=h5,
        labels_dir=labels_dir,
        model_path=model_path,
        n_pool=n_pool,
        cell_filter=cell_filter,
        seed=seed,
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
        help="Where to write best_model.pth + training_config.json. Default: "
             "~/.modelprep/models/<name>_<date>/ (or trained_<ts>/ without --name).",
    ),
    name: Optional[str] = typer.Option(
        None, "--name",
        help="Model name, as on the GUI's Source page (names the run directory "
             "and is recorded in training_config.json).",
    ),
    epochs: Optional[int] = typer.Option(
        None, "--epochs", min=1,
        help="Default: 40 when fine-tuning (--pretrained, as in the GUI), else 80.",
    ),
    batch_size: int = typer.Option(32, "--batch-size", min=1),
    lr: Optional[float] = typer.Option(
        None, "--lr", min=1e-7,
        help="Default: 1e-4 when fine-tuning (--pretrained, as in the GUI), else 1e-3.",
    ),
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
    holdout: bool = typer.Option(
        True, "--holdout/--no-holdout",
        help="Report metrics on the frozen hold-out test set (the 'test' split in "
             "labels.csv) when one exists, as the GUI does. --no-holdout splits "
             "the training labels internally instead. Test cells are never trained on.",
    ),
) -> None:
    """Train the binary cell-quality CNN from labelled HDF5 crops."""
    import numpy as np

    from .cell_quality_classifier import CLASS_NAMES, NUM_CLASSES
    from .crop_provenance import crop_options
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

    defaults = FINE_TUNE_DEFAULTS if pretrained_path else SCRATCH_DEFAULTS
    epochs = epochs if epochs is not None else defaults["epochs"]
    lr = lr if lr is not None else defaults["lr"]

    if output is None:
        if name:
            from ..gui.trained_models import new_run_dir  # pure-Python helper
            output = new_run_dir(name)
        else:
            output = _default_output_dir()

    reader = HDF5CropsReader(h5_path)
    store = LabelsStore(labels_dir_path)
    try:
        crops, labels, used_uids = store.to_training_dataset(reader)
        if holdout:
            eval_crops, eval_labels, eval_uids = store.to_eval_dataset(reader)
        else:
            eval_crops, eval_labels, eval_uids = np.empty((0,)), np.empty((0,), np.int64), []
    except ValueError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=2)
    finally:
        reader.close()

    typer.echo(f"Training on {len(used_uids)} labelled cells "
               f"(crops shape={crops.shape}, labels shape={labels.shape})")
    typer.echo(f"  {'fine-tuning from ' + pretrained_path.name if pretrained_path else 'from scratch'}: "
               f"epochs={epochs} lr={lr:g}")
    if len(eval_labels):
        typer.echo(f"  evaluating on the frozen hold-out: {len(eval_labels)} cells")
    elif holdout:
        typer.echo("  no hold-out test set yet: metrics come from an internal split")
    typer.echo(f"Output dir: {output}")
    output.mkdir(parents=True, exist_ok=True)

    class_counts = {CLASS_NAMES[i]: int((labels == i).sum()) for i in range(NUM_CLASSES)}
    provenance = {
        "model_name": name or output.name,
        "source_h5": h5_path.name,
        "labels_dir": str(labels_dir_path),
        "n_labelled": int(len(used_uids)),
        "used_uids_sha256": cell_uids_sha256(used_uids),
        "class_counts": class_counts,
        "n_holdout": int(len(eval_uids)),
        "holdout_class_counts": {CLASS_NAMES[i]: int((eval_labels == i).sum())
                                 for i in range(NUM_CLASSES)},
        "labels_schema_version": LABELS_SCHEMA_VERSION,
        "crop_options": crop_options(h5_path),
        "entry_point": "modelprep-cli train",
    }

    summary = run_train(
        crops=crops,
        labels=labels,
        eval_crops=eval_crops if len(eval_labels) else None,
        eval_labels=eval_labels if len(eval_labels) else None,
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
                   if split in ("test", "holdout") else f"  {split}_accuracy:  {acc:.3f}")
        bal = summary.get("balanced_acc")
        if bal is not None and bal == bal:
            typer.echo(f"  balanced acc:    {bal:.3f}  (majority baseline "
                       f"{summary.get('majority_baseline', float('nan')):.3f})")
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
    from .train_classifier import single_class_warning
    single = single_class_warning(summary.get("eval_class_counts") or {},
                                  {"holdout": "hold-out"}.get(split, split))
    if single:
        typer.echo(f"  WARNING: {single}")
    typer.echo(f"  model:           {summary['model_path']}")
    typer.echo(f"  config:          {output / 'training_config.json'}")


@app.command(name="feature-model")
def feature_model(
    h5: Path = typer.Option(..., "--h5", "-i", exists=True),
    labels_dir: Optional[Path] = typer.Option(None, "--labels-dir"),
    no_fragment: bool = typer.Option(
        False, "--no-fragment",
        help="Skip the crossing-fragment cues (use only the morphometric table).",
    ),
    no_context: bool = typer.Option(
        False, "--no-context",
        help="Skip the FOV-context features (neighbour density etc.).",
    ),
    no_focus: bool = typer.Option(
        False, "--no-focus",
        help="Skip the crop-based focus/sharpness features.",
    ),
    phase_only: bool = typer.Option(
        False, "--phase-only",
        help="Drop reporter/fluorescence intensity columns, keeping only "
             "phase-derived information (what the CNN sees).",
    ),
) -> None:
    """Train a feature-based classifier (morphometrics + fragment + FOV-context
    + focus cues) and report hold-out metrics — comparable to the CNN."""
    from .feature_classifier import load_feature_matrix, train_feature_classifier

    ld = labels_dir or default_labels_dir_for(h5)
    try:
        matrix = load_feature_matrix(h5, ld, with_fragment=not no_fragment,
                                     with_context=not no_context,
                                     with_focus=not no_focus,
                                     phase_only=phase_only)
        summary = train_feature_classifier(matrix)
    except (FileNotFoundError, ValueError, ImportError) as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=2)

    typer.echo("\nFeature-based model — held-out metrics")
    typer.echo("=" * 44)
    extras = [tag for tag, on in (("fragment", summary["with_fragment"]),
                                  ("context", summary["with_context"]),
                                  ("focus", summary["with_focus"])) if on]
    typer.echo(f"  train / hold-out:  {summary['n_train']} / {summary['n_test']} cells · "
               f"{summary['n_features']} features"
               f"{' (+ ' + ' + '.join(extras) + ' cues)' if extras else ''}")
    if summary["phase_only"]:
        dropped = summary["dropped_reporter_features"]
        typer.echo(f"  phase only:        dropped {len(dropped)} reporter columns"
                   f"{': ' + ', '.join(dropped) if dropped else ''}")
    typer.echo(f"  balanced accuracy: {summary['balanced_acc']:.3f}")
    typer.echo(f"  accuracy:          {summary['eval_acc']:.3f}")
    typer.echo(f"  ROC AUC:           {summary['roc_auc']:.3f}    AP: {summary['average_precision']:.3f}")
    c = summary["confusion"]
    typer.echo(f"  confusion:         tn={c['tn']} fp={c['fp']} fn={c['fn']} tp={c['tp']}")
    typer.echo("\n  top features (permutation importance on hold-out):")
    for name, mean, std in summary["top_importances"]:
        typer.echo(f"    {name:28s} {mean:+.4f} ± {std:.4f}")


def _resolve_model_arg(model: str) -> Path:
    """A preset name, a .pth/.joblib file, or a run directory."""
    p = Path(model).expanduser()
    if p.exists():
        return p
    return resolve_classifier_preset(model)


@app.command()
def evaluate(
    h5: Path = typer.Option(..., "--h5", "-i", exists=True),
    model: str = typer.Option(
        ..., "--model", "-m",
        help="A run directory, best_model.pth, feature_model.joblib or preset name.",
    ),
    labels_dir: Optional[Path] = typer.Option(None, "--labels-dir"),
    json_out: Optional[Path] = typer.Option(
        None, "--json", help="Also write the result as JSON to this file."),
    device: Optional[str] = typer.Option("cpu", "--device"),
) -> None:
    """Score a saved CNN or feature model on the CURRENT hold-out test set.

    Use it to re-score older runs after the hold-out changed, or to compare runs
    on one yardstick. holdout_sha256 identifies the exact (cell, label) set, so
    only compare numbers that share it.
    """
    import json as _json

    from .evaluate import evaluate_model

    ld = labels_dir or default_labels_dir_for(h5)
    try:
        res = evaluate_model(h5, ld, _resolve_model_arg(model), device=device)
    except (FileNotFoundError, ValueError, ImportError) as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=2)

    cc = res["eval_class_counts"]
    typer.echo(f"\n{res['model_type'].upper()} model on the current hold-out")
    typer.echo("=" * 44)
    typer.echo(f"  model:             {res['model_path']}")
    typer.echo(f"  hold-out:          {res['n']} cells (good {cc.get('good', 0)} / "
               f"bad {cc.get('bad', 0)})  sha256 {res['holdout_sha256'][:12]}")
    if res.get("trained_on_holdout_n") not in (None, res["n"]):
        typer.echo(f"  NOTE: the model reported metrics on a {res['trained_on_holdout_n']}-cell "
                   "hold-out when trained; this is a different set.")
    typer.echo(f"  accuracy:          {res['eval_acc']:.3f}   balanced: {res['balanced_acc']:.3f}"
               f"   (majority baseline {res['majority_baseline']:.3f})")
    typer.echo(f"  ROC AUC:           {res['roc_auc']:.3f}   AP: {res['average_precision']:.3f}")
    c = res["confusion"]
    typer.echo(f"  confusion:         tn={c['tn']} fp={c['fp']} fn={c['fn']} tp={c['tp']}")
    if res.get("warning"):
        typer.echo(f"  WARNING: {res['warning']}")
    if json_out:
        json_out.write_text(_json.dumps(res, indent=2))
        typer.echo(f"  wrote {json_out}")


@app.command(name="learning-curve")
def learning_curve(
    h5: Path = typer.Option(..., "--h5", "-i", exists=True),
    labels_dir: Optional[Path] = typer.Option(None, "--labels-dir"),
    points: int = typer.Option(5, "--points", min=2, max=12,
                               help="Number of training-set sizes."),
    epochs: int = typer.Option(25, "--epochs", min=1, help="Epochs per point."),
    seed: int = typer.Option(42, "--seed"),
    device: Optional[str] = typer.Option(None, "--device"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Directory for curve.json + curve.csv. "
             "Default: ~/.modelprep/learning_curves/<h5 stem>_<date>/",
    ),
) -> None:
    """Train from scratch on growing, class-stratified subsets of the train
    labels and score each on the frozen hold-out (same as the Models tab)."""
    import csv
    import json as _json

    from .learning_curve import compute_learning_curve

    ld = labels_dir or default_labels_dir_for(h5)
    store = LabelsStore(ld)
    with HDF5CropsReader(h5) as reader:
        try:
            tr_crops, tr_labels, _ = store.to_training_dataset(reader)
        except ValueError as e:
            typer.echo(f"ERROR: {e}", err=True)
            raise typer.Exit(code=2)
        ev_crops, ev_labels, _ = store.to_eval_dataset(reader)
    if len(set(ev_labels.tolist())) < 2:
        typer.echo("ERROR: the learning curve needs a hold-out test set with both "
                   "good and bad cells.", err=True)
        raise typer.Exit(code=2)

    pts = compute_learning_curve(
        tr_crops, tr_labels, ev_crops, ev_labels, n_points=points, epochs=epochs,
        seed=seed, device=device,
        progress_cb=lambda f, m: typer.echo(f"  [{f:4.0%}] {m}"),
    )
    if output is None:
        from ..gui.trained_models import human_timestamp
        output = Path.home() / ".modelprep" / "learning_curves" / f"{h5.stem}_{human_timestamp()}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "curve.json").write_text(_json.dumps(
        {"h5": str(h5), "labels_dir": str(ld), "epochs": epochs, "seed": seed,
         "points": pts}, indent=2))
    keys = ["n", "n_good", "n_bad", "roc_auc", "average_precision", "balanced_acc", "eval_acc"]
    with open(output / "curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows({k: p.get(k) for k in keys} for p in pts)

    typer.echo(f"\n  {'n':>6}  {'AUC':>6}  {'AP':>6}  {'bal.acc':>7}")
    for p in pts:
        typer.echo(f"  {p['n']:>6}  {p['roc_auc']:>6.3f}  {p['average_precision']:>6.3f}"
                   f"  {p['balanced_acc']:>7.3f}")
    typer.echo(f"  wrote {output}")


@app.command()
def consistency(
    labels_dir: Optional[Path] = typer.Option(
        None, "--labels-dir", help="Labels directory (or give --h5)."),
    h5: Optional[Path] = typer.Option(None, "--h5", "-i"),
) -> None:
    """Summarise the saved blind label-consistency checks (run them from the
    GUI: Label cells > More > Consistency check)."""
    import json as _json

    if labels_dir is None and h5 is None:
        typer.echo("ERROR: pass --labels-dir or --h5.", err=True)
        raise typer.Exit(code=2)
    ld = labels_dir or default_labels_dir_for(h5)
    runs = sorted((ld / "consistency").glob("*/summary.json"))
    if not runs:
        typer.echo(f"No consistency checks saved under {ld / 'consistency'}.")
        return
    typer.echo(f"{'run':20s} {'mode':9s} {'n':>5s} {'agree':>6s} {'kappa':>6s}  band")
    for f in runs:
        r = _json.loads(f.read_text())
        typer.echo(f"{f.parent.name:20s} {str(r.get('mode', '')):9s} {r.get('n', 0):>5} "
                   f"{r.get('raw_agreement', float('nan')):>6.1%} "
                   f"{r.get('cohen_kappa', float('nan')):>6.2f}  {r.get('kappa_band', '')}")
        flips = r.get("flips") or {}
        if flips:
            typer.echo("    flips: " + ", ".join(
                f"{a}->{b} {n}" for a, row in flips.items() for b, n in row.items() if a != b))


@app.command()
def export(
    model: str = typer.Option(..., "--model", "-m",
                              help="A CNN run directory or its best_model.pth."),
    dest: Path = typer.Option(..., "--dest", "-d",
                              help="Folder to export into (a subfolder is created)."),
    name: Optional[str] = typer.Option(None, "--name",
                                       help="Default: the run directory's name."),
) -> None:
    """Copy a trained CNN to a visible folder for MycoMorph's Custom-model picker."""
    from .export import export_for_mycomorph

    try:
        pth = export_for_mycomorph(Path(model).expanduser(), dest, name=name)
    except (FileNotFoundError, ValueError, OSError) as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=2)
    typer.echo(f"OK: {pth}\nIn MycoMorph: Classify > Custom model > choose {pth.name}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
