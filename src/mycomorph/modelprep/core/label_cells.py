#!/usr/bin/env python3
"""
Interactive cell-quality labeller — ModelPrep
=============================================
Matplotlib grid labeller that reads pre-extracted crops from MycoPrep's
``all_crops.h5`` and persists labels to ``modelprep_labels/labels.csv``
alongside the HDF5.

Workflow:
    1. Open ``all_crops.h5``
    2. (Optional) score every unlabeled cell with a trained classifier;
       rank by uncertainty (|0.5 - P(good)| ascending). Without a model,
       cells are shown in random order.
    3. Show a grid of phase + mask-outline thumbnails. Click to select,
       press a key:
           g = good
           b = bad
           u = undo (clear)
           s = save
           q = quit + save
           n / → = next page    p / ← = prev page

This module is also imported from the GUI's ``label_train_panel.py`` to share
the thumbnail rendering. See ``_make_contour_thumb``.
"""

from __future__ import annotations

import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .cell_quality_classifier import CLASS_NAMES, CROP_SIZE
from .crops_h5 import CropRecord, HDF5CropsReader
from .labels_store import (
    LABEL_TO_IDX,
    LabelRow,
    LabelsStore,
    default_labels_dir_for,
    make_session_id,
)


# ── Visual constants ──────────────────────────────────────────────────────────

CLASS_COLORS = {
    None: "#555555",
    0:    "#33cc33",
    1:    "#dd2222",
}
SELECTED_COLOR = "#ffff00"
SELECTED_LW = 4
NORMAL_LW = 2

GRID_COLS = 8
GRID_ROWS = 6
PAGE_SIZE = GRID_COLS * GRID_ROWS


# ─────────────────────────────────────────────────────────────────────────────
# Thumbnail rendering — shared with the Qt panel
# ─────────────────────────────────────────────────────────────────────────────

def _make_contour_thumb(crop, phase_idx=0, contour_color=(0.0, 1.0, 0.4)):
    """
    Build an RGB thumbnail with the mask contour overlaid on the phase image.

    Args:
        crop:          (C, H, W) float32 — assumed (phase, mask, [area])
        phase_idx:     index of the phase channel within ``crop``
        contour_color: RGB tuple [0-1] for the contour line

    Returns:
        thumb_rgb: (H, W, 3) float32 array suitable for imshow
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    phase = crop[min(phase_idx, crop.shape[0] - 2)]
    # In the (phase, mask, area) layout the mask is at index 1, not the last
    # channel. Pick the channel whose unique values are ~{0, 1} or default to 1.
    if crop.shape[0] >= 2:
        mask = crop[1] > 0.5
    else:
        mask = np.zeros_like(phase, dtype=bool)

    dilated = binary_dilation(mask, iterations=1)
    eroded = binary_erosion(mask, iterations=1)
    contour = dilated & ~eroded

    rgb = np.stack([phase, phase, phase], axis=-1)
    for ch_i in range(3):
        rgb[contour, ch_i] = contour_color[ch_i]

    return np.clip(rgb, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Pool building — what cells to show, in what order
# ─────────────────────────────────────────────────────────────────────────────

def build_pool(
    reader: HDF5CropsReader,
    *,
    already_labeled: set[str],
    n_target: int,
    model_path: Optional[Path] = None,
    progress_cb=None,
    seed: int = 42,
) -> tuple[list[int], np.ndarray]:
    """Return ``(pool_indices, p_good_scores)`` — the indices in the HDF5 to
    show next, in the order to show them.

    Without a ``model_path``, cells are shuffled. With one, every unlabeled
    cell is scored and the top-``n_target`` most uncertain are picked.
    """
    unseen_mask = np.array(
        [uid not in already_labeled for uid in reader.cell_uids],
        dtype=bool,
    )
    unseen_idx = np.flatnonzero(unseen_mask)
    if unseen_idx.size == 0:
        return [], np.zeros(0, dtype=np.float32)

    if model_path is None:
        rng = np.random.default_rng(seed)
        rng.shuffle(unseen_idx)
        picked = unseen_idx[:n_target]
        return picked.tolist(), np.full(len(picked), 0.5, dtype=np.float32)

    from .cell_quality_classifier import CellQualityClassifier

    clf = CellQualityClassifier(str(model_path), in_channels=3)
    scores = np.empty(len(unseen_idx), dtype=np.float32)
    batch_size = 256
    for lo in range(0, len(unseen_idx), batch_size):
        hi = min(lo + batch_size, len(unseen_idx))
        batch = reader.get_classifier_batch(unseen_idx[lo:hi].tolist())
        _, probs = clf.predict_batch(batch)
        scores[lo:hi] = probs[:, 0]
        if progress_cb is not None:
            progress_cb(hi / len(unseen_idx),
                        f"Scoring cells: {hi}/{len(unseen_idx)}")

    # |0.5 - P(good)| → most uncertain first
    rank = np.argsort(np.abs(0.5 - scores))
    picked = unseen_idx[rank][:n_target]
    scores_picked = scores[rank][:n_target]
    return picked.tolist(), scores_picked


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib grid labeller
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_labeling(
    h5_path: Path,
    labels_dir: Optional[Path] = None,
    *,
    model_path: Optional[Path] = None,
    n_pool: int = 1000,
    pixels_per_um: float = 13.8767,
) -> None:
    """Launch the matplotlib grid labeller against an HDF5 crops file.

    The user labels cells one page at a time; each save appends to
    ``labels.csv``. Pressing ``s`` saves without quitting; ``q`` saves and
    closes.
    """
    import matplotlib
    # Don't force a specific backend — let matplotlib pick whatever Qt
    # binding the environment provides (QtAgg works with PyQt5/PyQt6).
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    h5_path = Path(h5_path)
    if labels_dir is None:
        labels_dir = default_labels_dir_for(h5_path)
    else:
        labels_dir = Path(labels_dir)

    reader = HDF5CropsReader(h5_path)
    store = LabelsStore(labels_dir)
    store.ensure_initialised(
        source_h5=h5_path,
        total_cells=len(reader),
        h5_schema_version=reader.schema_version,
        cell_uids=reader.cell_uids,
    )

    # Existing labels (so we don't re-show)
    already_labeled = store.labeled_uids()
    print(f"Loaded {len(reader)} cells from {h5_path.name}")
    print(f"  {len(already_labeled)} already labeled  →  building pool of "
          f"up to {n_pool} unseen cells...")

    pool_indices, scores = build_pool(
        reader,
        already_labeled=already_labeled,
        n_target=n_pool,
        model_path=model_path,
        progress_cb=lambda f, m: print(f"    {m}"),
    )
    if not pool_indices:
        print("No unseen cells left to label. Nothing to do.")
        reader.close()
        return

    print(f"  pool ready: {len(pool_indices)} cells")
    n_total = len(pool_indices)
    n_pages = math.ceil(n_total / PAGE_SIZE)

    # Pre-fetch records for each pool index so paging is snappy.
    # Records hold the full HDF5 row + classifier-ready 64×64 crop.
    records: list[CropRecord] = [reader.get(int(i)) for i in pool_indices]

    # In-memory label state for the pool (idx → "good" | "bad" | None)
    pool_labels: dict[int, Optional[str]] = {i: None for i in range(n_total)}

    session_id = make_session_id()
    session_started_at = datetime.now(timezone.utc).isoformat()

    # ── Figure setup ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        GRID_ROWS, GRID_COLS,
        figsize=(GRID_COLS * 1.4, GRID_ROWS * 1.4 + 0.6),
        gridspec_kw={"wspace": 0.05, "hspace": 0.25},
    )
    axes_flat = axes.flatten()

    img_handles = []
    border_patches = []
    for ax in axes_flat:
        ax.axis("off")
        im = ax.imshow(
            np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.float32),
            interpolation="nearest",
        )
        img_handles.append(im)
        rect = mpatches.FancyBboxPatch(
            (0, 0), 1, 1,
            transform=ax.transAxes,
            boxstyle="square,pad=0",
            linewidth=NORMAL_LW,
            edgecolor=CLASS_COLORS[None],
            facecolor="none",
            clip_on=False,
        )
        ax.add_patch(rect)
        border_patches.append(rect)

    status_text = fig.text(
        0.5, 0.01, "",
        ha="center", va="bottom", fontsize=9,
        fontfamily="monospace",
    )
    fig.patch.set_facecolor("#1a1a1a")

    state = {"page": 0, "selected": None}

    def label_color(name):
        return CLASS_COLORS[LABEL_TO_IDX.get(name) if name is not None else None]

    def render_page():
        page = state["page"]
        sel_pos = state["selected"]
        start = page * PAGE_SIZE
        for pos, (ax, im, rect) in enumerate(zip(axes_flat, img_handles, border_patches)):
            idx = start + pos
            if idx < n_total:
                rec = records[idx]
                thumb = _make_contour_thumb(rec.crop, phase_idx=0)
                im.set_data(thumb)
                ax.set_visible(True)
                lbl = pool_labels[idx]
                if pos == sel_pos:
                    rect.set_edgecolor(SELECTED_COLOR)
                    rect.set_linewidth(SELECTED_LW)
                else:
                    rect.set_edgecolor(label_color(lbl))
                    rect.set_linewidth(NORMAL_LW)
                lbl_char = lbl[0].upper() if lbl else "·"
                lbl_color = label_color(lbl)
                area_um2 = rec.area_px / (pixels_per_um ** 2)
                size_str = f"{area_um2:.1f}µm²"
                score_str = ""
                if scores.size > idx and not math.isnan(float(scores[idx])):
                    score_str = f"  p={scores[idx]:.2f}"
                title = f"{lbl_char}  {size_str}{score_str}"
                ax.set_title(title, fontsize=8, color=lbl_color, pad=1)
            else:
                im.set_data(np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.float32))
                rect.set_edgecolor("#222222")
                ax.set_title("", pad=1)
                ax.set_visible(True)

        n_labeled = sum(1 for v in pool_labels.values() if v is not None)
        page_end = min(start + PAGE_SIZE, n_total)
        status_text.set_text(
            f"Page {page+1}/{n_pages}  |  Cells {start+1}–{page_end} of {n_total}"
            f"  |  Labeled: {n_labeled}/{n_total}"
            f"  |  g=good  b=bad  u=undo  s=save  q=quit"
        )
        status_text.set_color("#cccccc")
        fig.canvas.draw_idle()

    KEY_TO_LABEL = {"g": "good", "b": "bad"}

    def persist_labels() -> int:
        """Flush all currently-labeled pool entries to the CSV."""
        rows: list[LabelRow] = []
        used_indices: list[int] = []
        for idx, lbl in pool_labels.items():
            if lbl in ("good", "bad"):
                rec = records[idx]
                conf = float(scores[idx]) if (scores.size > idx and not math.isnan(float(scores[idx]))) else float("nan")
                rows.append(LabelRow(
                    cell_uid=rec.cell_uid,
                    label_idx=LABEL_TO_IDX[lbl],
                    label_name=lbl,
                    area_px=rec.area_px,
                    labeled_at=datetime.now(timezone.utc).isoformat(),
                    session_id=session_id,
                    source_h5=h5_path.name,
                    model_at_label=str(model_path) if model_path else "none",
                    confidence_good_at_label=conf,
                ))
                used_indices.append(idx)
        if not rows:
            return 0
        store.append_batch(rows)
        # Drop labels we just persisted so the next save doesn't duplicate.
        for idx in used_indices:
            pool_labels[idx] = None
        return len(rows)

    def on_key(event):
        page = state["page"]
        sel_pos = state["selected"]
        key = event.key

        if key in ("right", "n") and page + 1 < n_pages:
            state["page"] += 1; state["selected"] = None
            render_page()
            return
        if key in ("left", "p") and page > 0:
            state["page"] -= 1; state["selected"] = None
            render_page()
            return

        if key == "s":
            n = persist_labels()
            print(f"  Saved {n} labels.")
            render_page()
            return
        if key == "q":
            n = persist_labels()
            session_ended_at = datetime.now(timezone.utc).isoformat()
            store.record_session(session_id, session_started_at, session_ended_at, n)
            print(f"  Saved {n} labels and exiting.")
            plt.close(fig)
            return

        if sel_pos is None:
            return
        idx = page * PAGE_SIZE + sel_pos
        if idx >= n_total:
            return

        if key == "u":
            pool_labels[idx] = None
            render_page()
            return

        if key in KEY_TO_LABEL:
            pool_labels[idx] = KEY_TO_LABEL[key]
            # Auto-advance to next unlabeled cell on this page.
            for next_pos in range(sel_pos + 1, PAGE_SIZE):
                next_idx = page * PAGE_SIZE + next_pos
                if next_idx >= n_total:
                    break
                if pool_labels[next_idx] is None:
                    state["selected"] = next_pos
                    break
            else:
                state["selected"] = sel_pos
            render_page()

    def on_click(event):
        if event.inaxes is None:
            return
        for pos, ax in enumerate(axes_flat):
            if ax == event.inaxes:
                idx = state["page"] * PAGE_SIZE + pos
                if idx < n_total:
                    state["selected"] = pos
                    render_page()
                return

    # Drop matplotlib's arrow-key handlers so we can use them for pagination.
    try:
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    except AttributeError:
        pass

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)

    render_page()

    print("\n=== ModelPrep grid labeller ===")
    print("Click a cell (yellow border), then press a key:")
    print("  g=good  b=bad  u=undo  s=save  q=quit+save")
    print("  n/→=next page  p/←=prev page\n")

    plt.show(block=True)

    # Persist any unsaved labels on close.
    n = persist_labels()
    if n:
        print(f"Saved {n} pending labels on close.")
    counts = store.class_counts()
    total = sum(counts.values())
    print(f"Total labels on disk: {total} (good={counts['good']}, bad={counts['bad']})")
    print(f"  labels.csv: {store.csv_path}")
    reader.close()
