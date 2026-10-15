#!/usr/bin/env python3
"""
Active-learning review tool — ModelPrep
=======================================
Review and correct CNN classifier predictions on the cells stored in
``all_crops.h5``.

Unlike ``label_cells`` (label from scratch), this tool:
  1. Runs the supplied model on every (unlabeled) cell in the HDF5
  2. Sorts cells by confidence ascending — the most uncertain predictions
     surface first
  3. For each cell, shows the model's prediction; user can accept, flip,
     force good/bad, or revert
  4. Labels are appended to the same ``modelprep_labels/labels.csv`` the
     standard labeller writes

Keyboard shortcuts:
    a = accept model prediction
    f = flip label (good ↔ bad)
    g = force good
    b = force bad
    u = undo (revert to model prediction)
    A = accept all remaining on current page
    n / → = next page          p / ← = prev page
    s = save              q = quit + save
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .cell_quality_classifier import CLASS_NAMES, CROP_SIZE, CellQualityClassifier
from .crops_h5 import CropRecord, HDF5CropsReader
from .label_cells import _make_contour_thumb
from .labels_store import (
    LABEL_TO_IDX,
    LabelRow,
    LabelsStore,
    default_labels_dir_for,
    make_session_id,
)


GRID_COLS = 8
GRID_ROWS = 6
PAGE_SIZE = GRID_COLS * GRID_ROWS

COLOR_GOOD = "#33cc33"
COLOR_BAD = "#dd2222"
COLOR_UNREVIEWED = "#555555"
COLOR_SELECTED = "#ffff00"
COLOR_CORRECTED = "#ff9900"
PRED_COLORS = {0: COLOR_GOOD, 1: COLOR_BAD}
SELECTED_LW = 4
NORMAL_LW = 2
CORRECTED_LW = 3


def _score_cells(
    reader: HDF5CropsReader,
    model_path: Path,
    *,
    only_unlabeled: bool,
    already_labeled: set[str],
    progress_cb=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the model over every (eligible) cell.

    Returns ``(indices, pred_idx, confidence)`` where ``indices`` is the
    set of HDF5 rows scored, ``pred_idx`` is the predicted class for each,
    and ``confidence`` is the max class probability.
    """
    if only_unlabeled and already_labeled:
        unseen_mask = np.array(
            [uid not in already_labeled for uid in reader.cell_uids],
            dtype=bool,
        )
        indices = np.flatnonzero(unseen_mask)
    else:
        indices = np.arange(len(reader))

    if indices.size == 0:
        return indices, np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

    clf = CellQualityClassifier(str(model_path), in_channels=3)
    preds = np.empty(len(indices), dtype=np.int32)
    confs = np.empty(len(indices), dtype=np.float32)
    batch_size = 256
    for lo in range(0, len(indices), batch_size):
        hi = min(lo + batch_size, len(indices))
        batch = reader.get_classifier_batch(indices[lo:hi].tolist())
        class_idx, probs = clf.predict_batch(batch)
        preds[lo:hi] = class_idx
        confs[lo:hi] = probs.max(axis=1)
        if progress_cb is not None:
            progress_cb(hi / len(indices), f"Scoring cells: {hi}/{len(indices)}")
    return indices, preds, confs


class _ReviewCell:
    __slots__ = (
        "record", "pred_idx", "confidence",
        "final_label", "reviewed", "corrected",
    )

    def __init__(self, record: CropRecord, pred_idx: int, confidence: float) -> None:
        self.record = record
        self.pred_idx = int(pred_idx)
        self.confidence = float(confidence)
        self.final_label = int(pred_idx)
        self.reviewed = False
        self.corrected = False


def run_review_grid(
    h5_path: Path,
    model_path: Path,
    labels_dir: Optional[Path] = None,
    *,
    mode: str = "uncertain",
    confidence_threshold: float = 0.85,
    only_unlabeled: bool = True,
    pixels_per_um: float = 13.8767,
) -> None:
    """Launch the review UI against an HDF5 crops file + a trained model."""
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    h5_path = Path(h5_path)
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}")
        sys.exit(1)

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
    already_labeled = store.labeled_uids()

    print(f"Scoring {len(reader):,} cells with {model_path.name}...")
    indices, preds, confs = _score_cells(
        reader, model_path,
        only_unlabeled=only_unlabeled,
        already_labeled=already_labeled,
        progress_cb=lambda f, m: print(f"  {m}"),
    )
    if indices.size == 0:
        print("No cells to review (every cell is already labeled).")
        reader.close()
        return

    if mode == "uncertain":
        keep = confs < confidence_threshold
        n_before = int(indices.size)
        indices = indices[keep]
        preds = preds[keep]
        confs = confs[keep]
        print(f"  Uncertain mode: {indices.size}/{n_before} cells "
              f"have confidence < {confidence_threshold}")

    if indices.size == 0:
        print("No uncertain cells. Try --mode all or raise --confidence-threshold.")
        reader.close()
        return

    # Sort by confidence ascending — worst predictions first.
    order = np.argsort(confs)
    indices = indices[order]
    preds = preds[order]
    confs = confs[order]

    cells: list[_ReviewCell] = []
    for idx, p, c in zip(indices.tolist(), preds.tolist(), confs.tolist()):
        cells.append(_ReviewCell(reader.get(int(idx)), p, c))

    n_total = len(cells)
    n_pages = math.ceil(n_total / PAGE_SIZE)

    session_id = make_session_id()
    session_started_at = datetime.now(timezone.utc).isoformat()

    # ── UI ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        GRID_ROWS, GRID_COLS,
        figsize=(GRID_COLS * 1.4, GRID_ROWS * 1.4 + 0.8),
        gridspec_kw={"wspace": 0.05, "hspace": 0.35},
    )
    axes_flat = axes.flatten()
    img_handles = []
    border_patches = []
    for ax in axes_flat:
        ax.axis("off")
        im = ax.imshow(np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.float32),
                       interpolation="nearest")
        img_handles.append(im)
        rect = mpatches.FancyBboxPatch(
            (0, 0), 1, 1,
            transform=ax.transAxes,
            boxstyle="square,pad=0",
            linewidth=NORMAL_LW,
            edgecolor=COLOR_UNREVIEWED,
            facecolor="none",
            clip_on=False,
        )
        ax.add_patch(rect)
        border_patches.append(rect)

    status_text = fig.text(
        0.5, 0.01, "",
        ha="center", va="bottom", fontsize=8,
        fontfamily="monospace",
    )
    fig.patch.set_facecolor("#1a1a1a")

    state = {"page": 0, "selected": None}

    def render_page():
        page = state["page"]
        sel_pos = state["selected"]
        start = page * PAGE_SIZE
        for pos, (ax, im, rect) in enumerate(zip(axes_flat, img_handles, border_patches)):
            idx = start + pos
            if idx < n_total:
                cell = cells[idx]
                thumb = _make_contour_thumb(cell.record.crop, phase_idx=0)
                im.set_data(thumb)
                ax.set_visible(True)
                if pos == sel_pos:
                    rect.set_edgecolor(COLOR_SELECTED); rect.set_linewidth(SELECTED_LW)
                elif cell.corrected:
                    rect.set_edgecolor(COLOR_CORRECTED); rect.set_linewidth(CORRECTED_LW)
                else:
                    rect.set_edgecolor(PRED_COLORS.get(cell.final_label, COLOR_UNREVIEWED))
                    rect.set_linewidth(NORMAL_LW)
                lbl_char = CLASS_NAMES[cell.final_label][0].upper()
                area_um2 = cell.record.area_px / (pixels_per_um ** 2)
                size_str = f" {area_um2:.1f}µm²"
                mark = "✎" if cell.corrected else ("✓" if cell.reviewed else "")
                ax.set_title(
                    f"{lbl_char} {cell.confidence:.2f}{size_str} {mark}",
                    fontsize=7,
                    color=COLOR_CORRECTED if cell.corrected else PRED_COLORS.get(cell.final_label, "#cccccc"),
                    pad=1,
                )
            else:
                im.set_data(np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.float32))
                rect.set_edgecolor("#222222"); rect.set_linewidth(NORMAL_LW)
                ax.set_title("", pad=1)
                ax.set_visible(True)

        n_reviewed = sum(1 for c in cells if c.reviewed)
        n_corrected = sum(1 for c in cells if c.corrected)
        page_end = min(start + PAGE_SIZE, n_total)
        page_cells = cells[start:page_end]
        if page_cells:
            conf_str = (
                f"conf {min(c.confidence for c in page_cells):.2f}–"
                f"{max(c.confidence for c in page_cells):.2f}"
            )
        else:
            conf_str = ""
        status_text.set_text(
            f"Page {page+1}/{n_pages}  |  Cells {start+1}–{page_end} of {n_total}"
            f"  |  Reviewed: {n_reviewed}  Corrected: {n_corrected}"
            f"  |  {conf_str}  |  a=accept f=flip g/b u=undo A=accept-page s=save q=quit"
        )
        status_text.set_color("#cccccc")
        fig.canvas.draw_idle()

    def advance_from(pos: int) -> int:
        page = state["page"]
        for next_pos in range(pos + 1, PAGE_SIZE):
            next_idx = page * PAGE_SIZE + next_pos
            if next_idx >= n_total:
                break
            if not cells[next_idx].reviewed:
                return next_pos
        return pos

    def persist() -> int:
        rows: list[LabelRow] = []
        for cell in cells:
            if not cell.reviewed:
                continue
            rows.append(LabelRow(
                cell_uid=cell.record.cell_uid,
                label_idx=int(cell.final_label),
                label_name=CLASS_NAMES[cell.final_label],
                area_px=cell.record.area_px,
                labeled_at=datetime.now(timezone.utc).isoformat(),
                session_id=session_id,
                source_h5=h5_path.name,
                model_at_label=str(model_path),
                confidence_good_at_label=(
                    cell.confidence if cell.pred_idx == 0 else 1.0 - cell.confidence
                ),
            ))
        if not rows:
            return 0
        store.append_batch(rows)
        # Clear the reviewed flag so we don't re-emit the same row on next save.
        for c in cells:
            c.reviewed = False
        return len(rows)

    def on_key(event):
        page = state["page"]
        sel_pos = state["selected"]
        key = event.key
        if key in ("right", "n") and page + 1 < n_pages:
            state["page"] += 1; state["selected"] = None
            render_page(); return
        if key in ("left", "p") and page > 0:
            state["page"] -= 1; state["selected"] = None
            render_page(); return
        if key == "s":
            n = persist(); print(f"  Saved {n} labels."); render_page(); return
        if key == "q":
            n = persist()
            session_ended_at = datetime.now(timezone.utc).isoformat()
            store.record_session(session_id, session_started_at, session_ended_at, n)
            print(f"  Saved {n} labels and exiting.")
            plt.close(fig)
            return
        if key == "A":
            start = page * PAGE_SIZE
            count = 0
            for pos in range(PAGE_SIZE):
                idx = start + pos
                if idx >= n_total:
                    break
                if not cells[idx].reviewed:
                    cells[idx].reviewed = True
                    count += 1
            print(f"  Accepted {count} cells on page {page+1}")
            render_page(); return

        if sel_pos is None:
            return
        idx = page * PAGE_SIZE + sel_pos
        if idx >= n_total:
            return
        cell = cells[idx]

        if key == "a":
            cell.reviewed = True; cell.final_label = cell.pred_idx; cell.corrected = False
            state["selected"] = advance_from(sel_pos); render_page(); return
        if key == "f":
            cell.reviewed = True; cell.final_label = 1 - cell.final_label
            cell.corrected = (cell.final_label != cell.pred_idx)
            state["selected"] = advance_from(sel_pos); render_page(); return
        if key == "g":
            cell.reviewed = True; cell.final_label = 0
            cell.corrected = (cell.pred_idx != 0)
            state["selected"] = advance_from(sel_pos); render_page(); return
        if key == "b":
            cell.reviewed = True; cell.final_label = 1
            cell.corrected = (cell.pred_idx != 1)
            state["selected"] = advance_from(sel_pos); render_page(); return
        if key == "u":
            cell.reviewed = False; cell.final_label = cell.pred_idx; cell.corrected = False
            render_page(); return

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

    try:
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    except AttributeError:
        pass
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)

    render_page()

    print("\n=== ModelPrep review tool ===")
    print("Cells sorted by confidence (lowest first).")
    print("Click a cell, then: a=accept  f=flip  g=good  b=bad  u=undo")
    print("  A=accept-page  n/→=next  p/←=prev  s=save  q=quit+save\n")

    plt.show(block=True)
    n = persist()
    if n:
        print(f"Saved {n} pending reviews on close.")
    reader.close()
