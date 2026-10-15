#!/usr/bin/env python3
"""
Interactive cell-quality labeller — ModelPrep
=============================================
Matplotlib grid labeller that reads pre-extracted crops from MycoMorph's
``all_crops.h5`` and persists labels to ``modelprep_labels/labels.csv``
alongside the HDF5.

Workflow:
    1. Open ``all_crops.h5``
    2. (Optional) restrict to cells of chosen genes / conditions / wells
       (``cell_filter``). Then score every unlabeled candidate with a trained
       classifier and rank by uncertainty (|0.5 - P(good)| ascending), or,
       without a model, show them in a seeded random order. See ``build_pool``.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .cell_filter import CellFilter
from .cell_quality_classifier import CROP_SIZE
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

# How the next cells are chosen. "uncertain" falls back to "random" without a
# model; "good" (highest P(good) first) is for filling a balanced hold-out.
SELECT_MODES = ("uncertain", "good", "random")


class PoolCancelled(Exception):
    """Raised by :func:`build_pool` when ``should_cancel()`` returns True."""


@dataclass
class PoolResult:
    """The cells to show next, in order, plus what they were drawn from."""
    indices: np.ndarray          # HDF5 row indices, in display order
    scores: np.ndarray           # P(good) per cell; NaN when unscored (random order)
    n_matching: int              # cells matching the filter (whole file if none)
    n_candidates: int            # ...of those, not excluded (labelled / seen)
    mode: str                    # effective ordering: "random" | "uncertain" | "good"
    message: str = ""            # why the pool is empty, when it is

    def __len__(self) -> int:
        return len(self.indices)


def build_pool(
    reader: HDF5CropsReader,
    *,
    exclude_uids: set[str],
    n_target: int,
    model_path: Optional[Path] = None,
    select: str = "uncertain",
    cell_filter: Optional[CellFilter] = None,
    seed: int = 42,
    diversify: bool = False,
    uncertain_only: bool = False,
    confidence_threshold: float = 0.85,
    cache_scores: bool = False,
    progress_cb=None,
    should_cancel=None,
) -> PoolResult:
    """Choose up to ``n_target`` cells to label next. Shared by the GUI and CLI.

    1. Candidates are the cells matching ``cell_filter`` (all cells if empty).
    2. They are put in a fixed order: a seeded permutation
       (``np.random.default_rng(seed).permutation``) of every matching cell.
       The order depends only on the file, the filter and the seed — not on
       what is already labelled — so a random sample drawn across several
       sessions is the same cells as one drawn in a single sitting.
    3. Cells in ``exclude_uids`` (already labelled, hold-out, or shown this
       session) are skipped, keeping that order.
    4. ``random`` (or no model): the first ``n_target`` remaining cells.
       ``uncertain``: score them all, least-confident first (optionally only
       below ``confidence_threshold``, optionally spread across FOVs).
       ``good``: highest P(good) first. Ties keep the permutation order.

    A random pool carries NaN scores: it has no model confidence, and a 0.5
    placeholder would read as one. ``cache_scores`` reuses (and fills) the
    per-(model, HDF5) score cache in ``core.score_cache``.
    """
    if select not in SELECT_MODES:
        raise ValueError(f"select must be one of {SELECT_MODES}, got {select!r}")
    mode = "random" if model_path is None else select
    cell_filter = cell_filter or CellFilter()

    match_idx = np.flatnonzero(cell_filter.mask(reader))
    n_matching = int(match_idx.size)

    def _empty(n_candidates: int, message: str) -> PoolResult:
        return PoolResult(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32),
                          n_matching, n_candidates, mode, message)

    if n_matching == 0:
        return _empty(0, f"No cells match the filter ({cell_filter.describe()}).")

    order = np.random.default_rng(seed).permutation(match_idx)
    uids = reader.cell_uids
    keep = np.fromiter((uids[i] not in exclude_uids for i in order),
                       dtype=bool, count=order.size)
    cand = order[keep]
    if cand.size == 0:
        if cell_filter.is_empty():
            return _empty(0, "No unseen cells in this HDF5.")
        return _empty(0, f"All {n_matching:,} cells matching the filter "
                         f"({cell_filter.describe()}) are already labelled or "
                         f"were shown this session.")

    if mode == "random":
        picked = cand[:n_target]
        return PoolResult(picked, np.full(len(picked), np.nan, dtype=np.float32),
                          n_matching, int(cand.size), mode)

    scores = _score_cells(reader, cand, model_path, progress_cb, should_cancel,
                          cache_scores=cache_scores)

    if mode == "good":
        # Highest P(good) first — surfaces the rare 'good' class so a balanced
        # hold-out set fills without labelling thousands of mostly-'bad' cells.
        rank = np.argsort(-scores, kind="stable")
        return PoolResult(cand[rank][:n_target], scores[rank][:n_target],
                          n_matching, int(cand.size), mode)

    # Max-class confidence per cell (== how sure the model is); least-confident
    # first is identical to ranking by |0.5 - P(good)| ascending.
    maxconf = np.maximum(scores, 1.0 - scores)
    if uncertain_only:
        below = maxconf < confidence_threshold
        if not below.any():
            return _empty(int(cand.size),
                          f"No cells below confidence {confidence_threshold:.2f}. "
                          f"Raise the threshold or turn off 'uncertain only'.")
        cand, scores, maxconf = cand[below], scores[below], maxconf[below]
    rank = np.argsort(maxconf, kind="stable")
    ranked_idx, ranked_scores = cand[rank], scores[rank]
    if diversify:
        picked, picked_scores = _diversify_by_fov(reader, ranked_idx, ranked_scores, n_target)
    else:
        picked, picked_scores = ranked_idx[:n_target], ranked_scores[:n_target]
    return PoolResult(picked, picked_scores, n_matching, int(cand.size), mode)


def _score_cells(reader, indices, model_path, progress_cb, should_cancel,
                 cache_scores: bool = False) -> np.ndarray:
    """P(good) for each of ``indices`` (returned in the same order).

    With ``cache_scores``, cells already in the per-(model, HDF5) score cache
    are not rescored; new scores are written back even when the load is
    cancelled, so a cancelled load keeps the progress it made."""
    cache = None
    todo = indices
    if cache_scores:
        from .score_cache import ScoreCache

        cache = ScoreCache.open(reader.path, model_path, len(reader))
        todo = cache.missing(indices)
    # Read the HDF5 in file order (far faster than random access), then map
    # the scores back to the caller's order.
    todo = np.sort(todo)
    new_scores = np.empty(len(todo), dtype=np.float32)
    batch_size = 256
    try:
        if todo.size:
            from .cell_quality_classifier import CellQualityClassifier

            clf = CellQualityClassifier(str(model_path), in_channels=3)
            for lo in range(0, len(todo), batch_size):
                if should_cancel is not None and should_cancel():
                    raise PoolCancelled()
                hi = min(lo + batch_size, len(todo))
                batch = reader.get_classifier_batch(todo[lo:hi].tolist())
                _, probs = clf.predict_batch(batch)
                new_scores[lo:hi] = probs[:, 0]
                if cache is not None:
                    cache.scores[todo[lo:hi]] = probs[:, 0]
                if progress_cb is not None:
                    progress_cb(hi / len(todo), f"Scoring cells: {hi}/{len(todo)}")
    finally:
        if cache is not None and todo.size:
            cache.save()
    if cache is None:
        # ``indices`` are unique row numbers, so each has one slot in ``todo``.
        return new_scores[np.searchsorted(todo, indices)]
    n_cached = len(indices) - len(todo)
    if n_cached and progress_cb is not None:
        progress_cb(1.0, f"Reused cached scores for {n_cached:,} cells")
    return cache.scores[indices].copy()


def _diversify_by_fov(reader, ranked_idx, ranked_scores, n_target):
    """Spread the most-uncertain candidates across fields of view so a single
    noisy FOV can't dominate a page. FOVs are ``reader.fov_groups`` (unique
    across the file; ``fov_indices`` restart in every TIFF). Falls back to
    plain order."""
    try:
        from collections import OrderedDict, deque
        k = min(len(ranked_idx), n_target * 5)
        cand = ranked_idx[:k]
        cand_scores = ranked_scores[:k]
        fovs = reader.fov_groups[cand]
        buckets: "OrderedDict[int, deque]" = OrderedDict()
        for j, f in enumerate(fovs):
            buckets.setdefault(int(f), deque()).append(j)
        order: list[int] = []
        queues = list(buckets.values())
        while len(order) < n_target and any(queues):
            for q in queues:
                if q:
                    order.append(q.popleft())
                    if len(order) >= n_target:
                        break
        order = order[:n_target]
        return cand[order], cand_scores[order]
    except Exception:  # noqa: BLE001 — diversity is best-effort
        return ranked_idx[:n_target], ranked_scores[:n_target]


def sampling_record(
    result: PoolResult,
    *,
    cell_filter: Optional[CellFilter],
    seed: int,
    n_target: int,
    model_path: Optional[Path] = None,
    **extra,
) -> dict:
    """How a pool was drawn, for the session entry in ``manifest.json`` — enough
    for a methods section to state exactly how the labelled cells were chosen.
    ``extra`` adds caller-specific settings (split, diversify, …)."""
    cell_filter = cell_filter or CellFilter()
    ordering = {
        "random": "seeded permutation of the filter-matching cells",
        "uncertain": "model uncertainty, least confident first",
        "good": "model P(good), highest first",
    }[result.mode]
    rec = {
        "mode": result.mode,
        "ordering": ordering,
        "seed": int(seed),
        "filter": cell_filter.to_dict(),
        "filter_description": cell_filter.describe(),
        "n_matching": int(result.n_matching),
        "pool_size": int(n_target),
        # The model that ordered the pool ("none" when it played no part).
        "model": str(model_path) if model_path and result.mode != "random" else "none",
    }
    rec.update(extra)
    return rec


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
    cell_filter: Optional[CellFilter] = None,
    seed: int = 42,
) -> None:
    """Launch the matplotlib grid labeller against an HDF5 crops file.

    The user labels cells one page at a time; each save appends to
    ``labels.csv``. Pressing ``s`` saves without quitting; ``q`` saves and
    closes. ``cell_filter`` restricts the pool to matching cells; how the pool
    was drawn is recorded with the session in ``manifest.json``.
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
    cell_filter = cell_filter or CellFilter()
    print(f"Loaded {len(reader)} cells from {h5_path.name}")
    print(f"  {len(already_labeled)} already labeled  →  building pool of "
          f"up to {n_pool} unseen cells...")
    if not cell_filter.is_empty():
        print(f"  only cells where {cell_filter.describe()}")

    result = build_pool(
        reader,
        exclude_uids=already_labeled,
        n_target=n_pool,
        model_path=model_path,
        cell_filter=cell_filter,
        seed=seed,
        progress_cb=lambda f, m: print(f"    {m}"),
    )
    if not len(result):
        print(f"{result.message} Nothing to do.")
        reader.close()
        return
    pool_indices, scores = result.indices.tolist(), result.scores
    sampling = sampling_record(result, cell_filter=cell_filter, seed=seed,
                               n_target=n_pool, model_path=model_path,
                               split="train", source="modelprep-cli label")

    print(f"  {result.n_matching:,} cells match · {result.n_candidates:,} unlabelled"
          f"  →  pool ready: {len(pool_indices)} cells ({result.mode} order)")
    n_total = len(pool_indices)
    n_pages = math.ceil(n_total / PAGE_SIZE)

    # Pre-fetch records for each pool index so paging is snappy.
    # Records hold the full HDF5 row + classifier-ready 64×64 crop.
    records: list[CropRecord] = [reader.get(int(i)) for i in pool_indices]

    # In-memory label state for the pool (idx → "good" | "bad" | None)
    pool_labels: dict[int, Optional[str]] = {i: None for i in range(n_total)}

    session_id = make_session_id()
    session_started_at = datetime.now(timezone.utc).isoformat()
    n_session_labels = 0

    def record_session() -> None:
        store.record_session(session_id, session_started_at,
                             datetime.now(timezone.utc).isoformat(),
                             n_session_labels, sampling=sampling)

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
        """Flush all currently-labeled pool entries to the CSV, and record the
        session (with how its pool was drawn) in the manifest."""
        nonlocal n_session_labels
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
        n_session_labels += len(rows)
        record_session()
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
