"""Compare segmentation conditions object-by-object.

Run with the main MycoMorph venv (PYTHONPATH=<worktree>/src) after
prep_fields.py and segment.py. Every mask is converted to the objects the
pipeline actually carries forward (Cellpose labels -> add_cell_boundaries ->
binary -> skimage.label(connectivity=1), as segment_tiff + classify_filter_tiff
do), then compared pairwise against condition 1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import label as sk_label
from skimage.segmentation import find_boundaries

from mycomorph.core.api import PRESET_MODELS
from mycomorph.core.cell_quality_classifier import classify_and_filter_mask
from mycomorph.core.cellpose_pipeline import add_cell_boundaries
from mycomorph.core.extract import _midline

IOU_MATCH = 0.5
MSM_MODEL = Path.home() / ".modelprep/models/msm_mmc_v1_2026-06-14_15-01-07/best_model.pth"
CLASSIFIERS = {"Mtb": Path(PRESET_MODELS["mtb"]), "Msm": MSM_MODEL}

REF = "c1_411_default"
PAIRS = [  # (A, B, label)
    ("existing_02_segment", REF, "existing vs c1 (sanity)"),
    (REF, "c1b_411_default_repeat", "c1 vs c1 repeat (noise floor)"),
    (REF, "c2_421_cpsam", "c1 vs c2 (4.2.1.1 + cpsam)"),
    (REF, "c3_421_cpsam_v2", "c1 vs c3 (4.2.1.1 + cpsam_v2)"),
]
ORANGE = np.array([230, 159, 0], np.uint8)    # only in A
CYAN = np.array([86, 180, 233], np.uint8)      # only in B
YELLOW = np.array([240, 228, 66], np.uint8)    # split / merge partners
GREY = np.array([150, 150, 150], np.uint8)     # matched


def effective_labels(raw: np.ndarray) -> np.ndarray:
    return sk_label(add_cell_boundaries(raw, boundary_width=1) > 0, connectivity=1).astype(np.int32)


def overlap_table(a: np.ndarray, b: np.ndarray):
    """Return areas of a and b labels and a DataFrame of (la, lb, inter, iou)."""
    area_a = np.bincount(a.ravel())
    area_b = np.bincount(b.ravel())
    fg = (a > 0) & (b > 0)
    pairs = a[fg].astype(np.int64) * (int(b.max()) + 1) + b[fg]
    keys, inter = np.unique(pairs, return_counts=True)
    la = keys // (int(b.max()) + 1)
    lb = keys % (int(b.max()) + 1)
    iou = inter / (area_a[la] + area_b[lb] - inter)
    return area_a, area_b, pd.DataFrame({"la": la, "lb": lb, "inter": inter, "iou": iou})


def match(a: np.ndarray, b: np.ndarray):
    area_a, area_b, ov = overlap_table(a, b)
    ov = ov.sort_values("iou", ascending=False)
    used_a, used_b, matches = set(), set(), []
    for r in ov[ov.iou >= IOU_MATCH].itertuples():
        if r.la in used_a or r.lb in used_b:
            continue
        used_a.add(r.la); used_b.add(r.lb)
        matches.append((int(r.la), int(r.lb), float(r.iou)))
    ids_a = set(np.unique(a)) - {0}
    ids_b = set(np.unique(b)) - {0}
    only_a = sorted(ids_a - used_a)
    only_b = sorted(ids_b - used_b)

    # Split: one A object contains >=2 B objects (each >=50% inside A).
    # Merge: one B object contains >=2 A objects.
    frac_b_in_a = ov.inter / area_b[ov.lb]
    frac_a_in_b = ov.inter / area_a[ov.la]
    split_parts = ov[frac_b_in_a >= 0.5].groupby("la").lb.apply(list)
    splits = {int(k): [int(x) for x in v] for k, v in split_parts.items() if len(v) >= 2}
    merge_parts = ov[frac_a_in_b >= 0.5].groupby("lb").la.apply(list)
    merges = {int(k): [int(x) for x in v] for k, v in merge_parts.items() if len(v) >= 2}
    return matches, only_a, only_b, splits, merges, area_a, area_b


def local_contrast(phase: np.ndarray, lab: np.ndarray, ids, ring_px: int = 4) -> dict:
    """|median(inside) - median(ring)| / median(ring), ring excluding all objects."""
    out = {}
    objs = ndi.find_objects(lab)
    fg = lab > 0
    for i in ids:
        sl = objs[i - 1]
        if sl is None:
            continue
        y0 = max(sl[0].start - ring_px - 1, 0); y1 = min(sl[0].stop + ring_px + 1, lab.shape[0])
        x0 = max(sl[1].start - ring_px - 1, 0); x1 = min(sl[1].stop + ring_px + 1, lab.shape[1])
        sub = lab[y0:y1, x0:x1] == i
        ring = ndi.binary_dilation(sub, iterations=ring_px) & ~fg[y0:y1, x0:x1]
        p = phase[y0:y1, x0:x1].astype(np.float64)
        if ring.sum() < 5:
            out[i] = np.nan
            continue
        m_ring = np.median(p[ring])
        out[i] = abs(np.median(p[sub]) - m_ring) / max(m_ring, 1e-6)
    return out


def lengths_um(lab: np.ndarray, ids, px_per_um: float) -> dict:
    out = {}
    objs = ndi.find_objects(lab)
    for i in ids:
        sl = objs[i - 1]
        if sl is None:
            continue
        mf = _midline.midline_features(lab[sl] == i, px_per_um)
        out[i] = mf.length_um if mf is not None else np.nan
    return out


def classify(lab, channels, phase_ch, species, px_per_um) -> dict:
    """Per-object QC decision: 'edge_cell' / 'debris' / 'good' / 'bad'."""
    _, report = classify_and_filter_mask(
        labeled_mask=lab, image_channels=channels, phase_channel=phase_ch,
        model_path=str(CLASSIFIERS[species]), pixels_per_um=px_per_um,
        verbose=False,
    )
    det = report["details"]
    # Below-confidence 'good' calls are rejected by the filter too.
    return {int(k): (v[0] if not (v[0] == "good" and v[1] < 0.5) else "bad") for k, v in det.items()}


def to_rgb(phase: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(phase, (0.5, 99.5))
    g = np.clip((phase.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    return (np.stack([g] * 3, -1) * 255).astype(np.uint8)


def draw(rgb, lab, ids, colour, thick=False):
    if not len(ids):
        return
    m = np.isin(lab, ids)
    edge = find_boundaries(np.where(m, lab, 0), mode="inner")
    if thick:
        edge = ndi.binary_dilation(edge)
    rgb[edge] = colour


def overlay(phase, a, b, matches, only_a, only_b, splits, merges) -> np.ndarray:
    rgb = to_rgb(phase)
    draw(rgb, b, [m[1] for m in matches], GREY)
    draw(rgb, a, list(splits), YELLOW, thick=True)
    draw(rgb, b, list(merges), YELLOW, thick=True)
    draw(rgb, a, only_a, ORANGE, thick=True)
    draw(rgb, b, only_b, CYAN, thick=True)
    return rgb


def montage(phase, a, b, objs, tile=96, ncol=8):
    """Crops centred on each differing object: A outline orange, B outline cyan."""
    tiles = []
    for side, lab_id in objs:
        lab = a if side == "A" else b
        cy, cx = ndi.center_of_mass(lab == lab_id)
        y0 = int(np.clip(cy - tile // 2, 0, phase.shape[0] - tile))
        x0 = int(np.clip(cx - tile // 2, 0, phase.shape[1] - tile))
        sl = (slice(y0, y0 + tile), slice(x0, x0 + tile))
        rgb = to_rgb(phase[sl])
        ea = find_boundaries(a[sl], mode="inner")
        eb = find_boundaries(b[sl], mode="inner")
        rgb[ea] = ORANGE
        rgb[eb & ~ea] = CYAN
        rgb[ea & eb] = GREY
        tiles.append(np.pad(rgb, ((2, 2), (2, 2), (0, 0)), constant_values=255))
    if not tiles:
        return None
    while len(tiles) % ncol:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.concatenate(tiles[i:i + ncol], 1) for i in range(0, len(tiles), ncol)]
    return np.concatenate(rows, 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="restrict to these field ids (testing)")
    args = ap.parse_args()
    from PIL import Image

    fields = json.loads((args.cache / "fields.json").read_text())
    if args.only:
        fields = [f for f in fields if f["field_id"] in args.only]
    (args.out / "overlays").mkdir(parents=True, exist_ok=True)
    (args.out / "montages").mkdir(parents=True, exist_ok=True)
    field_rows, obj_rows, match_rows = [], [], []
    rng = np.random.default_rng(0)

    for f in fields:
        fid, species, group = f["field_id"], f["species"], f["group"]
        d = np.load(args.cache / f"{fid}.npz")
        channels, ph, px = d["channels"], int(d["phase_channel"]), float(d["pixels_per_um"])
        phase = channels[ph]
        labs = {}
        if "existing_mask" in d:
            labs["existing_02_segment"] = d["existing_mask"]
        for tag_dir in sorted(args.masks.iterdir()):
            p = tag_dir / f"{fid}.npz"
            if p.exists():
                labs[tag_dir.name] = effective_labels(np.load(p)["masks"])
        qc = {t: classify(l, channels, ph, species, px) for t, l in labs.items()}

        for ta, tb, plabel in PAIRS:
            if ta not in labs or tb not in labs:
                continue
            a, b = labs[ta], labs[tb]
            matches, only_a, only_b, splits, merges, area_a, area_b = match(a, b)
            n_a, n_b = int(a.max()), int(b.max())
            # Downstream view: cells the QC filter keeps ('good').
            good_a = {k for k, v in qc[ta].items() if v == "good"}
            good_b = {k for k, v in qc[tb].items() if v == "good"}
            good_both = sum(1 for la, lb, _ in matches if la in good_a and lb in good_b)
            ious = np.array([m[2] for m in matches]) if matches else np.array([np.nan])
            field_rows.append(dict(
                field_id=fid, species=species, group=group, pair=plabel, A=ta, B=tb,
                n_A=n_a, n_B=n_b, matched=len(matches), only_A=len(only_a), only_B=len(only_b),
                splits_A_to_B=len(splits), merges_A_to_B=len(merges),
                iou_median=float(np.median(ious)), iou_p05=float(np.percentile(ious, 5)),
                frac_iou_ge_0_9=float(np.mean(ious >= 0.9)),
                good_A=len(good_a), good_B=len(good_b), good_matched_both=good_both,
            ))
            if ta == REF or ta == "existing_02_segment":
                # Matched-object geometry (skip identical pairs to save time).
                changed = [(la, lb, i) for la, lb, i in matches if i < 0.999]
                sample = matches if len(matches) <= 400 else [matches[k] for k in rng.choice(len(matches), 400, replace=False)]
                len_a = lengths_um(a, [m[0] for m in sample], px)
                len_b = lengths_um(b, [m[1] for m in sample], px)
                for la, lb, iou in sample:
                    match_rows.append(dict(
                        field_id=fid, species=species, group=group, pair=plabel,
                        iou=iou, area_A_um2=area_a[la] / px**2, area_B_um2=area_b[lb] / px**2,
                        length_A_um=len_a.get(la, np.nan), length_B_um=len_b.get(lb, np.nan),
                        qc_A=qc[ta].get(la), qc_B=qc[tb].get(lb),
                    ))
                del changed
            # Unmatched objects: size, contrast, QC decision.
            ref_ids = [m[0] for m in matches]
            ref_sample = ref_ids if len(ref_ids) <= 300 else list(rng.choice(ref_ids, 300, replace=False))
            c_ref = local_contrast(phase, a, ref_sample)
            c_a = local_contrast(phase, a, only_a)
            c_b = local_contrast(phase, b, only_b)
            ref_vals = np.array([v for v in c_ref.values() if np.isfinite(v)])
            for side, lab, ids, cmap, q in (("only_A", a, only_a, c_a, qc[ta]), ("only_B", b, only_b, c_b, qc[tb])):
                areas = np.bincount(lab.ravel())
                for i in ids:
                    c = cmap.get(i, np.nan)
                    obj_rows.append(dict(
                        field_id=fid, species=species, group=group, pair=plabel, side=side,
                        label=i, area_um2=areas[i] / px**2, contrast=c,
                        contrast_pctile_vs_matched=float(np.mean(ref_vals <= c) * 100) if np.isfinite(c) and len(ref_vals) else np.nan,
                        qc=q.get(i),
                        in_split=(side == "only_A" and i in splits) or (side == "only_B" and any(i in v for v in splits.values())),
                        in_merge=(side == "only_B" and i in merges) or (side == "only_A" and any(i in v for v in merges.values())),
                    ))
            # Matched-reference contrast for context.
            for i, c in c_ref.items():
                obj_rows.append(dict(field_id=fid, species=species, group=group, pair=plabel,
                                     side="matched_ref_sample", label=i,
                                     area_um2=area_a[i] / px**2, contrast=c,
                                     contrast_pctile_vs_matched=np.nan, qc=qc[ta].get(i),
                                     in_split=False, in_merge=False))

            if ta == REF and tb in ("c2_421_cpsam", "c3_421_cpsam_v2"):
                short = tb.split("_", 1)[0]
                Image.fromarray(overlay(phase, a, b, matches, only_a, only_b, splits, merges)).save(
                    args.out / "overlays" / f"{fid}__c1_vs_{short}.png", optimize=True)
                objs = [("A", i) for i in only_a] + [("B", i) for i in only_b]
                if len(objs) > 64:
                    objs = [objs[k] for k in sorted(rng.choice(len(objs), 64, replace=False))]
                tile = int(round(13 * px / 16)) * 16   # ~13 um field of view per tile
                m = montage(phase, a, b, objs, tile=tile)
                if m is not None:
                    Image.fromarray(m).save(args.out / "montages" / f"{fid}__c1_vs_{short}.png", optimize=True)
        print(f"done {fid}", flush=True)

    pd.DataFrame(field_rows).to_csv(args.out / "per_field.csv", index=False)
    pd.DataFrame(obj_rows).to_csv(args.out / "unmatched_objects.csv", index=False)
    pd.DataFrame(match_rows).to_csv(args.out / "matched_objects_sample.csv", index=False)

    # Runtimes.
    rt = []
    for tag_dir in sorted(args.masks.iterdir()):
        if not (tag_dir / "run_meta.json").exists():
            continue
        meta = json.loads((tag_dir / "run_meta.json").read_text())
        for fid, s in meta["field_seconds"].items():
            rt.append(dict(tag=tag_dir.name, field_id=fid, seconds=s, cellpose=meta["cellpose"],
                           pretrained_model=meta["pretrained_model"], device=meta["device"]))
    pd.DataFrame(rt).to_csv(args.out / "runtimes.csv", index=False)


if __name__ == "__main__":
    main()
