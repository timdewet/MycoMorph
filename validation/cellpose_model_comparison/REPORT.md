# Cellpose 4.1.1/cpsam vs 4.2.1.1 (cpsam, cpsam_v2): segmentation comparison

Run on 2026-10-13 on an Apple M5 (MPS). This is a comparison only. MycoMorph, `pyproject.toml`, the constraints file and the main venv were not changed.

## Recommendation: (a) upgrade to 4.2.1.1, and pin `cpsam` in the same change

- **Upgrading while keeping cpsam changes nothing.** 4.2.1.1 with `pretrained_model="cpsam"` gave masks bit-identical to 4.1.1 on all 20 fields, in both species. The runtime was the same.
- **cpsam_v2 is a different segmentation, not a cleanup.**
  - About 8–10% of objects are not matched one-to-one.
  - Matched cells shift: median IoU is 0.90–0.95, and only 53–81% of matched objects reach IoU ≥ 0.9.
  - About 15% of the cells that pass the current QC filter are replaced by a different set.
  - Areas of QC-passing cells grow by a median 1–4%. The growth differs by condition: +3.6% for Mtb ftsZ against +1.3% for NT. That would bias knockdown-vs-control comparisons.
- **The claimed benefit doesn't show up in this data.** cpsam_v2 does not end up with fewer objects. It *added* 19% more objects in Msm MMC-, and it removed and added objects in roughly equal numbers in Mtb. The objects it removes are no lower in contrast than the ones it adds. 90–99% of all changed objects are rejected by the QC filter anyway, so cpsam_v2's net effect on the analysed population is to swap some cells for others and shift sizes.
- **Adopting cpsam_v2 (option b)** would mean reprocessing both biology datasets and relabelling or retraining both classifiers, for no demonstrated gain.
- **Option (a) vs (c).** Staying on 4.1.1 (c) gives the same masks as (a). (a) keeps the dependency current at no cost to the outputs.

**Caveat:** 4.2.1.1 switches the *default* model to cpsam_v2. MycoMorph builds `CellposeModel(gpu=...)` without naming a model in three places:

- `src/mycomorph/core/api.py:285`
- `src/mycomorph/core/cellpose_pipeline.py:670`
- `src/mycomorph/gui/widgets/live_preview/worker.py:577`

If the upgrade lands without the pin, every run silently moves to cpsam_v2. The pin and the model, Cellpose and MycoMorph versions in run outputs (out of scope here) should ship together with the version bump.

## Two findings that matter beyond this decision

1. **The real runs did not use the GUI's default diameter of 25 px.** The `run_manifest.json` files were overwritten by later partial runs (Embeddings only for Mtb; Normalisation and Foci only for plate4), so segmentation options aren't recorded anywhere. Testing variants against the existing `02_segment` masks showed the runs used diameter None (no rescale), not 25. In Cellpose 4, None and 30 are equivalent. Session restore overrides the shipped default.

   | Field | Diameter 25 | Diameter None |
   |---|---|---|
   | Mtb NT f00 | 1364/1521 matched, IoU median 0.879 | 1451/1521, IoU 0.963 |
   | Msm MMC- backbone f00 | 61/80 matched, IoU 0.912 | 79/80, IoU 0.997 |

   All conditions here use diameter None. The paper should report the diameter actually used, and runs should record it.

2. **The May Mtb outputs don't reproduce exactly with today's 4.1.1, but the August Msm outputs do.**

   | Run | Matched | Median IoU |
   |---|---|---|
   | Msm (plate4, Aug) | 97.9% | 0.998 |
   | Mtb (07_05, May) | 95.3% | 0.955 |

   Cellpose was unpinned in `pyproject.toml` until 2026-07-13, so the May run probably used an earlier 4.x release. CUDA vs MPS can't explain it, because the August Msm run was also on the Windows GPU machine. For scale, this existing gap is about half the size of the cpsam→cpsam_v2 change. It's another argument for recording versions in run outputs.

## Setup

| Condition | Environment | Model |
|---|---|---|
| c1 | main venv, Cellpose 4.1.1 | `CellposeModel(gpu=True)` → cpsam (what MycoMorph does) |
| c1b | same as c1, re-run | noise floor |
| c2 | `~/venvs/cellpose421`, Cellpose 4.2.1.1 | `pretrained_model="cpsam"` |
| c3 | `~/venvs/cellpose421`, Cellpose 4.2.1.1 | `pretrained_model="cpsam_v2"` (the new default) |

- **Controlled versions.** Both venvs use Python 3.13.12, torch 2.11.0, numpy 2.4.4 and opencv-python-headless 4.13.0.92; OpenCV was pinned down from the 5.0 that 4.2.1.1 pulled in. Only Cellpose and its weights differ.
- **Segmentation call.** Every condition calls MycoMorph's own `segment_phase` with flow 0.4, cellprob 0.0, min_size 15 and diameter None.
- **What gets compared.** Masks are converted to the objects the pipeline actually carries forward: `add_cell_boundaries` → binary → `skimage.label(connectivity=1)`, as `segment_tiff` and `classify_filter_tiff` do.
- **Fields: 20, all from focused images (the 01_focus input to segmentation).**
  - Mtb `Processed/07_05_2026`, iniB-mScarlet, EDF, 1024², 6.94 px/µm: NT FOVs 0, 2, 4, 6, 8 and ftsZ FOVs 0, 2, 4, 6, 8.
  - Msm `plate4_rep1`, single plane, 2048², 13.88 px/µm, control wells only:
    - MMC-: backbone_275 f0, f8; cntl2_276 f0, f8; cntl3_277 f0.
    - MMC+: the same five.
  - 2297_63 was not used.
- **Matching.** One-to-one matching at IoU ≥ 0.5. A *split* is one c1 object containing ≥ 2 c3 objects, each ≥ 50% inside it; a *merge* is the reverse.
- **QC filter.** Each object gets the real QC decision from `classify_and_filter_mask`: edge and debris rules plus the CNN, with confidence 0.5.
  - Mtb: the bundled `mtb_best_model.pth`.
  - Msm: `~/.modelprep/models/msm_mmc_v1_2026-06-14_15-01-07/best_model.pth`. It isn't bundled; I picked the newest and best of the msm_mmc_v1 runs (hold-out balanced accuracy 0.85). The plate4 run doesn't record which model it used.
- **Contrast.** Local phase contrast = |median(object) − median(4-px surrounding ring, excluding other objects)| / median(ring).
- **Lengths.** Lengths come from MycoMorph's `_midline.midline_features`, on up to 400 matched objects per field.

## Results: c1 (4.1.1 cpsam) vs c3 (4.2.1.1 cpsam_v2)

c1 vs c1b and c1 vs c2 were **bit-identical on 20/20 fields**: 0 added, 0 removed, every IoU 1.000. They're omitted from the table below; the full tables are in `outputs/summary_tables.md`.

### Object changes

| | Mtb NT | Mtb ftsZ | Msm MMC- | Msm MMC+ |
|---|---|---|---|---|
| Objects c1 → c3 | 7931 → 7940 (+0.1%) | 3404 → 3378 (−0.8%) | 676 → 807 (**+19.4%**) | 959 → 985 (+2.7%) |
| Matched at IoU ≥ 0.5 | 92.5% | 89.7% | 89.8% | 91.6% |
| Removed / added | 597 / 606 | 351 / 325 | 69 / 200 | 81 / 107 |
| Splits / merges (c1→c3) | 43 / 54 | 13 / 33 | 22 / 7 | 8 / 11 |
| Matched IoU median [p5] | 0.915 [0.733] | 0.904 [0.718] | 0.937 [0.689] | 0.946 [0.763] |
| Matched with IoU ≥ 0.9 | 60.7% | 52.5% | 68.7% | 80.6% |

### Effect on QC-passing cells

| | Mtb NT | Mtb ftsZ | Msm MMC- | Msm MMC+ |
|---|---|---|---|---|
| QC-good c1 → c3 | 1294 → 1323 (+2.2%) | 606 → 681 (+12.4%) | 125 → 121 (−3.2%) | 99 → 107 (+8.1%) |
| c1 QC-good cells still matched and QC-good | 84.7% | 90.4% | 83.2% | 83.8% |
| QC-good area Δ, median (share with \|Δ\| > 10%) | +1.3% (4.4%) | **+3.6%** (13.9%) | +0.8% (6.7%) | +1.7% (6.0%) |
| QC-good length Δ, median (share with \|Δ\| > 10%) | −0.1% (16.6%) | +0.0% (15.5%) | −0.0% (5.8%) | +0.2% (4.8%) |

### Are the changed objects junk?

| | Mtb NT | Mtb ftsZ | Msm MMC- | Msm MMC+ |
|---|---|---|---|---|
| QC reject rate: all c1 objects | 83% | 79% | 81% | 89% |
| QC reject rate: removed objects | 98.5% | 98.6% | 89.9% | 93.8% |
| QC reject rate: added objects | 97.5% | 95.4% | 96.5% | 88.8% |
| Contrast median: matched / removed / added | 0.36 / 0.31 / 0.31 | 0.36 / 0.29 / 0.29 | 0.38 / 0.33 / 0.30 | 0.33 / 0.26 / 0.31 |
| Area µm² median: matched / removed / added | 1.29 / 0.93 / 1.08 | 1.41 / 1.00 / 1.27 | 3.18 / 3.67 / 2.47 | 4.48 / 2.46 / 3.99 |

### Runtime per field, median seconds on MPS, excluding model load

| | c1 4.1.1 cpsam | c2 4.2.1.1 cpsam | c3 4.2.1.1 cpsam_v2 |
|---|---|---|---|
| Mtb 1024² | 10.6 | 9.9 | 9.9 |
| Msm 2048² | 42.2 | 39.8 | 40.0 |

The model makes no runtime difference. 4.2.1.1 is about 5% faster, but c1 ran first, so that may be warm-up or thermal noise.

### What this says about "fewer spurious masks in low-contrast regions"

- **Does cpsam_v2 preferentially remove low-contrast objects?** No. Removed and added objects have the same contrast as each other, and both are lower than matched cells: 55% of removed and 54% of added Mtb objects fall below the matched cells' 25th-percentile contrast.
- **Where does it actually differ?** Mainly in how it partitions clumps and touching cells.
  - In Msm MMC-, it breaks large cpsam blobs that span clumps or background into individual cells. In backbone f08 alone, 164 → 245 objects.
  - In Mtb, it re-partitions touching pairs and small groups.
- **Do the changes reach the analysed population?** Mostly not. The QC filter already rejects 90–99% of changed objects, against a baseline of about 80–89%. So the "spurious" objects cpsam_v2 drops were already being filtered out.
- **What does change downstream?** Mainly the masks of the cells that pass QC, which are slightly larger, by an amount that depends on condition.

## Overlays

- **`outputs/overlays/<field>__c1_vs_c3.png`:** full-resolution field.
  - Grey: matched objects.
  - Orange: cpsam only (removed by v2).
  - Cyan: cpsam_v2 only (added).
  - Yellow: split or merge partners.
- **`outputs/montages/<field>__c1_vs_c3.png`:** up to 64 crops, about 13 µm each, centred on changed objects. Orange is the c1 outline, cyan the c3 outline, grey where they coincide.
- **Good ones to start with:**
  - `msm_MMC-_backbone_275_f08` (clump break-up)
  - `msm_MMC+_cntl2_276_f00`
  - `mtb_ftsZ_f08`
  - `mtb_NT_f08` (densest field)

## Files

- **`prep_fields.py`** (main venv): caches the 20 fields and existing 02_segment masks. The cache lives in the session scratchpad; the field images are not stored here.
- **`segment.py`** (either venv): runs one condition and writes `outputs/masks/<tag>/` with masks and `run_meta.json` (versions, device, per-field timings).
- **`diagnose_existing.py`:** the settings test against existing 02_segment masks, which found the diameter.
- **`analyze.py`** (main venv): matching, QC, contrast, morphology, overlays and montages. It writes these CSVs:
  - `per_field.csv`
  - `unmatched_objects.csv`
  - `matched_objects_sample.csv`
  - `runtimes.csv`
- **`summarize.py`:** writes `summary_by_species.csv`, `summary_by_group.csv`, `summary_tables.md` and `runtime_summary.csv`.

To reproduce, with `S` = a cache dir, `V` = the main venv python and `N` = `~/venvs/cellpose421/bin/python`, all with `PYTHONPATH=src`:

```
$V prep_fields.py --cache $S
$V segment.py --cache $S --out outputs/masks --tag c1_411_default
$N segment.py --cache $S --out outputs/masks --tag c2_421_cpsam --model cpsam
$N segment.py --cache $S --out outputs/masks --tag c3_421_cpsam_v2 --model cpsam_v2
$V analyze.py --cache $S --masks outputs/masks --out outputs
$V summarize.py --out outputs
```

## Limitations

- **Sample size.** 20 fields: about 11k Mtb and 1.6k Msm objects. Msm control fields are sparse (80–340 cells each), so the Msm QC-good counts are small (99–125 per condition).
- **Msm classifier.** It is my choice of the msm_mmc_v1 runs, not a recorded one.
- **No ground truth.** "Better" can't be judged from this. The question here is how much changes, not which model is more accurate. If cpsam_v2 is ever reconsidered, the segmentation validation set for the paper would be the place to score both models against hand annotation.
