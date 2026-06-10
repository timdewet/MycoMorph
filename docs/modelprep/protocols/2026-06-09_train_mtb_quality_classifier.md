# Refresh *M. tb* cell-quality classifier — current image settings — ModelPrep

**Date:** _________________  **Operator:** _________________  **Status:** ☐ Planned ☐ Running ☐ Done

> **Dry-lab protocol.** Mirror this in the Notion **🧬 Lab Notebook**. Conventions:
> [`../DRY_LAB_NOTEBOOK.md`](../DRY_LAB_NOTEBOOK.md). Sibling protocol (Msm, net-new):
> [`2026-06-09_train_msm_quality_classifier.md`](2026-06-09_train_msm_quality_classifier.md).

## Aim

Retrain the *M. tuberculosis* **good/bad cell-quality classifier** on data acquired with the
**current image settings**, producing `mtb_2026-06` to supersede the bundled `mtb`. **Decision it
feeds:** whether `mtb_2026-06` beats the shipped `mtb` on current-settings fields and should become
the default for Mtb runs. **Done =** the gate (below) is met *and* `mtb_2026-06` ≥ old `mtb` in a
head-to-head on a held-out current-settings field.

## Background / context

The bundled `mtb` ([`../../core/models/`](../../src/modelprep/core/models/) → `mtb_best_model.pth`)
was trained on **earlier optics/exposure**. Acquisition settings have since been locked to a new
configuration (see acquisition SOP), so the shipped model systematically mis-scores cells on current
images. Unlike the Msm case this is **in-domain** (same species) — so active learning from `mtb`
works immediately and should need **fewer labels**. We keep the old `mtb` untouched and name the
refresh `mtb_2026-06` so the two can be compared directly.

> ⚠️ **Two apps, one boundary = `all_crops.h5`.** MycoPrep does CZI → focus → segment →
> **feature-extract → `all_crops.h5`**. ModelPrep does **label → train → review** on that HDF5.
> ([`../../README.md`](../../README.md) §"Relationship to MycoPrep".)

**Current state:** current-settings Mtb images are **acquired + segmented** in MycoPrep, but crops
are **not yet extracted** — so the first job is the MycoPrep Features step.

## Design

- **Inputs** — current-settings Mtb segmented hyperstacks (MycoPrep `…/03_segment/`) → MycoPrep
  Features → `…/05_features/all_crops.h5` (schema v1.0). Fingerprint: ___ cells, ___ source CZIs,
  acquisition settings = current locked config.
- **Method** — ModelPrep active-learning **label → train → review**, warm-started from `mtb`. Binary
  good/bad, lightweight 4-layer CNN (~400 K params); reader projects each crop to `(3, 64, 64)` =
  `[phase, mask, area]`.
- **Outputs** — `~/.modelprep/models/mtb_2026-06/` (`best_model.pth`, `final_model.pth`,
  `training_config.json`, `training_curves.png`, `confusion_matrix.png`, `roc_pr_curves.png`,
  `metrics.json`).

| Parameter | Value | Why |
|--|--|--|
| `--pretrained` | `mtb` | warm-start from the in-domain shipped model — fast convergence |
| `--n-pool` | 1000 | unlabeled cells fetched per labelling session |
| `--epochs` | 80 (default) | early stopping (`--patience 15`) usually stops sooner |
| `--batch-size` | 32 (default) | — |
| `--lr` | 1e-3 (default) | AdamW + cosine anneal |
| `--val-fraction` | 0.2 (default) | stratified train/val split |
| classes | `good`, `bad` | fixed binary head (`cell_quality_classifier.py`) |

## Environment

- **Hardware** — macOS / Apple Silicon → **MPS** auto-detected (else CPU). Override with `--device`.
- **Software** — `modelprep` conda env (Python 3.11): `conda create -n modelprep python=3.11 &&
  conda activate modelprep && pip install -e .` (run from the ModelPrep repo).
- **Code** — MycoPrep @ `b392235`; ModelPrep **v0.1.0** (⚠️ not yet under git — record version +
  date; see SOP).

## Plan & rounds

In-domain → **no random-order seed needed**; go straight to active learning from `mtb`.

| Round | Goal | Target / gate |
|--|--|--|
| 0 | MycoPrep Features → `all_crops.h5` (Classify **skipped**) | HDF5 written; `info` OK |
| 1 | **Active learning** from `mtb` → label uncertain → train `mtb_2026-06` | ~150–200 cells, focus where current settings differ |
| 2 | `review --mode uncertain` → correct → retrain | gate met + A/B vs old `mtb` |

## Steps

### Step 1 — Extract crops in MycoPrep (Features, Classify skipped)

Run MycoPrep **Features** on the segmented current-settings Mtb output with **Classify skipped** and
`save_crops = True`.

- [ ] `all_crops.h5` written at: ______________________________________________
- [ ] n cells: ________

### Step 2 — Inspect the HDF5 contract

```bash
modelprep-cli info --h5 <…/05_features/all_crops.h5>
```

- [ ] `schema_version` = 1.0, phase + mask channels detected, `3 × 64 × 64` projection reported

### Step 3 — Round 1: active-learning labelling from `mtb`

```bash
modelprep-cli label --h5 <…/all_crops.h5> --model mtb --n-pool 1000
```

Cells ranked by uncertainty vs the shipped `mtb`. Label the genuinely ambiguous ones — these are
exactly the cells current settings shifted. `g`/`b`/`u`/`s`, `n`/`p` to page.

- [ ] ~150–200 cells labelled (good: ___ / bad: ___)

### Step 4 — Round 1: train `mtb_2026-06`

```bash
modelprep-cli train --h5 <…/all_crops.h5> \
    --labels-dir <…/modelprep_labels> \
    --pretrained mtb \
    -o ~/.modelprep/models/mtb_2026-06/
```

- [ ] `best_val_acc` = ______ at epoch ______ — review `confusion_matrix.png` + `metrics.json`

### Step 5 — Review & correct

```bash
modelprep-cli review --h5 <…/all_crops.h5> \
    --model ~/.modelprep/models/mtb_2026-06/best_model.pth --mode uncertain
```

- [ ] corrections needed: ___ % — if &gt; ~10 %, label more and re-run Step 4; else proceed

### Step 6 — A/B head-to-head: old `mtb` vs `mtb_2026-06`

Run `review` with **each** model over the **same** unlabeled current-settings cells and compare how
many cells you'd correct (lower = better-calibrated on current settings).

```bash
modelprep-cli review --h5 <…/all_crops.h5> --model mtb --mode uncertain          # old
modelprep-cli review --h5 <…/all_crops.h5> \
    --model ~/.modelprep/models/mtb_2026-06/best_model.pth --mode uncertain        # new
```

- [ ] old `mtb` corrections: ___ %   ·   `mtb_2026-06` corrections: ___ %

### Step 7 — Verify inside MycoPrep

Load `~/.modelprep/models/mtb_2026-06/best_model.pth` as a Classify **Custom model**; run on a
**held-out** current-settings Mtb CZI; compare filtering side-by-side with old `mtb`.

- [ ] `mtb_2026-06` filters current-settings junk at least as well as old `mtb`; counts sane

### Step 8 — (optional) Register as a preset

```bash
cp ~/.modelprep/models/mtb_2026-06/best_model.pth \
   <ModelPrep>/src/modelprep/core/models/mtb_2026-06_best_model.pth   # then: pip install -e .
```

- [ ] `modelprep-cli models` lists both `mtb` and `mtb_2026-06`

## Run log

| Date | Commit / ver | Command (short) | Input (n cells) | Key params | Output dir | Key metric | Notes |
|--|--|--|--|--|--|--|--|
|  |  | `info` |  | — | — | schema OK? |  |
|  |  | `label` (AL) |  | --model mtb | labels.csv | n good/bad |  |
|  |  | `train` r1 |  | pretrained mtb | `…/mtb_2026-06` | best_val_acc |  |
|  |  | `review` A/B |  | mtb vs new | — | corr % old/new |  |

## Readouts

`best_val_acc` + best epoch; `confusion_matrix.png`; per-class precision/recall/F1 and ROC/PR from
`metrics.json`; labelled-cell counts; **A/B correction-rate** old vs new; and the MycoPrep held-out
side-by-side filtering check.

## Decision rule

**Replace `mtb` with `mtb_2026-06`** for current-settings runs when **all** hold:

- `best_val_acc ≥ 0.90`, **AND**
- minority-class (**bad**) recall ≥ 0.80, **AND**
- in the **A/B** (Step 6), `mtb_2026-06` correction-rate **≤** old `mtb` (i.e. as good or better),
  **AND**
- MycoPrep held-out filtering ≥ old `mtb` qualitatively.

If `mtb_2026-06` does **not** beat the old model, keep `mtb` — current settings didn't shift cell
appearance enough to matter, which is itself a useful (recorded) result.

## Verification

- `modelprep-cli info` confirms the schema/contract before labelling.
- `metrics.json` present; gate met; A/B numbers recorded.
- MycoPrep Classify side-by-side on a held-out current-settings CZI.

## References

- [`../../README.md`](../../README.md) — ModelPrep input contract, CLI, MycoPrep boundary
- [`../../src/modelprep/core/cli.py`](../../src/modelprep/core/cli.py) — exact subcommands/flags
- [`../../../MycoPrep/README.md`](../../../MycoPrep/README.md) — upstream pipeline (@ `b392235`)
- [`../../../DNALibrary/ProtocolDevelopment/acquisition_settings/pilot_protocol_acquisition_settings.md`](../../../DNALibrary/ProtocolDevelopment/acquisition_settings/pilot_protocol_acquisition_settings.md) — **definition of "current image settings"**
- Notion: 🧬 Lab Notebook entry [🖥️ Refresh *M. tb* cell-quality model — current settings (ModelPrep)](https://app.notion.com/p/37a4dd166da48171be16f959f778cc92)
- Sibling: [`2026-06-09_train_msm_quality_classifier.md`](2026-06-09_train_msm_quality_classifier.md)

## Notes / deviations

_(during run)_

## Results _(added after the run)_

| Metric | old `mtb` | `mtb_2026-06` |
|--|--|--|
| review corrections % (same cells) |  |  |
| best_val_acc | — |  |
| bad recall | — |  |

**Adopted model path:** ______   **Decision (replace / keep old):** ______
