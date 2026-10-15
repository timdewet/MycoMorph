# Train *M. smegmatis* cell-quality classifier — ModelPrep (net-new)

> **Note:** written for standalone ModelPrep and MycoPrep, before ModelPrep moved into MycoMorph (0.5.0).
> Where its paths and commands are now: [../README.md](../README.md#written-before-the-move).

**Date:** _________________  **Operator:** _________________  **Status:** ☐ Planned ☐ Running ☐ Done

> **Dry-lab protocol.** Mirror this in the Notion **🧬 Lab Notebook**. Conventions:
> [`../DRY_LAB_NOTEBOOK.md`](../DRY_LAB_NOTEBOOK.md). Sibling protocol (Mtb refresh):
> [`2026-06-09_train_mtb_quality_classifier.md`](2026-06-09_train_mtb_quality_classifier.md).

## Aim

Produce a *M. smegmatis*–specific **good/bad cell-quality classifier** so MycoPrep's Classify stage
reliably filters clumps, out-of-focus cells, and debris from Msm masks **before** feature
extraction. **Decision it feeds:** whether to adopt the new `msm` model as the default classifier
for Msm runs, replacing the out-of-domain `mtb`. **Done =** the decision-rule gate (below) is met
and the model is saved + verified inside MycoPrep on a held-out field.

## Background / context

MycoPrep ships **only** an `mtb` (M. tuberculosis) quality classifier
([`../../core/models/`](../../src/modelprep/core/models/) → `mtb_best_model.pth`). *M. smegmatis* is
faster-growing with different cell size/shape, so `mtb` mis-scores many Msm cells — no Msm model
exists yet, so this is **net-new**. **ModelPrep** (v0.1.0) is the labeller+trainer; it consumes the
HDF5 crops **MycoPrep** produces and emits a `.pth` that drops back into MycoPrep's "Custom model"
picker.

> ⚠️ **Two apps, one boundary = `all_crops.h5`.** MycoPrep does CZI → focus → segment →
> **feature-extract → `all_crops.h5`**. ModelPrep does **label → train → review** on that HDF5.
> ModelPrep never touches TIFFs. ([`../../README.md`](../../README.md) §"Relationship to MycoPrep".)

**Current state:** Msm images are **acquired + segmented** in MycoPrep, but crops are **not yet
extracted** — so the first job is the MycoPrep Features step.

## Design

- **Inputs** — Msm segmented hyperstacks (MycoPrep `…/03_segment/`) → MycoPrep Features →
  `…/05_features/all_crops.h5` (schema v1.0). Fingerprint: ___ cells, ___ source CZIs, settings per
  the acquisition SOP.
- **Method** — ModelPrep active-learning **label → train → review**, warm-started from `mtb`. Binary
  good/bad, lightweight 4-layer CNN (~400 K params); the reader projects each crop to `(3, 64, 64)`
  = `[phase, mask, area]`.
- **Outputs** — `~/.modelprep/models/msm/` (`best_model.pth`, `final_model.pth`,
  `training_config.json`, `training_curves.png`, `confusion_matrix.png`, `roc_pr_curves.png`,
  `metrics.json`).

| Parameter | Value | Why |
|--|--|--|
| `--pretrained` | `mtb` | warm-start from the only existing model (out-of-domain but a sane init) |
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
- **Code** — MycoPrep @ `b392235`; ModelPrep **v0.1.0** (⚠️ not yet under git — record the version +
  date; see SOP, "put it in git").

## Plan & rounds

`mtb` is out-of-domain for Msm, so seed with a **random-order** pass first, *then* switch to
active learning once a rough `msm` exists.

| Round | Goal | Target / gate |
|--|--|--|
| 0 | MycoPrep Features → `all_crops.h5` (Classify **skipped**) | HDF5 written; `info` OK |
| 1 | **Random-order** labelling (omit `--model`) → seed balanced set → train rough `msm` | ~150–200 cells, roughly balanced good/bad |
| 2 | **Active learning** from round-1 `msm` (label uncertain) → retrain | +~150 cells; aim ≥120 "bad" total |
| 3 | `review --mode uncertain` → correct mistakes → retrain | gate met (see Decision rule) |

## Steps

### Step 1 — Extract crops in MycoPrep (Features, Classify skipped)

In MycoPrep, run the **Features** stage on the segmented Msm output with **Classify skipped** and
`save_crops = True`, so clumps / OOF / debris stay in the pool to label as **bad**.

- [ ] `all_crops.h5` written at: ______________________________________________
- [ ] n cells (from the Features log): ________

### Step 2 — Inspect the HDF5 contract

```bash
modelprep-cli info --h5 <…/05_features/all_crops.h5>
```

- [ ] `schema_version` = 1.0, phase + mask channels detected, native crop → `3 × 64 × 64`
  projection reported (if this errors, the MycoPrep Features export is mismatched — fix there first)

### Step 3 — Round 1: random-order labelling (seed set)

```bash
modelprep-cli label --h5 <…/all_crops.h5> --n-pool 1000
```

Label in the grid: `g` good · `b` bad · `u` undo · `s` save · `n`/`p` page. Aim ~150–200, roughly
balanced. Labels append to `<h5 parent>/modelprep_labels/labels.csv`.

- [ ] ~150–200 cells labelled (good: ___ / bad: ___)

### Step 4 — Round 1: train a rough `msm`

```bash
modelprep-cli train --h5 <…/all_crops.h5> \
    --labels-dir <…/modelprep_labels> \
    --pretrained mtb \
    -o ~/.modelprep/models/msm/
```

- [ ] `best_val_acc` = ______ at epoch ______ — review `confusion_matrix.png` + `metrics.json`

### Step 5 — Round 2: active learning from `msm`

```bash
modelprep-cli label --h5 <…/all_crops.h5> \
    --model ~/.modelprep/models/msm/best_model.pth --n-pool 1000
```

Now cells are ranked by uncertainty (`|0.5 − P(good)|`). Label the genuinely ambiguous ones; push
the minority class. Then **re-run Step 4** to retrain.

- [ ] +~150 labelled (total good: ___ / bad: ___); retrained `best_val_acc` = ______

### Step 6 — Review & correct

```bash
modelprep-cli review --h5 <…/all_crops.h5> \
    --model ~/.modelprep/models/msm/best_model.pth --mode uncertain
```

- [ ] corrections needed: ___ % — if &gt; ~10 %, label more (Step 5) and retrain; else proceed

### Step 7 — Verify inside MycoPrep

Load `~/.modelprep/models/msm/best_model.pth` as a Classify **Custom model**; run Classify on a
**held-out** Msm CZI (not used for labelling).

- [ ] clumps / OOF / debris filtered, good cells kept; cell counts sane vs `mtb` baseline

### Step 8 — (optional) Register as a preset

```bash
cp ~/.modelprep/models/msm/best_model.pth \
   <ModelPrep>/src/modelprep/core/models/msm_best_model.pth   # then: pip install -e .
```

- [ ] `modelprep-cli models` lists `msm`

## Run log

| Date | Commit / ver | Command (short) | Input (n cells) | Key params | Output dir | Key metric | Notes |
|--|--|--|--|--|--|--|--|
|  |  | `info` |  | — | — | schema OK? |  |
|  |  | `label` (random) |  | n-pool 1000 | labels.csv | n good/bad |  |
|  |  | `train` r1 |  | pretrained mtb | `…/models/msm` | best_val_acc |  |
|  |  | `label` (AL) |  | --model msm | labels.csv | n good/bad |  |
|  |  | `train` r2 |  | pretrained mtb | `…/models/msm` | best_val_acc |  |

## Readouts

`best_val_acc` + best epoch; `confusion_matrix.png`; per-class precision/recall/F1 and ROC/PR from
`metrics.json`; labelled-cell counts per class; % corrections in the review pass; and the qualitative
MycoPrep held-out filtering check.

## Decision rule

**Adopt `msm`** as the default Msm classifier when **all** hold:

- `best_val_acc ≥ 0.90`, **AND**
- minority-class (**bad**) recall ≥ 0.80 (don't want to *keep* junk cells), **AND**
- `review --mode uncertain` needs **< ~10 %** corrections, **AND**
- MycoPrep held-out Classify visually filters clumps/OOF/debris while keeping good cells.

Else: label more (focus the failure mode / minority class) and retrain. Expect **round-1 metrics to
be rough** — `mtb` is out-of-domain, so the random-order seed quality matters most early.

## Verification

- `modelprep-cli info` confirms the schema/contract before any labelling.
- `metrics.json` present and the gate above is met.
- MycoPrep Classify with `msm` on a held-out CZI filters as expected; counts compared to `mtb`.

## References

- [`../../README.md`](../../README.md) — ModelPrep input contract, CLI, MycoPrep boundary
- [`../../src/modelprep/core/cli.py`](../../src/modelprep/core/cli.py) — exact subcommands/flags
- [`../../../MycoPrep/README.md`](../../../MycoPrep/README.md) — upstream pipeline (@ `b392235`)
- [`../../../DNALibrary/ProtocolDevelopment/acquisition_settings/pilot_protocol_acquisition_settings.md`](../../../DNALibrary/ProtocolDevelopment/acquisition_settings/pilot_protocol_acquisition_settings.md) — current image settings
- Notion: 🧬 Lab Notebook entry [🖥️ Train *M. smegmatis* cell-quality model (ModelPrep)](https://app.notion.com/p/37a4dd166da4819e9207f81b76f20109)
- Sibling: [`2026-06-09_train_mtb_quality_classifier.md`](2026-06-09_train_mtb_quality_classifier.md)

## Notes / deviations

_(during run)_

## Results _(added after the run)_

| Metric | Round 1 | Final |
|--|--|--|
| n labelled (good / bad) |  |  |
| best_val_acc |  |  |
| bad recall |  |  |
| review corrections % |  |  |

**Adopted `msm` model path:** ______   **Decision (adopt / iterate):** ______
