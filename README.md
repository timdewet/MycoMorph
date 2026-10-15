# MycoMorph

Bacterial microscopy pipeline and morphological-profiling workbench with a PyQt6 desktop GUI.

MycoMorph takes raw Zeiss CZI plates or TIFF images from focus picking, through Cellpose-SAM segmentation and cell-quality filtering, to per-cell morphology features, fluorescent foci detection, and CNN embeddings. It also compares morphological profiles across experiments. It was built around *M. tuberculosis* imaging: the bundled cell-quality classifier was trained on Mtb cells. The other stages are not species-specific.

Everything runs from one desktop app. The same functionality is also available as a Python package (`mycomorph.core`) and two command-line tools.

## Contents

- [Install](#install)
- [Update](#update)
- [Using the app](#using-the-app)
- [Pipeline stages](#pipeline-stages)
- [Output folder](#output-folder)
- [Models](#models)
- [Analysis and the morphology library](#analysis-and-the-morphology-library)
- [Command line](#command-line)
- [Configuration](#configuration)
- [Development](#development)
- [Building a desktop bundle](#building-a-desktop-bundle)
- [Releases](#releases)
- [Acknowledgments](#acknowledgments)

## Install

### For end-users (Windows)

1. Download the latest `MycoMorph-windows.zip` from the [Releases](https://github.com/timdewet/MycoMorph/releases) page.
2. Unzip it anywhere (e.g. `C:\Program Files\MycoMorph\`).
3. Double-click `MycoMorph.exe`.

You don't need to install Python. The first segmentation run downloads a few hundred MB of Cellpose model weights to `%USERPROFILE%\.cellpose\`, so it needs internet access once. After that the app works offline. The Windows bundle runs on the CPU.

### For developers (from source)

This works on macOS, Linux, and Windows. You need Python 3.10 or newer.

```bash
git clone https://github.com/timdewet/MycoMorph.git
cd MycoMorph
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e ".[test]"
```

PyTorch and Cellpose are large, so the first install can take a while. MycoMorph uses CUDA, or Apple-silicon MPS, when PyTorch can see it. For an NVIDIA GPU, install the CUDA build of `torch`/`torchvision` from the [PyTorch install selector](https://pytorch.org/get-started/locally/) before running `pip install -e .`.

#### Optional extras

Install these alongside the core package, e.g. `pip install -e ".[foci,profiling]"`.

| Extra | Installs | Needed for |
| --- | --- | --- |
| `foci` | `bm3d`, `trackpy`, `basicpy` | BM3D denoising, the trackpy detector, and BaSiC flat-field correction in the foci stages. Each one loads only when you use it, so you can install just the ones you need. |
| `profiling` | `harmonypy` | Harmony batch correction in Analysis and `library compare-representations`. Without it, batch correction is skipped with a warning. |
| `napari` | `napari` | `mycomorph-cli label --mode napari` |
| `test` | `pytest`, `pytest-qt`, `ruff`, `mypy` | The test suite and CI's static checks |

## Update

### End-users (Windows zip)

When a newer release is available, the app shows a banner at launch. Click **View Release** to open the Releases page, then:

1. Download the new `MycoMorph-windows.zip` from [Releases](https://github.com/timdewet/MycoMorph/releases).
2. Unzip it over your existing `MycoMorph\` folder, replacing the old files. Your Cellpose weights in `%USERPROFILE%\.cellpose\` and your morphology library in `%USERPROFILE%\.mycomorph\` are kept.
3. Double-click `MycoMorph.exe`.

If Windows Explorer still shows the old icon after an update, move the folder once or run `ie4uinit.exe -show` to refresh the icon cache.

### Developers (from source)

```bash
git pull
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e .
```

`git pull` on its own is not enough when `pyproject.toml`, the dependencies, or bundled data files (e.g. `src/mycomorph/core/models/`) change. Re-run `pip install -e .` as well, because editable installs don't pick up new package-data entries automatically.

The update banner only appears in the packaged app. Running from source never shows it.

## Using the app

Launch with `mycomorph` (or `python -m mycomorph`). The sidebar lists the pages below. You can visit them in any order, and each stage shows **Done**, **Ready to run**, or **Needs upstream output**.

1. **Input**. Choose an input mode and an output folder.
   - **Single plate**: one or more multi-position CZIs with Zeiss well metadata. Extra CZIs must have the same channels as the first.
   - **Single file**: one CZI or TIFF, labelled with inline fields.
   - **Multiple files**: a table of CZIs and TIFFs. You can add a folder recursively, remap paths that came from another machine, and save or load the table as CSV. Rows with identical labels are pooled into one sample.
   - **Train embeddings**: skips image processing and trains a CNN on crops already in the morphology library.

   TIFF input can be OME-TIFF, ImageJ hyperstacks, or multi-series files. The phase channel is guessed from the channel names, and you can override it.
2. **Plate layout** (Single plate mode only). Select wells by clicking or dragging, fill in *Condition*, *Reporter*, *Mutant / drug*, and *Replica*, then **Apply to selection**. Only wells with a Condition are processed. You can import, export, and validate the layout as CSV. A `plate_layout.csv` already in the output folder loads automatically.
3. **Image Processing**. These pages hold the options for Focus, Segment & Classify, Features, Fluorescent Normalisation, and Foci Detection. Each has a **live preview**: pick a sample and field of view to see the current settings applied. Segmentation runs inside a draggable ROI, and cell outlines are coloured by fate: kept, edge, debris, or rejected. On the foci pages you can drag histogram thresholds to filter detected foci without re-running detection.
4. **Embeddings**. Options for the CNN embedding stage.
5. **Run**. Tick the stages to run and press **Run pipeline**. Progress and the log appear here. **Reuse existing outputs** is on by default (see [Re-runs and provenance](#re-runs-and-provenance)).
6. **Analysis**. Explore the results (see [Analysis](#analysis-and-the-morphology-library)).

For single-plane images, untick **Focus** and keep **Split**. Z-stacks need Focus.

## Pipeline stages

| Stage | Default | What it does |
| --- | --- | --- |
| Split | on | Writes one TIFF per well or sample without focus picking. It is skipped when Focus is ticked, because Focus writes the same per-sample files. |
| Focus | on | Collapses each Z-stack to one in-focus image per field of view. Modes: extended depth of field (default), per-tile, or whole-field. The ensemble metric is the default; six single metrics are also available. You can also save Z-maps or a MIP companion channel. |
| Segment | on | Runs Cellpose-SAM on the phase channel and appends a `BinaryMask` channel to each image. Options: diameter, minimum size, flow and cell-probability thresholds, GPU. Pixel size is read from the metadata. |
| Classify | on | A rule-based edge and debris filter, plus an optional CNN cell-quality classifier. Cells it rejects are removed from the mask. |
| Features | off | Measures each cell's morphology (MicrobeJ-style shape descriptors plus sub-pixel midline measurements) and intensity, and makes QC plots. It also saves cell crops to HDF5 (on by default), and can optionally register the run in the [morphology library](#analysis-and-the-morphology-library). |
| Fluorescent Normalisation | off | Prepares fluorescence channels for foci detection with one of: none, top-hat, Gaussian low-pass, Richardson–Lucy, BM3D, BM3D + deconvolution, or BaSiC. Can also subtract spectral crosstalk. Foci Detection reads its output, so run it first; the `none` method passes images through unchanged. |
| Foci Detection | off | Finds fluorescent foci in each cell using one of seven detectors: DoG, LoG, trackpy, h-maxima, wavelet (default), radial symmetry, or bacteroidal. Options include an SNR floor, sub-pixel refinement, and splitting merged foci. **Save filter…** and **Label foci…** keep threshold choices and manual labels alongside the run. |
| Embeddings | off | Trains or fine-tunes a CNN (ResNet-18 or a lightweight network, optionally with supervised-contrastive loss) on the Features crops and writes a per-cell embedding. Requires Features with HDF5 crops. |

### Re-runs and provenance

With **Reuse existing outputs** on, a re-run picks up where the last one stopped:

- From Segment onwards, each output has a `<file>.provenance.json` sidecar. It records a digest of the input file, stage options, channels, and classifier weights. An output is reused only if the digest still matches; otherwise it is rebuilt. Outputs left half-written by an interrupted run are detected and redone.
- Split and Focus outputs are reused whenever the expected file exists. **If you change the Focus options, untick Reuse or delete the old `01_*` folder.**
- If you relabel wells, existing outputs are renamed to match the new labels instead of being recomputed.
- `run_manifest.json` records each stage's options, outputs, and timing. It also records the software versions (MycoMorph, Python, Cellpose, PyTorch, and the git commit for source installs). Versions are not part of the digest, so upgrading MycoMorph does not force recomputation.

## Output folder

Files are named `<condition>__<reporter>__<mutant_or_drug>__R<replica>`. A typical full run produces:

```
<output folder>/
├── run_manifest.json                 # options, timings, outputs, software versions
├── plate_layout.csv                  # Single plate mode (bulk_entries.csv in other modes)
├── 01_split_and_focused/             # <stem>_focused.tif (+ _zmaps.tif), <stem>__acquisition.json
│                                     #   01_focus/ or 01_split/ when only one of the two runs
├── 02_segment/                       # images + BinaryMask channel
├── 03_classify/                      # images with rejected cells removed from the mask
├── 04_features/
│   ├── <stem>.parquet / .csv         # per-cell features for each sample
│   ├── <stem>__crops.h5
│   ├── all_features.parquet / .csv   # all samples combined
│   ├── all_crops.h5
│   ├── qc_plots/
│   └── embeddings/                   # cnn_embeddings.parquet, best_model.pth, final_model.pth
├── 04b_fluorescent_normalisation/    # float32 TIFFs, __crosstalk.json
├── 04c_foci_detection/               # <stem>.parquet, one row per focus
├── 04d_foci_filtered/                # written by "Save filter…"
├── 05_foci_filters/                  # threshold JSONs for each filtered set
└── 05_foci_labels/                   # written by "Label foci…"
```

Each output file from Segment onwards also has its own `.provenance.json` sidecar (not shown above).

In **Train embeddings** mode, results go to `<output folder>/library_embeddings/` instead.

## Models

### Segmentation

Segmentation uses Cellpose-SAM. The default model is `cpsam`, pinned so that masks don't change when Cellpose is upgraded. `cpsam_v2` (Cellpose ≥ 4.2) is also offered, but it changes about 15% of QC-passing cells and grows cell areas by 1–4%. The bundled classifiers were trained on `cpsam` masks, so the app warns when you choose v2. The comparison is in [validation/cellpose_model_comparison/REPORT.md](validation/cellpose_model_comparison/REPORT.md).

### Cell-quality classifier

On the **Segment & Classify** page, Classify offers three choices:

- **none (rules only)**: the edge and debris filter on its own.
- **mtb**: the bundled *M. tuberculosis* model (`src/mycomorph/core/models/mtb_best_model.pth`). Its training curves and config are in `assets/models_mtb/`.
- **custom…**: your own `.pth`.

By default, a cell is kept only if the classifier calls it good with P(good) at or above the **Confidence threshold** slider (0.5 by default; higher is stricter). Models trained in ModelPrep store a tuned decision threshold in a `training_config.json` next to the `.pth`. When that file is present, you can tick **Use the model's tuned threshold** to keep a cell only if P(bad) is below that threshold. This is off by default.

**Labeller / trainer** opens a window where you label uncertain cells from a segmented folder as good or bad. Once you have 20 labels, you can train a new classifier. **Model details** shows ROC and PR curves for the selected model.

## Analysis and the morphology library

The **morphology library** collects per-cell features and crops from many runs so that experiments can be compared. It lives at `~/.mycomorph/morphology_library/`, and you can change the folder in the app. Older `~/.mycoprep/` or `feature_library` locations are moved there automatically.

- To add a run, tick library registration on the Features page, or use `mycomorph-cli library add`.
- The library stores a copy of each run's features, but only a *path* to its `all_crops.h5`. Moving a run folder breaks embedding training from it. `scripts/relink_library.py` repairs the paths.
- **Morphology Library Browser** (opened from the Features, Embeddings, and Analysis pages) lists the registered runs. You can filter, import, edit, and remove them there.

The **Analysis** page plots a UMAP of per-condition profiles. Profiles are built from either features or CNN embeddings, and summarised either as means or by optimal-transport distance. On this page you can:

- colour points by cluster, run, condition, reporter, replica, pathway, or any feature;
- choose a pooled or per-run baseline;
- apply Harmony batch correction (requires the `profiling` extra);
- merge replicates;
- highlight genes and compare them in a heatmap;
- export ranked matches and permutation tests as CSV, or open the plot in a browser.

## Command line

| Command | Purpose |
| --- | --- |
| `mycomorph` | Launch the GUI (same as `python -m mycomorph`) |
| `mycomorph-cli split` | Split a multi-position CZI plate into per-well TIFFs |
| `mycomorph-cli segment` | Run Cellpose-SAM segmentation on CZI or TIFF input |
| `mycomorph-cli label` | Label cells interactively (grid, napari, or montage) |
| `mycomorph-cli train` | Train a cell-quality classifier |
| `mycomorph-cli review` | Active-learning review of classifier predictions |
| `mycomorph-cli library …` | Manage the morphology library: `list`, `summary`, `add`, `update`, `remove`, `precompute-ot-views`, `gene-template`, `compare-representations` |
| `mycomorph-cli focus …` | Same as `mycomorph-focus` |
| `mycomorph-focus process` | Pick best-focus slices from a CZI and write OME-TIFFs |
| `mycomorph-focus review` / `label` / `compare` | Sample scenes, label best slices by hand, and score the focus metrics against those labels |

Each command takes `--help`. `mycomorph-focus process` defaults to whole-field focus with the `vollath_f4` metric. The GUI defaults to EDF with the ensemble metric. By default, `process` moves the source CZI into `raw/` when it succeeds; pass `--no-archive` to leave it in place.

`scripts/` holds maintenance tools for the morphology library, each run as `python scripts/<name>.py --help`:

- `relink_library.py`: rewrite crop paths after moving the library to another machine.
- `drop_model.py`: delete a trained embedding model so it is retrained.
- `inspect_embeddings.py`: report how much of each run and condition each architecture's embeddings cover.

## Configuration

| Environment variable | Effect |
| --- | --- |
| `MYCOMORPH_FOCI_WORKERS` | Number of worker processes for batch foci detection |
| `MYCOMORPH_CROP_CACHE_BYTES` | RAM cache for crops during embedding training (default 12 GB) |
| `MYCOMORPH_OT_JOBS` | Parallel jobs for optimal-transport precompute |
| `MYCOMORPH_SMOKE_TEST` | Set to `1` to make the GUI quit shortly after opening (used by CI) |

## Development

```
src/mycomorph/
├── core/                 # library: no Qt imports
│   ├── api.py            # facade the GUI stages call
│   ├── focus/            # focus picking + mycomorph-focus CLI
│   ├── extract/          # per-cell features, crops, QC plots, OT analysis, morphology library
│   ├── foci/             # normalisation, crosstalk, foci detectors
│   ├── autoencoder/      # CNN embedding models and training
│   ├── models/           # bundled classifier weights (package data)
│   └── provenance.py     # output signatures and software versions
└── gui/
    ├── panels/           # one panel per sidebar page
    ├── pipeline/         # stage adapters, runner, run context
    └── widgets/          # live preview, plate map, library browser, …
assets/                   # app icon; Mtb classifier training artifacts
packaging/                # PyInstaller spec
constraints/              # pinned Python 3.12 stack for the desktop bundle
validation/               # one-off validation studies (e.g. Cellpose model comparison)
tests/
```

MycoMorph merges the former **ImagingPipeline** (core library) and **ImagePipelineGUI** (frontend) projects and was previously named `mycoprep`.

Run the same checks as CI:

```bash
ruff check src tests
QT_QPA_PLATFORM=offscreen pytest -q
```

The tests need no GPU or microscopy data, because Cellpose is replaced with a fake. On headless Linux, PyQt6 needs the system libraries listed in [.github/workflows/ci.yml](.github/workflows/ci.yml).

CI runs on Ubuntu and macOS with Python 3.10 and 3.11. Pull requests also build and smoke-launch the Windows bundle.

## Building a desktop bundle

The bundle config is in [packaging/mycomorph.spec](packaging/mycomorph.spec). Release builds use Python 3.12 and the pinned stack in [constraints/release-py312.txt](constraints/release-py312.txt). PyInstaller does **not** cross-compile, so build on the OS you intend to ship to.

### Windows (`.exe`)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -c constraints\release-py312.txt -e . pyinstaller==6.16.0
pyinstaller packaging\mycomorph.spec
```

This produces `dist\MycoMorph\`, which contains `MycoMorph.exe` plus its DLLs and data. Zip the **whole folder** to distribute it.

### macOS / Linux

```bash
pip install -c constraints/release-py312.txt -e . pyinstaller==6.16.0
pyinstaller packaging/mycomorph.spec
```

This produces `dist/MycoMorph/`, which runs only on the OS family it was built on.

### Building Windows from a non-Windows host

PyInstaller can't cross-compile. Use one of:

- **GitHub Actions** with `runs-on: windows-latest`. This is how releases are built (see below).
- A Windows VM (Parallels, UTM, VMware).
- A spare Windows machine.

## Releases

Pre-built Windows bundles are published on the [Releases](https://github.com/timdewet/MycoMorph/releases) page. Changes are listed in [CHANGELOG.md](CHANGELOG.md).

### Cutting a new release

Releases are driven by [release-please](https://github.com/googleapis/release-please-action) and [Conventional Commits](https://www.conventionalcommits.org/). You don't bump versions, tag, or run PyInstaller by hand.

1. Merge PRs into `main` with conventional-commit messages (see below).
2. release-please opens a single "chore(main): release mycomorph X.Y.Z" PR and keeps it up to date. The PR bumps the version in `src/mycomorph/__init__.py` and `pyproject.toml` and writes a CHANGELOG entry.
3. When you're ready to ship, merge that PR. release-please tags the commit and creates a GitHub Release with the changelog as its body.
4. A Windows runner builds `dist/MycoMorph/`, checks that the tag matches `__version__`, zips the folder as `MycoMorph-windows.zip`, and uploads it to the release.
5. Installed copies of the app show the update banner on their next launch.

**Conventional-commit prefixes that drive version bumps:**

| Prefix | Meaning | Bump |
| --- | --- | --- |
| `feat: …` | new user-facing feature | minor |
| `fix: …` | bug fix | patch |
| `feat!: …` or `BREAKING CHANGE:` in the footer | breaking API/UX change | major |
| `chore: …`, `docs: …`, `refactor: …`, `test: …`, `style: …`, `ci: …` | maintenance | none |

A PR with no `feat:` or `fix:` commits doesn't trigger a release, so refactors and README changes ship with the next real release.

**Manual rebuild:** if a build fails for a transient reason, re-run the failed job from the GitHub Actions UI. The release already exists; the re-run only rebuilds and re-uploads the zip.

## Acknowledgments

MycoMorph uses methods from previously published research:

- **Cellpose-SAM** (Pachitariu, Rariden & Stringer, *bioRxiv* 2025) for cell segmentation.
- **MicrobeJ** (Ducret et al., *Nat. Microbiol.* 2016), **Oufti** (Paintdakhi et al., *Mol. Microbiol.* 2016), and **PSICIC** (Guberman et al., *PLoS Comput. Biol.* 2008) for the sub-pixel contour, medial-axis, and gradient-snap midline methods.
- **MOMIA** ([jzrolling/MOMIA](https://github.com/jzrolling/MOMIA), MIT licence) as inspiration for the midline-derived per-cell morphology columns.
- For foci detection and normalisation:
  - the à trous wavelet detector (Olivo-Marin, *Pattern Recognit.* 2002);
  - radial-symmetry localisation (Parthasarathy, *Nat. Methods* 2012);
  - h-maxima detection (Smal et al., *IEEE Trans. Med. Imaging* 2010);
  - the bacteroidal detector benchmarked by Wang et al. (*Biophys. J.* 2015);
  - Crocker–Grier centroid detection via trackpy;
  - BM3D (Dabov et al. 2007);
  - BaSiC (Peng et al. 2017).
- **Harmony** (Korsunsky et al., *Nat. Methods* 2019) for batch correction, via harmonypy.

Our implementations are written from scratch on top of scikit-image and SciPy; no upstream code is vendored. Each module cites its sources in its docstrings (e.g. `src/mycomorph/core/extract/_midline.py`, `src/mycomorph/core/foci/detectors/`). Credit for the underlying methods belongs to their original authors.

## License

MIT. See [LICENSE](LICENSE).
