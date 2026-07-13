# MycoMorph

Bacterial microscopy pre-processing pipeline with a PyQt6 desktop GUI.

MycoMorph wraps focus picking, FOV split, Cellpose-SAM segmentation, and a cell-quality classifier into a single wizard-style application for fungal/bacterial specimen prep.

This repo merges the former **ImagingPipeline** (core library) and **ImagePipelineGUI** (PyQt6 frontend) projects into a single Python package: `mycomorph` with `mycomorph.core` (library) and `mycomorph.gui` (GUI) subpackages.

## Install

### For end-users (Windows)

1. Download the latest `MycoMorph-windows.zip` from the [Releases](https://github.com/timdewet/MycoMorph/releases) page.
2. Unzip anywhere (e.g. `C:\Program Files\MycoMorph\`).
3. Double-click `MycoMorph.exe`.

First launch downloads ~few hundred MB of Cellpose model weights to `%USERPROFILE%\.cellpose\` — needs internet once. Subsequent launches are offline. No Python install required.

### For developers (from source)

Works on macOS, Linux, and Windows.

```bash
git clone https://github.com/timdewet/MycoMorph.git
cd MycoMorph
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e .[test]
```

Requires Python 3.10+. PyTorch + Cellpose are heavy installs; first install may take a while.

## Update

### End-users (Windows zip)

The frozen app checks GitHub on launch and shows a banner when a newer release is available. Click **View Release** to open the Releases page, then:

1. Download the new `MycoMorph-windows.zip` from [Releases](https://github.com/timdewet/MycoMorph/releases).
2. Unzip over your existing `MycoMorph\` folder, replacing the old files. Cellpose weights in `%USERPROFILE%\.cellpose\` are kept.
3. Double-click `MycoMorph.exe`.

If Windows Explorer still shows the old icon after an update, move the folder once or run `ie4uinit.exe -show` to flush the icon cache.

### Developers (from source)

```bash
git pull
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e .                  # re-runs if pyproject.toml or package-data changed
```

Re-run `pip install -e .` (not just `git pull`) whenever `pyproject.toml`, dependencies, or shipped data files (e.g. `src/mycomorph/core/models/`) change — editable installs don't auto-pick up new package-data entries.

The in-app update check is a no-op when running from source (it only fires for frozen PyInstaller bundles), so dev runs never show a banner.

## Run

| Command | Purpose |
| --- | --- |
| `mycomorph` | Launch the GUI |
| `mycomorph-cli --help` | Top-level CLI (was `imagingpipeline`) |
| `mycomorph-focus --help` | Focus-picking CLI (was `focuspicker`) |
| `python -m mycomorph` | Equivalent to `mycomorph` |

## Layout

```
src/mycomorph/
├── core/
│   ├── models/  # bundled classifier weights shipped via package-data
│   └── ...      # focus, segmentation, classification, CZI handling
└── gui/         # PyQt6 wizard: panels, pipeline runner, live preview
assets/
├── logo/        # app icon (svg + ico)
└── models_mtb/  # training artifacts (final_model.pth, curves, configs)
packaging/
└── mycomorph.spec  # PyInstaller spec for desktop bundle
```

## Building a desktop bundle

The bundle config lives in [packaging/mycomorph.spec](packaging/mycomorph.spec). PyInstaller does **not** cross-compile — build on the OS you intend to ship to.

### Windows (`.exe`)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -c constraints\release-py311.txt -e . pyinstaller==6.16.0
pyinstaller packaging\mycomorph.spec
```

Output: `dist\MycoMorph\` containing `MycoMorph.exe` plus DLLs/data. Zip the **whole folder** to distribute.

For GPU (CUDA) inference, install the matching PyTorch wheel before `pip install -e .`:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

CPU-only is fine if you don't need GPU; the bundle is much smaller.

### macOS / Linux

```bash
pip install -c constraints/release-py311.txt -e . pyinstaller==6.16.0
pyinstaller packaging/mycomorph.spec
```

Output: `dist/MycoMorph/`. Runs only on the OS family it was built on.

### Building Windows from a non-Windows host

PyInstaller can't cross-compile. Use one of:

- **GitHub Actions** with `runs-on: windows-latest` (recommended for repeatable releases).
- A Windows VM (Parallels, UTM, VMware) on your Mac.
- A borrowed/spare Windows machine.

## Releases

Pre-built Windows bundles are published on the [Releases](https://github.com/timdewet/MycoMorph/releases) page. Each release ships a zipped `dist\MycoMorph\` folder — see the end-user install steps above.

### Cutting a new release

Releases are driven by [release-please](https://github.com/googleapis/release-please-action) and [Conventional Commits](https://www.conventionalcommits.org/). You no longer manually bump versions, tag, or run PyInstaller — the workflow does it all.

**Workflow:**

1. Land PRs on `main` with conventional-commit messages (see below).
2. release-please opens (and keeps updated) a single "chore: release vX.Y.Z" PR that bumps the version in `src/mycomorph/__init__.py` and `pyproject.toml` and writes a CHANGELOG entry.
3. When you're ready to ship, merge that PR. release-please tags the commit and creates a GitHub Release with the changelog as the body.
4. The Windows runner builds `dist/MycoMorph/`, zips it as `MycoMorph-windows.zip`, and uploads it to the release.
5. Installed copies of the app pick up the new release on next launch via the in-app update banner.

**Conventional-commit prefixes that drive bumps:**

| Prefix             | Meaning                              | Bump  |
| ------------------ | ------------------------------------ | ----- |
| `feat: …`          | new user-facing feature              | minor |
| `fix: …`           | bug fix                              | patch |
| `feat!: …` or `BREAKING CHANGE:` in the footer | breaking API/UX change | major |
| `chore: …`, `docs: …`, `refactor: …`, `test: …`, `style: …`, `ci: …` | maintenance | none  |

A PR with no `feat:` or `fix:` commits won't trigger a release — perfect for refactors, README tweaks, etc.

**Manual rebuild:** if a build fails for a transient reason, re-run the failed job from the GitHub Actions UI. The release itself is already created; the rerun just rebuilds and re-uploads the asset.

## Acknowledgments

MycoMorph incorporates methods derived from previously published research:

- **Cellpose-SAM** for cell segmentation.
- **MicrobeJ** (Ducret et al., *Nat. Microbiol.* 2016), **Oufti** (Paintdakhi et al., *Mol. Microbiol.* 2016), and **PSICIC** (Guberman et al., *PLoS Comput. Biol.* 2008) for the sub-pixel contour / medial-axis / gradient-snap midline methodology.
- **MOMIA** ([jzrolling/MOMIA](https://github.com/jzrolling/MOMIA), MIT licence) as inspiration for the midline-derived per-cell morphology columns.

Our implementations are rewritten on top of modern scikit-image / SciPy primitives — no upstream code is vendored. Full per-module citations live alongside the code (see `src/mycomorph/core/extract/_midline.py`). Credit for the underlying methods belongs to their original authors.

## License

MIT — see [LICENSE](LICENSE).
