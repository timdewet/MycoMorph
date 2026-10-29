# ModelPrep docs

ModelPrep, MycoMorph's labeller and trainer for the cell-quality classifier, was a separate app until MycoMorph 0.5.0. How to use it is in the [ModelPrep section of the main README](../../README.md#modelprep-label-cells-and-train-the-classifier). This folder holds:

| File | What it is |
| --- | --- |
| [labelling_rubric.md](labelling_rubric.md) | What "good" and "bad" mean when labelling. ModelPrep's Label page links to it, and the desktop bundle ships it. It is a template until the lab fills it in. |
| [DRY_LAB_NOTEBOOK.md](DRY_LAB_NOTEBOOK.md) | The dry-lab notebook conventions (SOP) that ModelPrep's training runs follow. |
| [protocols/](protocols/) | Dry-lab protocols for training the *M. tuberculosis* and *M. smegmatis* classifiers, and a template for new ones. |
| [screenshots/phase-c/](screenshots/phase-c/README.md) | ModelPrep's pages before and after it moved onto MycoMorph's design system. |
| [ModelPrep-commit-map.txt](ModelPrep-commit-map.txt) | Old and new commit hashes from importing ModelPrep's history into MycoMorph. |

## Written before the move

`DRY_LAB_NOTEBOOK.md` and the two dated protocols were written for standalone ModelPrep, next to MycoPrep (MycoMorph's former name). They are kept as written. Their paths and commands correspond to these in MycoMorph:

| In those documents | In MycoMorph |
| --- | --- |
| The ModelPrep repository and its `pip install -e .` | This repository: `pip install -e .` here installs `modelprep` and `modelprep-cli` too |
| `src/modelprep/…` | `src/mycomorph/modelprep/…` |
| `src/modelprep/core/models/` (bundled presets) | `src/mycomorph/core/models/` |
| ModelPrep's README | The [ModelPrep section](../../README.md#modelprep-label-cells-and-train-the-classifier) of MycoMorph's README |
| `<repo>/docs/protocols/` | `docs/modelprep/protocols/` |
| `~/.modelprep/models/trained_<ts>/` | `~/.modelprep/models/<name>_<date>/` for new runs; `trained_<ts>/` runs are still listed |
| MycoPrep's `05_features/all_crops.h5` | `04_features/all_crops.h5` in a MycoMorph output folder |
| A ModelPrep commit hash | Its new hash in `ModelPrep-commit-map.txt` (left column old, right column new). All zeros means the commit only changed files outside the imported folders, so it wasn't imported. |

Their links to `../MycoPrep/` and `../DNALibrary/` point to folders outside this repository.
