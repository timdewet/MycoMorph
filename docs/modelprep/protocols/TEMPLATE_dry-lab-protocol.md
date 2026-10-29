# &lt;Project&gt; — &lt;what you're computing&gt; (dry-lab protocol)

**Date:** _________________  **Operator:** _________________  **Status:** ☐ Planned ☐ Running ☐ Done

> **Dry-lab protocol.** The computational analogue of a bench protocol — same discipline, terminal
> instead of pipette. Mirror this in the Notion **🧬 Lab Notebook** (one entry per task).
> Conventions: [`../DRY_LAB_NOTEBOOK.md`](../DRY_LAB_NOTEBOOK.md).

## Aim

One paragraph: the question this work answers and the **decision** it feeds. State what "done"
looks like (the success criterion), not just the activity.

## Background / context

What already exists, what changed, why now. Link the upstream protocol / run / data and any prior
attempt.

## Design

- **Inputs** — data path(s), format, and a fingerprint (n items, size, or a hash) so the run is
  reproducible.
- **Method** — tool/script + **version** (repo @ commit, or package version).
- **Parameters** — the knobs that matter (table below).
- **Outputs** — what lands where.

| Parameter | Value | Why |
|--|--|--|
|  |  |  |

## Environment

- **Hardware** — CPU / GPU (model), RAM if relevant.
- **Software** — env name + key package versions (`conda env export` / `pip freeze | grep …`).
- **Code** — repo + commit: `git -C <repo> rev-parse --short HEAD` → ______

## Plan & rounds

The dry-lab "schedule": planned jobs / rounds and their targets, so you know when to stop.

| Round | Goal | Target / gate |
|--|--|--|
| 1 |  |  |

## Steps

Numbered, with the **actual command** in a fenced block. Tick as you go.

### Step 1 — &lt;name&gt;

```bash
<command>
```

- [ ] ran; output at ______

## Run log

The core record — one row per job actually executed.

| Date | Commit | Command (short) | Input | Key params | Output dir | Key metric | Notes |
|--|--|--|--|--|--|--|--|
|  |  |  |  |  |  |  |  |

## Readouts

The metrics / plots / files produced and how each is read.

## Decision rule

Explicit thresholds with AND/OR logic — what makes you **proceed**, **iterate**, or **stop**.

- Proceed if: ______ AND ______
- Else: ______

## Verification

Sanity checks + an end-to-end test that proves the output is real (not just that the job exited 0).

## References

Repos, related protocols (relative paths), Notion links, papers / methods.

## Notes / deviations

_(during run)_

## Results _(added after the run)_

Fill the run-log metrics, paste key plots, and state the verdict against the decision rule.

**Outcome:** ______
