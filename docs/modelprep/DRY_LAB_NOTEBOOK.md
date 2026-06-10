# Dry-lab notebook — conventions (SOP)

**Locked:** 2026-06-09 · **Operator:** Tim de Wet

Keep computational / data-analysis work to the same standard as bench work: one protocol per task,
planned before and recorded during, synced to the Notion **🧬 Lab Notebook**. This page is the
convention; [`protocols/TEMPLATE_dry-lab-protocol.md`](protocols/TEMPLATE_dry-lab-protocol.md) is the
starting point for each new task.

## When to open an entry

Open a dry-lab entry when the work would be hard to reproduce or re-justify in a month: training a
model, a multi-step analysis, a pipeline run that feeds a decision, a parameter sweep. Skip it for
throwaway one-liners.

## What to record per run — the non-negotiable five

1. **Command** — the exact invocation (copy-paste-able).
2. **Code version** — repo + commit hash (`git rev-parse --short HEAD`). If the code isn't in git
   yet, **put it in git** — a dry-lab run you can't pin to a commit isn't reproducible.
3. **Input fingerprint** — data path + something that identifies *which* data (n items, size, or a
   hash).
4. **Output location** — where artifacts landed (model dir, results file, plots).
5. **Key metric + decision** — the number that mattered and what you concluded.

Everything else (params, plots, deviations) is nice-to-have on top of these five.

## Where artifacts live

- **Code** — the repo it belongs to, under git.
- **Protocol `.md`** — `<repo>/docs/protocols/`, versioned alongside the code it drives.
- **Large outputs** (models, HDF5, big plots) — *not* in git; record their path in the Run log.
  e.g. ModelPrep models land in `~/.modelprep/models/trained_<ts>/`.
- **The narrative + key plots** — the Notion entry.

## Notion ↔ local

- One Notion **Lab Notebook** entry per task; set Project / Status / Effort / Priority just like a
  wet-lab entry.
- The `.md` is the source of truth for the *procedure*; the Notion entry mirrors it and carries the
  **Results** (filled after the run) + embedded plots.
- Cross-link both ways: entry ↔ `.md` path ↔ commit / PR.

## Results-added-later discipline

A protocol is written **before** the run (Aim → Decision rule). After the run, fill the **Results**
section: the run-log metrics, the plot, and the verdict against the decision rule. An entry with an
empty Results section is unfinished.

## Checklist for a new dry-lab task

- [ ] Copy `TEMPLATE_dry-lab-protocol.md` → `docs/protocols/<date>_<slug>.md`
- [ ] Fill **Aim** + **Decision rule** *before* running anything
- [ ] Create the matching Notion Lab Notebook entry (Status: In progress)
- [ ] Record the non-negotiable five per run in the Run log
- [ ] After: fill **Results**, set Status: Done, link the commit / PR
