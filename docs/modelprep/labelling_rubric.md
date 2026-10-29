# Cell-quality labelling rubric

> **TEMPLATE — not yet written.** Every `TODO` below is for the person who owns
> the labels to fill in. Until it is filled in, labels made with ModelPrep have
> no written definition of "good" and "bad", and another lab can't reproduce
> them. Cite the finished version (with its commit) wherever the labels are used.

- **Version:** TODO (e.g. v1, YYYY-MM-DD)
- **Author(s):** TODO
- **Applies to:** TODO (organism, strain(s), imaging modality, magnification,
  MycoMorph version and crop settings, e.g. background masking off)

## Purpose

TODO: one or two sentences on what the classifier is for, e.g. which
downstream measurements a "bad" cell would corrupt.

## What you see in ModelPrep

TODO: describe the thumbnail (phase/brightfield crop, cyan segmentation
outline) and the inspector (larger crop, optional original-image context), and
which of them labels should be based on.

## Label GOOD when ALL of these hold

1. TODO: e.g. a single cell, not touching or overlapping a neighbour
2. TODO: e.g. in focus
3. TODO: e.g. the outline follows the whole cell (no cut-off poles, no merged cells)
4. TODO
5. TODO

## Label BAD when ANY of these hold

1. TODO: e.g. clump or overlapping cells
2. TODO: e.g. out of focus
3. TODO: e.g. debris, background or a segmentation artefact
4. TODO: e.g. cell cut by the image edge
5. TODO

## Borderline cases and how to decide

| Situation | Decision | Why |
|---|---|---|
| TODO: e.g. dividing cell with a visible septum | TODO | TODO |
| TODO: e.g. slightly out of focus but outline correct | TODO | TODO |
| TODO | TODO | TODO |

When still unsure: TODO (e.g. "label bad" or "skip, leave unlabelled").

## Example crops

Add 6–10 example crops per class (screenshots from the inspector), with one
line each saying which rule applies.

### Good

- TODO: `examples/good_01.png` — rule G1

### Bad

- TODO: `examples/bad_01.png` — rule B2

### Borderline

- TODO: `examples/borderline_01.png` — decided ___ because ___

## Hold-out test set

TODO: how the hold-out is built (random sample vs "Prioritise good"), its
target size, and the rule that hold-out cells are labelled once and never
re-labelled in Review.

## Consistency

TODO: record self-consistency results from ModelPrep's consistency check
(`modelprep-cli consistency`), e.g. "90% raw agreement, κ = 0.80 on 200
balanced cells, YYYY-MM-DD", and any rule changes that followed from the
disagreements.

## Change log

| Date | Version | Change |
|---|---|---|
| TODO | v1 | First version |
