# ModelPrep on MycoMorph's design system: before and after

Offscreen `window.grab()` captures of every ModelPrep page and dialog, taken
while ModelPrep moved onto MycoMorph's `mycomorph.gui.ui` (plan Phase C, step 3).

- `before/`: ModelPrep's own vendored copy of the design system (`mycomorph/modelprep/gui/ui/`), at commit `7052b1a`.
- `after/`: the same pages on `mycomorph.gui.ui`, with MycoMorph's spin boxes (decision D7).

| File | What |
|---|---|
| `01_source` | Source page with an HDF5 loaded |
| `02_label` | Label cells, pool loaded with the `mtb` preset, two thumbnails labelled |
| `03_finetune` | Fine-tune |
| `04_models` | Models tab, first run selected |
| `05_cell_filter_dialog` | "Only show cells from…" |
| `06_cell_detail` | Cell inspector (crop mode) |
| `07_consistency_dialog` | Label-consistency check after sampling |
| `*_dark` | The four pages again, with the dark theme |

All data is synthetic: a 96-cell `all_crops.h5` of drawn rods, labels written
for the screenshots, and two 2-epoch CPU training runs. The metrics in the
Models tab mean nothing. HOME pointed at a scratch folder, so no real labels
or `~/.modelprep` were read or written.

What changed:
- **Spin boxes:** they now have MycoMorph's stepper buttons. ModelPrep removed
  the buttons on the Label and Fine-tune pages, so those fields are wider, and
  the Label toolbar wraps one control earlier.
- **Consistency dialog:** the cells differ because the sample is random.
- **Incidental differences:** the scratch-folder paths on the Source page, and
  the sidebar's live RAM, CPU and disk readouts.
- Everything else is pixel-identical, including both dialogs that have no spin
  boxes. Tokens, elevation, icons and the sidebar's layout were already the same
  in both copies.
- **Optional polish (not done):** ModelPrep's old style drew the stepper column
  inside the field's rounded border. MycoMorph's draws it as a separate column
  at the right edge (compare `03_finetune`). Adopting ModelPrep's rule would
  change every MycoMorph spin box, so that is a separate decision.
