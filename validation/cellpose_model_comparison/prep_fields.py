"""Extract the comparison fields from the focused TIFFs into a local cache.

Run with the main MycoMorph venv (PYTHONPATH=<worktree>/src). Writes one
.npz per field holding all image channels, the phase-channel index, pixel
size, and — where an existing 02_segment output is available — the mask that
the real 4.1.1/cpsam run produced for that field (pipeline-effective labels).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from skimage.measure import label as sk_label

from mycomorph.core.label_cells import load_hyperstack

BASE = Path("/Users/timdewet/Library/CloudStorage/Dropbox-MMRU/Network Data/Microscopy/Tim/2026")
MTB = BASE / "Mtb_CRISPRi/Processed/07_05_2026"
MSM = BASE / "DNA_Array/plate4_rep1"

# (field_id, species, group, focus tiff, segment tiff, fov index)
FIELDS: list[tuple[str, str, str, Path, Path, int]] = []
for gene in ("NT", "ftsZ"):
    name = f"ATc+__iniB-mScarlet__{gene}__R1_focused.tif"
    for fov in (0, 2, 4, 6, 8):
        FIELDS.append((f"mtb_{gene}_f{fov:02d}", "Mtb", f"Mtb {gene}",
                       MTB / "01_focus" / name, MTB / "02_segment" / name, fov))
for cond in ("MMC-", "MMC+"):
    for well, fovs in (("backbone_275", (0, 8)), ("cntl2_276", (0, 8)), ("cntl3_277", (0,))):
        name = f"ATc+{cond}__DnaN-mCherry_imuB-eGFP__{well}__R1_focused.tif"
        for fov in fovs:
            FIELDS.append((f"msm_{cond}_{well}_f{fov:02d}", "Msm", f"Msm {cond}",
                           MSM / cond / "processed/01_focus" / name,
                           MSM / cond / "processed/02_segment" / name, fov))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    index = []
    by_file: dict[Path, list] = {}
    for f in FIELDS:
        by_file.setdefault(f[3], []).append(f)

    for focus_path, fields in by_file.items():
        data, meta = load_hyperstack(focus_path)
        labels = list(meta.get("Labels") or [])[: data.shape[1]]
        spacing = float(meta.get("spacing") or 0) or None
        px_per_um = 1.0 / spacing if spacing else 13.8767
        # Phase is the brightfield/phase channel; Focus labels it 'Bright' or 'Phase'.
        phase_ch = next((i for i, l in enumerate(labels)
                         if l.lower().startswith(("bright", "phase", "ph", "tl"))), 0)

        seg_path = fields[0][4]
        seg = None
        # Sanity check against existing outputs only for Mtb and the plate4
        # backbone wells (the others are 0.5 GB online-only files each).
        wants_existing = "Mtb" in str(seg_path) or "backbone" in seg_path.name
        if wants_existing and seg_path.exists() and seg_path.stat().st_size > 0:
            try:
                seg, _ = load_hyperstack(seg_path)
            except Exception as exc:  # online-only or unreadable
                print(f"  could not read {seg_path.name}: {exc}")

        for fid, species, group, _fp, _sp, fov in fields:
            out = {
                "channels": data[fov],
                "phase_channel": phase_ch,
                "pixels_per_um": px_per_um,
            }
            has_existing = False
            if seg is not None and seg.shape[0] > fov:
                # Same relabelling classify_filter_tiff applies to the binary mask.
                out["existing_mask"] = sk_label(seg[fov, -1] > 0, connectivity=1).astype(np.int32)
                has_existing = True
            np.savez_compressed(args.cache / f"{fid}.npz", **out)
            index.append({
                "field_id": fid, "species": species, "group": group,
                "source": str(focus_path), "fov": fov, "labels": labels,
                "phase_channel": phase_ch, "pixels_per_um": px_per_um,
                "shape": list(data[fov].shape), "has_existing_02_segment": has_existing,
            })
            print(f"{fid}: shape={data[fov].shape} phase={phase_ch} ({labels[phase_ch] if labels else '?'}) "
                  f"px/um={px_per_um:.3f} existing={has_existing}")

    (args.cache / "fields.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
