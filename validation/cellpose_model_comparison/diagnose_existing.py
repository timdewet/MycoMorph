"""Which settings reproduce the existing 02_segment masks?

Segments one field under several settings variants with the venv's Cellpose
and reports object agreement with the mask stored in 02_segment.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze import effective_labels, match  # noqa: E402

from mycomorph.core.cellpose_pipeline import segment_phase  # noqa: E402

VARIANTS = {
    "d25_mps": dict(gpu=True, diameter=25.0),
    "d25_cpu": dict(gpu=False, diameter=25.0),
    "dNone_mps": dict(gpu=True, diameter=None),
    "d20_mps": dict(gpu=True, diameter=20.0),
    "d30_mps": dict(gpu=True, diameter=30.0),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--field", required=True)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    from cellpose.models import CellposeModel

    d = np.load(args.cache / f"{args.field}.npz")
    phase = d["channels"][int(d["phase_channel"])]
    existing = d["existing_mask"]
    models = {}
    print(f"cellpose {md.version('cellpose')}  field {args.field}  existing n={existing.max()}")
    for name in args.variants:
        v = VARIANTS[name]
        kw = {"pretrained_model": args.model} if args.model else {}
        if v["gpu"] not in models:
            models[v["gpu"]] = CellposeModel(gpu=v["gpu"], **kw)
        raw = segment_phase(phase, models[v["gpu"]], diameter=v["diameter"],
                            flow_threshold=0.4, cellprob_threshold=0.0, min_size=15)
        lab = effective_labels(raw)
        matches, only_a, only_b, *_ = match(existing, lab)
        ious = np.array([m[2] for m in matches])
        print(f"  {name:10s} n={lab.max():5d} matched={len(matches):5d} "
              f"only_existing={len(only_a):4d} only_new={len(only_b):4d} "
              f"IoU median={np.median(ious):.3f} frac>=0.9={np.mean(ious >= 0.9):.2f}", flush=True)


if __name__ == "__main__":
    main()
