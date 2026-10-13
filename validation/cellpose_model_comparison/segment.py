"""Segment the cached comparison fields with one Cellpose configuration.

Runs under either venv (PYTHONPATH=<worktree>/src) and calls MycoMorph's own
``segment_phase`` so the only thing that differs between conditions is the
CellposeModel passed in. Settings are those of the real runs: diameter None
(no rescaling), flow 0.4, cellprob 0.0, min_size 15. The manifests no longer
record segment options, and the Segment panel's default diameter of 25 px does
NOT reproduce the existing 02_segment outputs; diameter None does (see
diagnose_existing.py and REPORT.md).

    --model default   -> CellposeModel(gpu=True)            (what MycoMorph does)
    --model cpsam     -> CellposeModel(gpu=True, pretrained_model="cpsam")
    --model cpsam_v2  -> CellposeModel(gpu=True, pretrained_model="cpsam_v2")
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import time
from pathlib import Path

import numpy as np

from mycomorph.core.cellpose_pipeline import segment_phase

SEG_KW = dict(diameter=None, flow_threshold=0.4, cellprob_threshold=0.0, min_size=15)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", choices=["default", "cpsam", "cpsam_v2"], default="default")
    args = ap.parse_args()

    import torch
    from cellpose.models import CellposeModel

    kw = {} if args.model == "default" else {"pretrained_model": args.model}
    t0 = time.perf_counter()
    model = CellposeModel(gpu=True, **kw)
    load_s = time.perf_counter() - t0

    fields = json.loads((args.cache / "fields.json").read_text())
    out_dir = args.out / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warm-up (MPS kernel compilation) on a crop of the first field; not timed.
    first = np.load(args.cache / f"{fields[0]['field_id']}.npz")
    segment_phase(first["channels"][int(first["phase_channel"])][:512, :512], model, **SEG_KW)

    timings = {}
    for f in fields:
        d = np.load(args.cache / f"{f['field_id']}.npz")
        phase = d["channels"][int(d["phase_channel"])]
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t = time.perf_counter()
        masks = segment_phase(phase, model, **SEG_KW)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        dt = time.perf_counter() - t
        timings[f["field_id"]] = dt
        np.savez_compressed(out_dir / f"{f['field_id']}.npz", masks=masks.astype(np.int32))
        print(f"{args.tag} {f['field_id']}: {int(masks.max())} masks in {dt:.1f}s", flush=True)

    meta = {
        "tag": args.tag,
        "model_arg": args.model,
        "pretrained_model": str(getattr(model, "pretrained_model", "")),
        "cellpose": md.version("cellpose"),
        "torch": md.version("torch"),
        "opencv": md.version("opencv-python-headless"),
        "device": str(getattr(model, "device", "")),
        "segment_kwargs": SEG_KW,
        "model_load_s": load_s,
        "field_seconds": timings,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "field_seconds"}, indent=2))


if __name__ == "__main__":
    main()
