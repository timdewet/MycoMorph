"""Aggregate analyze.py outputs into the tables used in REPORT.md."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PAIR_ORDER = [
    "existing vs c1 (sanity)",
    "c1 vs c1 repeat (noise floor)",
    "c1 vs c2 (4.2.1.1 + cpsam)",
    "c1 vs c3 (4.2.1.1 + cpsam_v2)",
]


def to_md(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids a tabulate dependency in the main venv)."""
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        idx = " / ".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)
        lines.append("| " + " | ".join([idx] + [str(v) for v in row.values]) + " |")
    return "\n".join(lines)


def pct(x):
    return f"{100 * x:.1f}%" if np.isfinite(x) else "–"


def summarise(pf, mo, uo, keys):
    rows = []
    for key, g in pf.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        sel = dict(zip(keys, key))
        m = mo.copy(); u = uo.copy()
        for k, v in sel.items():
            m = m[m[k] == v]; u = u[u[k] == v]
        nA, nB = g.n_A.sum(), g.n_B.sum()
        rej = lambda s: float(np.mean(s != "good")) if len(s) else np.nan
        oa = u[u.side == "only_A"]; ob = u[u.side == "only_B"]; ref = u[u.side == "matched_ref_sample"]
        area_rel = m.area_B_um2 / m.area_A_um2 - 1
        len_rel = m.length_B_um / m.length_A_um - 1
        rows.append({
            **sel,
            "fields": len(g),
            "objects A": int(nA),
            "objects B": int(nB),
            "Δ count": pct((nB - nA) / nA),
            "matched (IoU≥0.5)": f"{g.matched.sum()} ({pct(g.matched.sum() / nA)})",
            "only in A (removed)": int(g.only_A.sum()),
            "only in B (added)": int(g.only_B.sum()),
            "splits A→B": int(g.splits_A_to_B.sum()),
            "merges A→B": int(g.merges_A_to_B.sum()),
            "IoU median [p5]": f"{m.iou.median():.3f} [{m.iou.quantile(0.05):.3f}]" if len(m) else "–",
            "IoU ≥0.9": pct(float(np.mean(m.iou >= 0.9))) if len(m) else "–",
            "area Δ median (|Δ| median)": f"{100 * area_rel.median():+.1f}% ({100 * area_rel.abs().median():.1f}%)" if len(m) else "–",
            "length Δ median (|Δ| median)": f"{100 * len_rel.median():+.1f}% ({100 * len_rel.abs().median():.1f}%)" if len(m) else "–",
            "QC-good A": int(g.good_A.sum()),
            "QC-good B": int(g.good_B.sum()),
            "Δ QC-good": pct((g.good_B.sum() - g.good_A.sum()) / max(g.good_A.sum(), 1)),
            "QC-good in both (matched)": f"{g.good_matched_both.sum()} ({pct(g.good_matched_both.sum() / max(g.good_A.sum(), 1))} of A-good)",
            "QC reject rate, all A (ref sample)": pct(rej(ref.qc)),
            "QC reject rate, removed": pct(rej(oa.qc)),
            "QC reject rate, added": pct(rej(ob.qc)),
            "removed: edge/debris/bad/good": "/".join(str(int((oa.qc == c).sum())) for c in ("edge_cell", "debris", "bad", "good")),
            "added: edge/debris/bad/good": "/".join(str(int((ob.qc == c).sum())) for c in ("edge_cell", "debris", "bad", "good")),
            "area µm² median: matched / removed / added": f"{ref.area_um2.median():.2f} / {oa.area_um2.median():.2f} / {ob.area_um2.median():.2f}",
            "contrast median: matched / removed / added": f"{ref.contrast.median():.3f} / {oa.contrast.median():.3f} / {ob.contrast.median():.3f}",
            "removed below matched p25 contrast": pct(float(np.mean(oa.contrast < ref.contrast.quantile(0.25)))) if len(oa) else "–",
            "added below matched p25 contrast": pct(float(np.mean(ob.contrast < ref.contrast.quantile(0.25)))) if len(ob) else "–",
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    pf = pd.read_csv(args.out / "per_field.csv")
    mo = pd.read_csv(args.out / "matched_objects_sample.csv")
    uo = pd.read_csv(args.out / "unmatched_objects.csv")
    for df in (pf, mo, uo):
        df["pair"] = pd.Categorical(df["pair"], PAIR_ORDER, ordered=True)
    pf = pf.sort_values(["species", "pair"])

    by_species = summarise(pf, mo, uo, ["species", "pair"])
    by_group = summarise(pf.sort_values(["group", "pair"]), mo, uo, ["group", "pair"])
    by_species.to_csv(args.out / "summary_by_species.csv", index=False)
    by_group.to_csv(args.out / "summary_by_group.csv", index=False)

    rt = pd.read_csv(args.out / "runtimes.csv")
    rt["species"] = np.where(rt.field_id.str.startswith("mtb"), "Mtb (1024²)", "Msm (2048²)")
    rts = rt.groupby(["species", "tag"]).seconds.agg(["median", "mean", "min", "max"]).round(1)
    rts.to_csv(args.out / "runtime_summary.csv")

    with open(args.out / "summary_tables.md", "w") as fh:
        for name, df in (("By species", by_species), ("By species × condition", by_group)):
            fh.write(f"## {name}\n\n")
            # Transposed: one column per (species, pair) reads better than 25 columns.
            t = df.set_index([c for c in df.columns[:2]]).T
            t.columns = [f"{a} — {b}" for a, b in t.columns]
            fh.write(to_md(t) + "\n\n")
        fh.write("## Runtime per field (s, MPS)\n\n" + to_md(rts) + "\n")
    print((args.out / "summary_tables.md").read_text())


if __name__ == "__main__":
    main()
