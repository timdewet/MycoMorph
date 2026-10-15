"""Condition filter for the labeller: which cells a pool may be drawn from."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from mycomorph.modelprep.core import label_cells
from mycomorph.modelprep.core.cell_filter import CellFilter, distinct_values
from mycomorph.modelprep.core.crops_h5 import HDF5CropsReader
from mycomorph.modelprep.core.label_cells import build_pool, sampling_record
from mycomorph.modelprep.core.labels_store import LabelsStore


def _pool(reader, cell_filter=None, *, exclude=(), n=1000, seed=42, **kw):
    return build_pool(reader, exclude_uids=set(exclude), n_target=n,
                      cell_filter=cell_filter, seed=seed, **kw)


def _column(reader, dataset, indices):
    return reader.metadata(dataset)[np.asarray(indices, dtype=int)].tolist()


# ── reader metadata ──────────────────────────────────────────────────────────

def test_reader_exposes_conditions_and_metadata(multi_condition_h5, synthetic_h5):
    with HDF5CropsReader(multi_condition_h5) as r:
        assert set(r.metadata("conditions")) == {"ATc+MMC-", "ATc+MMC+"}
        assert r.get(0).condition == "ATc+MMC-"
        assert r.get(4).condition == "ATc+MMC+"
        with pytest.raises(KeyError):
            r.metadata("not_a_dataset")
    # A file without a 'conditions' dataset reads as blank, not an error.
    import h5py
    with h5py.File(synthetic_h5, "r+") as f:
        del f["conditions"]
    with HDF5CropsReader(synthetic_h5) as r:
        assert set(r.metadata("conditions")) == {""}
        assert r.get(0).condition == ""


def test_distinct_values_with_counts(multi_condition_h5):
    with HDF5CropsReader(multi_condition_h5) as r:
        assert distinct_values(r, "gene") == [("", 12), ("dnaN", 12), ("inhA", 12), ("NT1", 12)]
        assert distinct_values(r, "condition") == [("ATc+MMC+", 24), ("ATc+MMC-", 24)]


# ── filtering ────────────────────────────────────────────────────────────────

def test_filter_restricts_pool_to_matching_cells(multi_condition_h5):
    with HDF5CropsReader(multi_condition_h5) as r:
        res = _pool(r, CellFilter.from_mapping({"gene": ["dnaN"]}))
        assert res.n_matching == 12 and len(res) == 12
        assert set(_column(r, "genes", res.indices)) == {"dnaN"}


def test_values_within_a_field_are_ored(multi_condition_h5):
    with HDF5CropsReader(multi_condition_h5) as r:
        res = _pool(r, CellFilter.from_mapping({"gene": ["dnaN", "NT1"]}))
        assert Counter(_column(r, "genes", res.indices)) == {"dnaN": 12, "NT1": 12}


def test_combined_fields_are_anded(multi_condition_h5):
    with HDF5CropsReader(multi_condition_h5) as r:
        f = CellFilter.from_mapping({"gene": ["dnaN", "inhA"], "condition": ["ATc+MMC-"]})
        res = _pool(r, f)
        assert res.n_matching == 12
        assert set(_column(r, "genes", res.indices)) == {"dnaN", "inhA"}
        assert set(_column(r, "conditions", res.indices)) == {"ATc+MMC-"}
        assert f.describe() == "gene ∈ {dnaN, inhA} AND condition = ATc+MMC-"


def test_blank_values_and_condition_label_select_controls(multi_condition_h5):
    """Control-like wells have gene "" — selectable as blank, or by label."""
    with HDF5CropsReader(multi_condition_h5) as r:
        by_gene = _pool(r, CellFilter.from_mapping({"gene": [""]}))
        by_label = _pool(r, CellFilter.from_mapping({"condition_label": ["empty"]}))
        assert sorted(by_gene.indices) == sorted(by_label.indices)
        assert set(_column(r, "condition_labels", by_gene.indices)) == {"empty"}
        assert CellFilter.from_mapping({"gene": [""]}).describe() == "gene = (blank)"


def test_empty_filter_behaves_like_no_filter(multi_condition_h5):
    with HDF5CropsReader(multi_condition_h5) as r:
        empty = CellFilter.from_mapping({"gene": [], "condition": []})
        assert empty.is_empty() and empty.to_dict() == {}
        assert empty.describe() == "all cells"
        assert empty.mask(r).all()
        a, b = _pool(r, None, n=20), _pool(r, empty, n=20)
        assert a.n_matching == b.n_matching == len(r)
        assert a.indices.tolist() == b.indices.tolist()


def test_filter_matching_nothing_gives_a_clear_message(multi_condition_h5):
    with HDF5CropsReader(multi_condition_h5) as r:
        f = CellFilter.from_mapping({"gene": ["dnaN"], "condition": ["ATc+MMC=?"]})
        res = _pool(r, f)
        assert len(res) == 0 and res.n_matching == 0
        assert "No cells match the filter" in res.message
        assert "ATc+MMC=?" in res.message
        assert f.unknown_values(r) == {"condition": ["ATc+MMC=?"]}


def test_all_matching_cells_labelled_message(multi_condition_h5):
    with HDF5CropsReader(multi_condition_h5) as r:
        f = CellFilter.from_mapping({"gene": ["inhA"]})
        done = set(r.cell_uids[f.mask(r)])
        res = _pool(r, f, exclude=done)
        assert len(res) == 0 and res.n_matching == 12 and res.n_candidates == 0
        assert "already labelled" in res.message


def test_unknown_field_is_rejected():
    with pytest.raises(ValueError, match="Unknown filter field"):
        CellFilter.from_mapping({"well": ["A1"]})


# ── reproducibility ──────────────────────────────────────────────────────────

def test_same_seed_same_sample(multi_condition_h5):
    f = CellFilter.from_mapping({"condition": ["ATc+MMC+"]})
    with HDF5CropsReader(multi_condition_h5) as r:
        a = _pool(r, f, n=10, seed=7).indices.tolist()
        b = _pool(r, f, n=10, seed=7).indices.tolist()
        c = _pool(r, f, n=10, seed=8).indices.tolist()
    assert a == b
    assert a != c


def test_sample_continues_after_labelling(multi_condition_h5):
    """The order is fixed by file + filter + seed: labelling part of a sample
    and drawing again yields the rest of the same sample."""
    f = CellFilter.from_mapping({"gene": ["dnaN", "NT1"]})
    with HDF5CropsReader(multi_condition_h5) as r:
        whole = _pool(r, f, n=10).indices.tolist()
        labelled = set(r.cell_uids[whole[:4]])
        rest = _pool(r, f, exclude=labelled, n=6).indices.tolist()
    assert rest == whole[4:]


# ── existing behaviour inside the filtered set ───────────────────────────────

def test_exclusions_apply_within_filter(multi_condition_h5, tmp_path):
    """Already-labelled cells (incl. hold-out) are skipped inside the filter."""
    from mycomorph.modelprep.core.labels_store import SPLIT_TEST, SPLIT_TRAIN, LabelRow

    store = LabelsStore(tmp_path / "labels")
    store.append_batch([
        LabelRow(cell_uid="cell_0000", label_idx=0, label_name="good", split=SPLIT_TRAIN),
        LabelRow(cell_uid="cell_0008", label_idx=1, label_name="bad", split=SPLIT_TEST),
    ])
    f = CellFilter.from_mapping({"gene": ["dnaN"]})  # cells 0, 4, 8, …
    with HDF5CropsReader(multi_condition_h5) as r:
        res = _pool(r, f, exclude=store.labeled_uids())
        uids = set(r.cell_uids[res.indices])
    assert res.n_matching == 12 and res.n_candidates == 10
    assert not uids & {"cell_0000", "cell_0008"}


def _fake_scores(monkeypatch):
    """Deterministic P(good) per HDF5 row, so ranking is testable without a CNN."""
    def fake(reader, indices, model_path, progress_cb, should_cancel, **_kw):
        return (np.asarray(indices, dtype=np.float32) % 10) / 10.0
    monkeypatch.setattr(label_cells, "_score_cells", fake)


def test_uncertainty_ranking_within_filter(multi_condition_h5, monkeypatch, tmp_path):
    _fake_scores(monkeypatch)
    f = CellFilter.from_mapping({"gene": ["inhA"], "condition": ["ATc+MMC-"]})
    with HDF5CropsReader(multi_condition_h5) as r:
        res = _pool(r, f, model_path=tmp_path / "m.pth", select="uncertain")
        assert res.mode == "uncertain" and len(res) == 6
        assert set(_column(r, "genes", res.indices)) == {"inhA"}
        assert set(_column(r, "conditions", res.indices)) == {"ATc+MMC-"}
        conf = np.maximum(res.scores, 1 - res.scores)
        assert np.all(np.diff(conf) >= 0)  # least confident first

        good = _pool(r, f, model_path=tmp_path / "m.pth", select="good")
        assert np.all(np.diff(good.scores) <= 0)  # most likely good first
        assert sorted(good.indices) == sorted(res.indices)

        only = _pool(r, f, model_path=tmp_path / "m.pth", uncertain_only=True,
                     confidence_threshold=0.75)
        assert np.all(np.maximum(only.scores, 1 - only.scores) < 0.75)



def test_diversify_spreads_across_fields_not_fov_numbers(multi_condition_h5):
    """fov_indices restart in every TIFF, so FOV 0 of two wells is two fields.
    Cells 0/24 share one TIFF's FOV 0 and 12/36 another's; 1 and 2 are FOVs 1, 2."""
    with HDF5CropsReader(multi_condition_h5) as r:
        assert r.fov_indices[[0, 12, 24, 36]].tolist() == [0, 0, 0, 0]
        ranked = np.array([0, 24, 12, 36, 1, 2])
        picked, _ = label_cells._diversify_by_fov(
            r, ranked, np.linspace(0.5, 0.9, len(ranked)), n_target=4)
    # Bucketing by bare fov_index would give [0, 1, 2, 24].
    assert picked.tolist() == [0, 12, 1, 2]

def test_random_select_ignores_model(multi_condition_h5, tmp_path):
    f = CellFilter.from_mapping({"gene": ["NT1"]})
    with HDF5CropsReader(multi_condition_h5) as r:
        a = _pool(r, f, n=5)
        b = _pool(r, f, n=5, model_path=tmp_path / "unused.pth", select="random")
    assert a.mode == b.mode == "random"
    assert a.indices.tolist() == b.indices.tolist()
    # Nothing scored them, so they carry no model confidence (not 0.5).
    assert np.isnan(a.scores).all() and np.isnan(b.scores).all()


def test_filtered_pools_share_the_score_cache(multi_condition_h5, tmp_path, monkeypatch):
    """Scores cached while labelling one gene are reused when the filter
    changes: only cells not scored before are run through the model."""
    import mycomorph.core.cell_quality_classifier as cqc
    from mycomorph.modelprep.core import score_cache

    monkeypatch.setattr(score_cache, "score_cache_root", lambda: tmp_path / "cache")
    score_cache.clear_memory()
    scored: list[int] = []

    class FakeClf:
        def __init__(self, *_a, **_k):
            pass

        def predict_batch(self, batch):
            scored.append(len(batch))
            p = np.full((len(batch), 2), 0.5, dtype=np.float32)
            return p.argmax(1), p

    monkeypatch.setattr(cqc, "CellQualityClassifier", FakeClf)
    model = tmp_path / "m.pth"
    model.write_bytes(b"w")
    dnan = CellFilter.from_mapping({"gene": ["dnaN"]})
    with HDF5CropsReader(multi_condition_h5) as r:
        first = _pool(r, dnan, model_path=model, cache_scores=True)
        assert len(first) == 12 and sum(scored) == 12
        assert set(_column(r, "genes", first.indices)) == {"dnaN"}
        _pool(r, dnan, model_path=model, cache_scores=True)
        assert sum(scored) == 12                       # all from the cache
        whole = _pool(r, model_path=model, cache_scores=True)
        assert len(whole) == 48 and sum(scored) == 48  # only the other 36 scored
    score_cache.clear_memory()


# ── provenance ───────────────────────────────────────────────────────────────

def test_sampling_record_describes_the_draw(multi_condition_h5):
    f = CellFilter.from_mapping({"gene": ["dnaN", "inhA"], "condition": ["ATc+MMC-"]})
    with HDF5CropsReader(multi_condition_h5) as r:
        res = _pool(r, f, n=5, seed=3)
    rec = sampling_record(res, cell_filter=f, seed=3, n_target=5, split="train")
    assert rec["mode"] == "random"
    assert rec["seed"] == 3
    assert rec["filter"] == {"gene": ["dnaN", "inhA"], "condition": ["ATc+MMC-"]}
    assert rec["filter_description"] == f.describe()
    assert rec["n_matching"] == 12 and rec["pool_size"] == 5
    assert rec["model"] == "none" and rec["split"] == "train"


def test_record_session_stores_sampling_and_updates_in_place(multi_condition_h5, tmp_path):
    store = LabelsStore(tmp_path / "labels")
    store.ensure_initialised(source_h5=multi_condition_h5, total_cells=48,
                             h5_schema_version="1.0")
    sampling = {"mode": "random", "seed": 42, "filter": {"gene": ["dnaN"]}}
    store.record_session("s1", "t0", "t1", 5, sampling=sampling)
    store.record_session("s1", "t0", "t2", 12, sampling=sampling)
    store.record_session("s2", "t3", "t4", 1)  # legacy call: no sampling
    s1 = store.session("s1")
    assert s1 == {"id": "s1", "started_at": "t0", "ended_at": "t2", "n_labels": 12,
                  "sampling": sampling}
    assert "sampling" not in store.session("s2")
    assert [s["id"] for s in store._read_manifest()["sessions"]] == ["s1", "s2"]
    assert store.session("nope") is None


# ── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_info_lists_filter_values(multi_condition_h5):
    from typer.testing import CliRunner
    from mycomorph.modelprep.core.cli import app

    out = CliRunner().invoke(app, ["info", "--h5", str(multi_condition_h5), "--values"])
    assert out.exit_code == 0, out.output
    assert "--condition" in out.output and "'ATc+MMC-'" in out.output
    assert "'inhA'" in out.output


def test_cli_label_passes_filter_and_seed(multi_condition_h5, monkeypatch):
    from typer.testing import CliRunner
    from mycomorph.modelprep.core.cli import app

    seen = {}
    monkeypatch.setattr(label_cells, "run_grid_labeling", lambda **kw: seen.update(kw))
    out = CliRunner().invoke(app, [
        "label", "--h5", str(multi_condition_h5), "--n-pool", "200",
        "--gene", "dnaN", "--gene", "inhA", "--condition", "ATc+MMC-", "--seed", "7",
    ])
    assert out.exit_code == 0, out.output
    assert seen["cell_filter"].to_dict() == {"gene": ["dnaN", "inhA"],
                                             "condition": ["ATc+MMC-"]}
    assert seen["seed"] == 7 and seen["n_pool"] == 200

    # No filter options → an empty filter (whole file).
    seen.clear()
    out = CliRunner().invoke(app, ["label", "--h5", str(multi_condition_h5)])
    assert out.exit_code == 0, out.output
    assert seen["cell_filter"].is_empty()


def test_cli_label_rejects_unknown_values(multi_condition_h5, monkeypatch):
    from typer.testing import CliRunner
    from mycomorph.modelprep.core.cli import app

    monkeypatch.setattr(label_cells, "run_grid_labeling",
                        lambda **kw: pytest.fail("labeller should not launch"))
    out = CliRunner().invoke(app, ["label", "--h5", str(multi_condition_h5),
                                   "--gene", "dnaNN"])
    assert out.exit_code == 2
    assert "dnaNN" in out.output and "'dnaN'" in out.output  # lists what's available


def test_grid_labeller_records_filter_with_session(multi_condition_h5, tmp_path, monkeypatch):
    """Drive the matplotlib labeller headlessly: click a cell, press g, quit.
    The saved row comes from the filtered pool and the session records how the
    pool was drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import KeyEvent, MouseEvent

    def fake_show(block=True):
        fig = plt.gcf()
        ax = fig.axes[0]
        x, y = ax.transAxes.transform((0.5, 0.5))
        for event in (MouseEvent("button_press_event", fig.canvas, x, y, button=1),
                      KeyEvent("key_press_event", fig.canvas, "g"),
                      KeyEvent("key_press_event", fig.canvas, "q")):
            fig.canvas.callbacks.process(event.name, event)

    monkeypatch.setattr(plt, "show", fake_show)
    f = CellFilter.from_mapping({"gene": ["inhA"], "condition": ["ATc+MMC+"]})
    labels_dir = tmp_path / "labels"
    label_cells.run_grid_labeling(multi_condition_h5, labels_dir, n_pool=50,
                                  cell_filter=f, seed=5)

    store = LabelsStore(labels_dir)
    rows = store.load().rows
    assert len(rows) == 1 and rows[0].label_name == "good"
    with HDF5CropsReader(multi_condition_h5) as r:
        first = _pool(r, f, n=50, seed=5).indices[0]
        assert rows[0].cell_uid == r.cell_uids[first]
    sampling = store.session(rows[0].session_id)["sampling"]
    assert sampling["filter"] == {"gene": ["inhA"], "condition": ["ATc+MMC+"]}
    assert sampling["seed"] == 5 and sampling["mode"] == "random"
    assert sampling["source"] == "modelprep-cli label"
    assert store.session(rows[0].session_id)["n_labels"] == 1
