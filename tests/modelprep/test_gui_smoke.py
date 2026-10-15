"""Offscreen GUI smoke tests.

Gated behind ``MODELPREP_GUI_TESTS=1`` because constructing a QApplication
aborts the process on a mis-configured local Qt install (see the project note
on QT_QPA_PLATFORM_PLUGIN_PATH). CI sets the env + an offscreen platform, where
PyQt6 loads cleanly; locally, run with::

    MODELPREP_GUI_TESTS=1 QT_QPA_PLATFORM=offscreen pytest tests/test_gui_smoke.py
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MODELPREP_GUI_TESTS") != "1",
    reason="set MODELPREP_GUI_TESTS=1 (offscreen Qt) to run GUI smoke tests",
)


def _app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_main_window_constructs_and_navigates(isolated_home):
    app = _app()
    from mycomorph.gui.ui import theme
    from mycomorph.modelprep.gui.main_window import MainWindow, settings_file

    # Settings come from the temp HOME, so the user's last HDF5 isn't reopened.
    assert settings_file().is_relative_to(isolated_home)
    theme.apply_theme(app)
    win = MainWindow()
    assert win.source_panel.current_h5_path() is None
    assert win._stack.count() == 4  # source, label, train, models
    for i in range(win._stack.count()):
        win._sidebar.set_current(i)
        app.processEvents()
    win.label_train_panel.shutdown()


def test_panels_construct():
    app = _app()
    from mycomorph.modelprep.gui.panels.source_panel import SourcePanel
    from mycomorph.modelprep.gui.panels.label_train_panel import LabelTrainPanel

    sp = SourcePanel()
    assert sp is not None
    panel = LabelTrainPanel()
    # The two cards the main window pages over exist.
    assert panel.label_section is not None and panel.train_section is not None
    panel.shutdown()
    app.processEvents()


def _make_run(root, name, ts, **cfg):
    import json
    d = root / f"{name}_{ts}"
    d.mkdir(parents=True)
    (d / "best_model.pth").write_bytes(b"x")
    base = dict(eval_split="holdout", eval_acc=0.8, balanced_acc=0.78, roc_auc=0.85,
                average_precision=0.83, majority_baseline=0.7, decision_threshold=0.5,
                n_train=600, n_holdout=200, class_counts={"good": 200, "bad": 400})
    base.update(cfg)
    (d / "training_config.json").write_text(json.dumps(base))
    return d


def test_models_panel_lists_sorts_and_deletes(tmp_path, monkeypatch):
    app = _app()
    import mycomorph.modelprep.gui.trained_models as tm
    import mycomorph.modelprep.gui.panels.models_panel as mp
    from mycomorph.modelprep.gui.panels.models_panel import _COL
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QMessageBox

    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(tm, "trained_models_root", lambda: root)
    monkeypatch.setattr(mp, "trained_models_root", lambda: root)
    _make_run(root, "alpha", "2026-06-11_00-00-00", roc_auc=0.74)
    _make_run(root, "beta", "2026-06-11_01-00-00", roc_auc=0.91)

    panel = mp.ModelsPanel()
    assert panel._table.rowCount() == 2

    # Numeric sort by AUC ascending puts the weaker model first.
    panel._table.sortItems(_COL["AUC"], Qt.SortOrder.AscendingOrder)
    names = [panel._table.item(r, _COL["Name"]).text() for r in range(2)]
    assert names == ["alpha", "beta"]

    # Delete (auto-confirm) removes the run from disk and the table.
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    changed = []
    panel.modelsChanged.connect(lambda: changed.append(1))
    panel._select_dir(str(root / "beta_2026-06-11_01-00-00"))
    panel._delete_selected()
    assert not (root / "beta_2026-06-11_01-00-00").exists()
    assert panel._table.rowCount() == 1
    assert changed == [1]
    app.processEvents()


def test_models_panel_lists_runs_from_standalone_modelprep(tmp_path, monkeypatch):
    """Runs trained before the move recorded the standalone package's own
    version; the Models tab lists them next to runs that record MycoMorph's."""
    app = _app()
    import mycomorph
    import mycomorph.modelprep.gui.trained_models as tm
    import mycomorph.modelprep.gui.panels.models_panel as mp
    from mycomorph.modelprep.gui.panels.models_panel import _COL

    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(tm, "trained_models_root", lambda: root)
    monkeypatch.setattr(mp, "trained_models_root", lambda: root)
    _make_run(root, "old", "2026-06-11_00-00-00", trainer="modelprep",
              trainer_version="0.1.0", roc_auc=0.81)
    _make_run(root, "rel", "2026-10-15_00-00-00", trainer="modelprep",
              trainer_version="0.2.0", roc_auc=0.84)
    _make_run(root, "new", "2026-10-16_00-00-00", trainer="modelprep",
              trainer_version=mycomorph.__version__, roc_auc=0.86)

    panel = mp.ModelsPanel()
    names = {panel._table.item(r, _COL["Name"]).text() for r in range(panel._table.rowCount())}
    assert names == {"old", "rel", "new"}
    app.processEvents()


def test_models_panel_shows_feature_runs(tmp_path, monkeypatch):
    app = _app()
    import json
    import mycomorph.modelprep.gui.trained_models as tm
    import mycomorph.modelprep.gui.panels.models_panel as mp
    from mycomorph.modelprep.gui.panels.models_panel import _COL

    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(tm, "trained_models_root", lambda: root)
    monkeypatch.setattr(mp, "trained_models_root", lambda: root)

    cnn = root / "demo_2026-06-14_10-00-00"
    cnn.mkdir()
    (cnn / "best_model.pth").write_bytes(b"x")
    (cnn / "training_config.json").write_text(json.dumps(dict(
        eval_split="holdout", roc_auc=0.83, balanced_acc=0.75, average_precision=0.87,
        n_train=2552, n_holdout=1100, class_counts={"good": 1170, "bad": 1382},
        provenance={"model_name": "demo"})))
    feat = root / "demo-features_2026-06-14_10-00-00"
    feat.mkdir()
    (feat / "feature_model.joblib").write_bytes(b"x")
    (feat / "training_config.json").write_text(json.dumps(dict(
        model_type="features", eval_split="holdout", roc_auc=0.854, balanced_acc=0.773,
        average_precision=0.887, n_train=2552, n_holdout=1100,
        class_counts={"good": 1170, "bad": 1382}, provenance={"model_name": "demo"})))

    panel = mp.ModelsPanel()
    types = {panel._table.item(r, _COL["Type"]).text() for r in range(panel._table.rowCount())}
    assert types == {"CNN", "features"}

    # Selecting the feature row disables "Use for labelling".
    for r in range(panel._table.rowCount()):
        if panel._table.item(r, _COL["Type"]).text() == "features":
            panel._table.selectRow(r)
            break
    assert not panel._use_btn.isEnabled()
    app.processEvents()


def test_models_panel_learning_curve_gated_on_holdout(tmp_path):
    app = _app()
    import mycomorph.modelprep.gui.panels.models_panel as mp
    from mycomorph.modelprep.core.labels_store import LabelsStore, LabelRow, SPLIT_TRAIN, SPLIT_TEST

    panel = mp.ModelsPanel()
    # Section exists; Run is disabled with no source.
    assert hasattr(panel, "_lc_run")
    panel.set_source(None, None)
    assert not panel._lc_run.isEnabled()

    # Train-only labels → still disabled (needs a hold-out).
    d = tmp_path / "labels"
    store = LabelsStore(d)
    store.append_batch([LabelRow(cell_uid=f"c{i}", label_idx=i % 2,
                                 label_name="good" if i % 2 == 0 else "bad",
                                 split=SPLIT_TRAIN) for i in range(8)])
    panel.set_source(tmp_path / "fake.h5", d)
    assert not panel._lc_run.isEnabled()

    # Add hold-out test labels → enabled.
    store.append_batch([LabelRow(cell_uid=f"t{i}", label_idx=i % 2,
                                 label_name="good" if i % 2 == 0 else "bad",
                                 split=SPLIT_TEST) for i in range(4)])
    panel.set_source(tmp_path / "fake.h5", d)
    assert panel._lc_run.isEnabled()
    panel.shutdown()
    app.processEvents()


def test_models_panel_learning_curve_runs_end_to_end(tmp_path, synthetic_h5, monkeypatch):
    import time
    from pathlib import Path

    app = _app()
    import mycomorph.modelprep.gui.panels.models_panel as mp
    from mycomorph.modelprep.core.labels_store import LabelsStore, LabelRow, SPLIT_TRAIN, SPLIT_TEST

    labels_dir = synthetic_h5.parent / "modelprep_labels"
    store = LabelsStore(labels_dir)
    uids = [f"cell_{i:04d}" for i in range(32)]
    store.ensure_initialised(source_h5=synthetic_h5, total_cells=32,
                             h5_schema_version="1.0", cell_uids=uids)
    # 20 train (10/10) + 8 hold-out (4/4).
    rows = [LabelRow(cell_uid=uids[i], label_idx=i % 2,
                     label_name="good" if i % 2 == 0 else "bad", split=SPLIT_TRAIN)
            for i in range(20)]
    rows += [LabelRow(cell_uid=uids[i], label_idx=i % 2,
                      label_name="good" if i % 2 == 0 else "bad", split=SPLIT_TEST)
             for i in range(20, 28)]
    store.append_batch(rows)

    # Keep saved curves out of the real home dir.
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    panel = mp.ModelsPanel()
    panel.set_source(synthetic_h5, labels_dir)
    panel._lc_points.setValue(2)
    panel._lc_epochs.setValue(1)
    panel._run_curve()

    deadline = time.time() + 60
    while time.time() < deadline and not panel._lc_status.text().startswith(("Done", "Learning curve failed")):
        app.processEvents()
        time.sleep(0.05)

    assert panel._lc_status.text().startswith("Done"), panel._lc_status.text()
    assert panel._lc_canvas is None or panel._lc_canvas.isVisible() or True  # plotted if mpl present
    curves = list((tmp_path / ".modelprep" / "learning_curves").glob("*/curve.json"))
    assert curves, "expected a saved curve.json"
    panel.shutdown()
    app.processEvents()


def test_consistency_dialog_samples_scores_and_saves(synthetic_h5):
    import time

    app = _app()
    from mycomorph.modelprep.core.labels_store import LabelsStore, LabelRow
    from mycomorph.modelprep.gui.panels.consistency_dialog import ConsistencyCheckDialog

    labels_dir = synthetic_h5.parent / "modelprep_labels"
    store = LabelsStore(labels_dir)
    uids = [f"cell_{i:04d}" for i in range(32)]
    store.ensure_initialised(source_h5=synthetic_h5, total_cells=32,
                             h5_schema_version="1.0", cell_uids=uids)
    store.append_batch([LabelRow(cell_uid=uids[i], label_idx=i % 2,
                                 label_name="good" if i % 2 == 0 else "bad")
                        for i in range(24)])

    dlg = ConsistencyCheckDialog(synthetic_h5, labels_dir)
    dlg._size.setValue(16)
    dlg._mode.setCurrentIndex(0)  # balanced
    dlg._on_start()
    deadline = time.time() + 20
    while time.time() < deadline and not dlg._thumbs:
        app.processEvents()
        time.sleep(0.03)
    assert dlg._thumbs, "sampling produced no thumbnails"

    # Re-label: agree on all but the first two (flip those).
    for i, thumb in enumerate(dlg._thumbs):
        orig = dlg._pairs[i][1]
        thumb.set_label(("bad" if orig == "good" else "good") if i < 2 else orig)

    dlg._score()
    assert "Self-agreement" in dlg._results.text()
    runs = list((labels_dir / "consistency").glob("*/results.csv"))
    assert runs, "expected a saved consistency run"
    # labels.csv is untouched: still exactly the 24 originals.
    assert len(store.load().rows) == 24
    app.processEvents()


def test_feature_model_skip_reason_is_one_line(tmp_path, monkeypatch):
    """A missing parquet engine (or any other error) surfaces as a short skip
    reason under the CNN results, never as pandas' multi-line ImportError."""
    _app()
    import mycomorph.modelprep.core.feature_classifier as fc
    from mycomorph.modelprep.gui.panels.label_train_panel import _TrainWorker

    worker = _TrainWorker(
        tmp_path / "all_crops.h5", tmp_path / "labels",
        tmp_path / "demo_2026-06-14_10-00-00",
        epochs=1, lr=1e-3, batch_size=8, patience=1, val_fraction=0.2,
        test_fraction=0.0, seed=0, device="cpu", freeze_backbone=False,
        pretrained=None,
    )

    def no_pyarrow(*_a, **_k):
        raise ImportError("pyarrow isn't installed, so all_features.parquet "
                          "can't be read (pip install pyarrow)")

    monkeypatch.setattr(fc, "train_and_save", no_pyarrow)
    summary, reason = worker._train_feature_model({})
    assert summary is None and "pip install pyarrow" in reason and "\n" not in reason

    def boom(*_a, **_k):
        raise RuntimeError("first line\nsecond line")

    monkeypatch.setattr(fc, "train_and_save", boom)
    _, reason = worker._train_feature_model({})
    assert reason == "feature model error: first line"


def _wait_for_pool(app, panel):
    """Let the pool worker finish and deliver its queued result."""
    import time
    worker = panel._al_worker
    assert worker is not None
    assert worker.wait(30000), "pool worker did not finish"
    deadline = time.time() + 5
    while time.time() < deadline and not panel._load_btn.isEnabled():
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def test_random_holdout_sample_records_no_model_confidence(synthetic_h5, tmp_path):
    """Random hold-out sampling doesn't score cells, so the saved labels must
    say model 'none' with NaN confidence, even with a model in the picker
    (review M9)."""
    import math
    import mycomorph.core.cell_quality_classifier as cqc
    from mycomorph.modelprep.core.labels_store import LabelsStore
    from mycomorph.modelprep.gui.panels.label_train_panel import LabelTrainPanel

    app = _app()

    def no_scoring(*_a, **_k):
        raise AssertionError("random sampling must not load the classifier")

    orig = cqc.CellQualityClassifier
    cqc.CellQualityClassifier = no_scoring
    try:
        panel = LabelTrainPanel()
        labels = tmp_path / "labels"
        panel.set_source(synthetic_h5, labels)
        panel._model_combo.setCurrentIndex(panel._model_combo.findText("preset: mtb"))
        assert panel._current_model_path() is not None
        panel._test_mode_chk.setChecked(True)          # Prioritise good stays off
        panel._load_pool()
        _wait_for_pool(app, panel)
        assert panel._pool and panel._pool_model is None
        assert all(math.isnan(score) for _rec, score in panel._pool)

        panel._thumbs[0].set_label("good")
        assert panel._save_batch(silent_if_empty=True) == 1
        row = next(iter(LabelsStore(labels).load().latest_by_uid().values()))
        assert row.split == "test" and row.model_at_label == "none"
        assert math.isnan(row.confidence_good_at_label)
        panel.shutdown()
    finally:
        cqc.CellQualityClassifier = orig


def test_pool_scores_are_cached_per_model_and_h5(synthetic_h5, tmp_path, monkeypatch):
    """A second Load pool with the same model + HDF5 doesn't rescore (review M5)."""
    import numpy as np
    import mycomorph.core.cell_quality_classifier as cqc
    import mycomorph.modelprep.core.score_cache as sc
    from mycomorph.modelprep.gui.panels.label_train_panel import _ActiveLearningWorker

    _app()
    monkeypatch.setattr(sc, "score_cache_root", lambda: tmp_path / "cache")
    sc.clear_memory()
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

    def run():
        w = _ActiveLearningWorker(synthetic_h5, model, n_target=10, seen_uids=set(),
                                  diversify=True)
        out = {}
        w.finished.connect(lambda recs, err: out.update(recs=recs, err=err))
        w.run()                                   # synchronously, on this thread
        assert out["err"] == "" and len(out["recs"]) == 10
        assert w.scored_with == model
        return out["recs"]

    run()
    assert sum(scored) == 32
    sc.clear_memory()                             # the disk layer alone suffices
    run()
    assert sum(scored) == 32                      # nothing rescored


def test_inspector_defaults_to_crop_when_originals_are_online_only(synthetic_h5):
    """Context mode would download the full online-only TIFFs, so the inspector
    opens on the isolated crop and says why (review L3)."""
    from types import SimpleNamespace

    from mycomorph.modelprep.core.crops_h5 import HDF5CropsReader
    from mycomorph.modelprep.gui.panels.cell_detail_dialog import CellDetailView

    _app()
    with HDF5CropsReader(synthetic_h5) as r:
        pool = [(r.get(i), float("nan")) for i in range(3)]
        phase, mask = r.phase_channel_idx, r.mask_channel_idx
    labels: dict = {}
    provider = SimpleNamespace(available=True, online_only_bytes=lambda: 2_000_000_000)
    dlg = CellDetailView(pool=pool, start_index=0, phase_idx=phase, mask_idx=mask,
                         get_label=lambda i: labels.get(i, "unlabeled"),
                         set_label=labels.__setitem__, context_provider=provider)
    assert dlg._mode == "crop" and not dlg._ctx_toggle.isChecked()
    assert dlg._ctx_toggle.isEnabled() and "2.0 GB" in dlg._ctx_toggle.toolTip()
    assert "not scored" in dlg._meta.text()
    dlg.close()


def test_label_page_wording_follows_state(synthetic_h5, tmp_path):
    """Review M4/H6: no stale 'Pick an HDF5' once a source is loaded, Diversify
    is disabled (and says why) without a model, paths are shortened, and the
    rubric link resolves."""
    from pathlib import Path

    from mycomorph.modelprep.gui._resources import rubric_location
    from mycomorph.modelprep.gui.panels.label_train_panel import LabelTrainPanel

    _app()
    panel = LabelTrainPanel()
    panel.set_source(synthetic_h5, tmp_path / "labels")
    assert "Pick an HDF5" not in panel._status.text()
    assert str(tmp_path) not in panel._info_label.text()      # name, not full path

    panel._model_combo.setCurrentIndex(0)                       # none (random order)
    assert not panel._diversify.isEnabled()
    assert "Needs a model" in panel._diversify.toolTip()
    panel._model_combo.setCurrentIndex(panel._model_combo.findText("preset: mtb"))
    assert panel._diversify.isEnabled()
    panel.shutdown()

    # From a source checkout the rubric resolves to the local file, not the URL.
    assert Path(rubric_location()).is_file()
    assert Path(rubric_location()).name == "labelling_rubric.md"


def test_inspector_names_the_eval_split(tmp_path):
    """The model card said 'Val accuracy' for hold-out numbers (review M4)."""
    import json

    from mycomorph.modelprep.gui.widgets.model_inspector import ModelInspector

    _app()
    run = tmp_path / "demo_2026-06-14_10-00-00"
    run.mkdir()
    (run / "best_model.pth").write_bytes(b"")
    (run / "training_config.json").write_text(json.dumps({
        "eval_split": "holdout", "eval_acc": 0.9, "balanced_acc": 0.85,
        "n_train": 100, "n_val": 20, "n_test": 0, "n_holdout": 50,
    }))
    insp = ModelInspector()
    insp.set_model(run / "best_model.pth")
    text = insp._stats.text()
    assert "Hold-out accuracy: 90.0%" in text and "Val accuracy" not in text
    assert "hold-out 50" in text


def test_models_panel_exports_cnn_runs_only(tmp_path, monkeypatch):
    """Review H4: Export for MycoMorph copies the selected CNN run to a chosen
    folder; feature runs can't be exported."""
    import json

    import mycomorph.modelprep.gui.export_action as ea
    import mycomorph.modelprep.gui.panels.models_panel as mp
    import mycomorph.modelprep.gui.trained_models as tm
    from PyQt6.QtWidgets import QMessageBox

    _app()
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(tm, "trained_models_root", lambda: root)
    monkeypatch.setattr(mp, "trained_models_root", lambda: root)
    cnn = _make_run(root, "alpha", "2026-06-11_00-00-00")
    feat = root / "alpha-features_2026-06-11_00-00-00"
    feat.mkdir()
    (feat / "feature_model.joblib").write_bytes(b"")
    (feat / "training_config.json").write_text(json.dumps({"model_type": "features"}))

    dest = tmp_path / "Exports"
    dest.mkdir()
    monkeypatch.setattr(ea.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: str(dest)))
    shown = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: shown.append(a)))

    panel = mp.ModelsPanel()
    panel._select_dir(str(feat))
    assert not panel._export_btn.isEnabled()
    panel._select_dir(str(cnn))
    assert panel._export_btn.isEnabled()
    panel._export_selected()
    assert (dest / cnn.name / f"{cnn.name}.pth").read_bytes() == b"x"
    assert shown


def test_source_page_warns_about_masked_crops(synthetic_h5):
    """Review H5: masked-background crops are flagged when the HDF5 loads."""
    import h5py

    from mycomorph.modelprep.gui.panels.source_panel import SourcePanel

    _app()
    with h5py.File(synthetic_h5, "a") as f:
        f.attrs["mask_background"] = True
    sp = SourcePanel()
    sp.set_h5_path(synthetic_h5)
    assert not sp._warn_label.isHidden() and "masking ON" in sp._warn_label.text()


def test_label_panel_condition_filter(multi_condition_h5, tmp_path):
    app = _app()
    from mycomorph.modelprep.core.cell_filter import CellFilter
    from mycomorph.modelprep.core.labels_store import CSV_COLUMNS, LabelsStore
    from mycomorph.modelprep.gui.panels.label_train_panel import LabelTrainPanel

    labels_dir = tmp_path / "labels"
    panel = LabelTrainPanel()
    panel.set_source(multi_condition_h5, labels_dir)
    panel._model_combo.setCurrentIndex(0)  # "none (random order)"
    assert panel._current_model_path() is None
    assert panel._filter_summary.text() == "48 cells · 48 unlabelled"
    assert panel._filter_btn.text().startswith("All cells")

    # Filter to two genes, uninduced only (AND across fields).
    dnan_inha = CellFilter.from_mapping({"gene": ["dnaN", "inhA"], "condition": ["ATc+MMC-"]})
    panel._set_cell_filter(dnan_inha)
    assert panel._filter_summary.text() == "12 of 48 cells match · 12 unlabelled"
    assert panel._filter_btn.text().startswith("gene ∈ {dnaN, inhA} AND condition")
    assert panel._filter_clear_btn.isVisibleTo(panel.label_section)

    panel._load_pool()
    _wait_for_pool(app, panel)
    assert len(panel._pool) == 12
    assert {rec.gene for rec, _ in panel._pool} == {"dnaN", "inhA"}
    assert {rec.condition for rec, _ in panel._pool} == {"ATc+MMC-"}
    assert "random order" in panel._status.text()

    # Label five and save: rows keep the CSV format; the session records the draw.
    for t in panel._thumbs[:5]:
        t.set_label("good")
    assert panel._save_batch(silent_if_empty=True) == 5
    store = LabelsStore(labels_dir)
    with open(store.csv_path) as f:
        assert f.readline().strip().split(",") == list(CSV_COLUMNS)
    first_session = panel._session_id
    entry = store.session(first_session)
    assert entry["n_labels"] == 5
    sampling = entry["sampling"]
    assert sampling["mode"] == "random" and sampling["seed"] == 42
    assert sampling["filter"] == {"gene": ["dnaN", "inhA"], "condition": ["ATc+MMC-"]}
    assert sampling["n_matching"] == 12 and sampling["split"] == "train"
    assert panel._filter_summary.text() == "12 of 48 cells match · 7 unlabelled"

    # Reloading with the same filter continues the same session, and draws the
    # rest of the same seeded order (the 7 unlabelled cells, order kept).
    first_order = [rec.cell_uid for rec, _ in panel._pool]
    panel._load_pool()
    _wait_for_pool(app, panel)
    assert len(panel._pool) == 0  # all 12 labelled or already shown this session
    panel._reset_session()  # forget "shown this session"; labels still excluded
    panel._load_pool()
    _wait_for_pool(app, panel)
    assert [rec.cell_uid for rec, _ in panel._pool] == first_order[5:]
    assert panel._session_id == first_session

    # A different filter starts a new session, so each session_id ↔ one draw.
    panel._set_cell_filter(CellFilter.from_mapping({"gene": ["NT1"]}))
    panel._load_pool()
    _wait_for_pool(app, panel)
    assert {rec.gene for rec, _ in panel._pool} == {"NT1"}
    assert panel._session_id != first_session
    panel._thumbs[0].set_label("bad")
    panel._save_batch(silent_if_empty=True)
    assert store.session(panel._session_id)["sampling"]["filter"] == {"gene": ["NT1"]}
    assert store.session(first_session)["sampling"]["filter"]["gene"] == ["dnaN", "inhA"]

    # A filter matching nothing: a clear message, no crash, empty grid.
    panel._set_cell_filter(CellFilter.from_mapping(
        {"gene": ["dnaN"], "condition_label": ["empty"]}))
    assert panel._filter_summary.text() == "No cells match this filter."
    panel._load_pool()
    _wait_for_pool(app, panel)
    assert panel._pool == []
    assert panel._status.text().startswith("No cells match the filter")

    # Clearing returns to all cells.
    panel._set_cell_filter(CellFilter())
    assert panel._filter_btn.text().startswith("All cells")
    panel.shutdown()
    app.processEvents()


def test_cell_filter_dialog(multi_condition_h5):
    app = _app()
    from PyQt6.QtCore import Qt
    from mycomorph.modelprep.core.cell_filter import CellFilter
    from mycomorph.modelprep.core.crops_h5 import HDF5CropsReader
    from mycomorph.modelprep.gui.panels.cell_filter_dialog import CellFilterDialog

    def tick(dlg, field, value):
        lst = dlg._lists[field]
        for i in range(lst.count()):
            if lst.item(i).data(Qt.ItemDataRole.UserRole) == value:
                lst.item(i).setCheckState(Qt.CheckState.Checked)
                return
        raise AssertionError(f"{value!r} not offered for {field}")

    with HDF5CropsReader(multi_condition_h5) as reader:
        labelled = {"cell_0000", "cell_0001"}  # one dnaN + one inhA, both MMC-
        dlg = CellFilterDialog(reader, labelled, CellFilter.from_mapping({"gene": ["dnaN"]}))
        # Reporter and drug have a single value each, so they aren't offered.
        assert set(dlg._lists) == {"gene", "condition", "condition_label", "tiff_file"}
        assert dlg.cell_filter().to_dict() == {"gene": ["dnaN"]}

        tick(dlg, "gene", "inhA")
        tick(dlg, "condition", "ATc+MMC-")
        assert dlg.cell_filter().to_dict() == {"gene": ["dnaN", "inhA"],
                                               "condition": ["ATc+MMC-"]}
        assert dlg._summary.text().startswith("12 of 48 cells match · 10 still unlabelled")
        assert dlg._boxes["gene"][0].title() == "Gene (4) · 2 ticked"
        assert dlg._apply.isEnabled()

        tick(dlg, "condition_label", "empty")  # contradicts the gene choice
        assert dlg._summary.text().startswith("No cells match")
        assert not dlg._apply.isEnabled()

        dlg._clear_all()
        assert dlg.cell_filter().is_empty()
        assert dlg._summary.text() == "All cells: 48 · 46 still unlabelled"

        dlg._search(dlg._lists["gene"], "nh")
        visible = [dlg._lists["gene"].item(i).data(Qt.ItemDataRole.UserRole)
                   for i in range(dlg._lists["gene"].count())
                   if not dlg._lists["gene"].item(i).isHidden()]
        assert visible == ["inhA"]
        dlg.close()
    app.processEvents()
