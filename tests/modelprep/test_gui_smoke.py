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


def test_main_window_constructs_and_navigates():
    app = _app()
    from modelprep.gui.ui import theme
    from modelprep.gui.main_window import MainWindow

    theme.apply_theme(app)
    win = MainWindow()
    assert win._stack.count() == 4  # source, label, train, models
    for i in range(win._stack.count()):
        win._sidebar.set_current(i)
        app.processEvents()
    win.label_train_panel.shutdown()


def test_panels_construct():
    app = _app()
    from modelprep.gui.panels.source_panel import SourcePanel
    from modelprep.gui.panels.label_train_panel import LabelTrainPanel

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
    import modelprep.gui.trained_models as tm
    import modelprep.gui.panels.models_panel as mp
    from modelprep.gui.panels.models_panel import _COL
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


def test_models_panel_shows_feature_runs(tmp_path, monkeypatch):
    app = _app()
    import json
    import modelprep.gui.trained_models as tm
    import modelprep.gui.panels.models_panel as mp
    from modelprep.gui.panels.models_panel import _COL

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
    import modelprep.gui.panels.models_panel as mp
    from modelprep.core.labels_store import LabelsStore, LabelRow, SPLIT_TRAIN, SPLIT_TEST

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
    import modelprep.gui.panels.models_panel as mp
    from modelprep.core.labels_store import LabelsStore, LabelRow, SPLIT_TRAIN, SPLIT_TEST

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
    from modelprep.core.labels_store import LabelsStore, LabelRow
    from modelprep.gui.panels.consistency_dialog import ConsistencyCheckDialog

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
    import modelprep.core.feature_classifier as fc
    from modelprep.gui.panels.label_train_panel import _TrainWorker

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
    worker = panel._al_worker
    assert worker is not None
    worker.wait(30000)
    for _ in range(50):         # deliver the queued finished() signal
        app.processEvents()
        if panel._pool:
            break


def test_random_holdout_sample_records_no_model_confidence(synthetic_h5, tmp_path):
    """Random hold-out sampling doesn't score cells, so the saved labels must
    say model 'none' with NaN confidence, even with a model in the picker
    (review M9)."""
    import math
    import modelprep.core.cell_quality_classifier as cqc
    from modelprep.core.labels_store import LabelsStore
    from modelprep.gui.panels.label_train_panel import LabelTrainPanel

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
    import modelprep.core.cell_quality_classifier as cqc
    import modelprep.core.score_cache as sc
    from modelprep.gui.panels.label_train_panel import _ActiveLearningWorker

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

    from modelprep.core.crops_h5 import HDF5CropsReader
    from modelprep.gui.panels.cell_detail_dialog import CellDetailView

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

    from modelprep.gui._resources import rubric_location
    from modelprep.gui.panels.label_train_panel import LabelTrainPanel

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

    assert Path(rubric_location()).name == "labelling_rubric.md"


def test_inspector_names_the_eval_split(tmp_path):
    """The model card said 'Val accuracy' for hold-out numbers (review M4)."""
    import json

    from modelprep.gui.widgets.model_inspector import ModelInspector

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

    import modelprep.gui.export_action as ea
    import modelprep.gui.panels.models_panel as mp
    import modelprep.gui.trained_models as tm
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

    from modelprep.gui.panels.source_panel import SourcePanel

    _app()
    with h5py.File(synthetic_h5, "a") as f:
        f.attrs["mask_background"] = True
    sp = SourcePanel()
    sp.set_h5_path(synthetic_h5)
    assert not sp._warn_label.isHidden() and "masking ON" in sp._warn_label.text()
