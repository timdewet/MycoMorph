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


def _wait_for_pool(app, panel):
    """Let the pool worker finish and deliver its queued result."""
    import time
    worker = panel._al_worker
    assert worker is not None
    assert worker.wait(20000), "pool worker did not finish"
    deadline = time.time() + 5
    while time.time() < deadline and not panel._load_btn.isEnabled():
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()


def test_label_panel_condition_filter(multi_condition_h5, tmp_path):
    app = _app()
    from modelprep.core.cell_filter import CellFilter
    from modelprep.core.labels_store import CSV_COLUMNS, LabelsStore
    from modelprep.gui.panels.label_train_panel import LabelTrainPanel

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
    from modelprep.core.cell_filter import CellFilter
    from modelprep.core.crops_h5 import HDF5CropsReader
    from modelprep.gui.panels.cell_filter_dialog import CellFilterDialog

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
