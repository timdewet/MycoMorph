"""ModelPrep's tuned decision threshold in the Classify stage.

ModelPrep stores ``decision_threshold`` in the ``training_config.json`` next
to the ``.pth``: a Youden-J cutoff on P(bad) (class index 1). When the user
opts in, Classify keeps a cell iff P(bad) < t instead of argmax + the P(good)
confidence slider.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mycomorph.core import api
from mycomorph.core import cell_quality_classifier as cqc
from mycomorph.core.cell_quality_classifier import (
    classify_and_filter_mask,
    effective_p_bad_cutoff,
    read_decision_threshold,
)
from mycomorph.core.provenance import checkpoint_signature


def _model_dir(tmp_path: Path, config: dict | str | None) -> Path:
    """A model folder with a dummy .pth and (optionally) a sidecar config."""
    d = tmp_path / "model"
    d.mkdir()
    pth = d / "best_model.pth"
    pth.write_bytes(b"not a real model")
    if config is not None:
        text = config if isinstance(config, str) else json.dumps(config)
        (d / "training_config.json").write_text(text, encoding="utf-8")
    return pth


# ── Reading the sidecar ───────────────────────────────────────────────────────


def test_reads_decision_threshold_from_sidecar(tmp_path):
    pth = _model_dir(tmp_path, {"decision_threshold": 0.42, "trainer": "modelprep"})
    assert read_decision_threshold(pth) == pytest.approx(0.42)


@pytest.mark.parametrize(
    "config",
    [
        None,                                   # no training_config.json
        {"best_val_acc": 0.9},                  # key missing (MycoMorph trainer)
        {"decision_threshold": None},
        {"decision_threshold": "abc"},
        {"decision_threshold": 0.0},
        {"decision_threshold": 1.0},
        {"decision_threshold": 1.5},
        {"decision_threshold": -0.2},
        '{"decision_threshold": Infinity}',     # roc_curve's +inf sentinel
        '{"decision_threshold": NaN}',
        "[0.4]",                                # not a dict
        "{not json",
    ],
)
def test_unusable_sidecar_gives_none(tmp_path, config):
    pth = _model_dir(tmp_path, config)
    assert read_decision_threshold(pth) is None


def test_no_model_gives_none():
    assert read_decision_threshold(None) is None


# ── The P(bad) cutoff each rule applies ──────────────────────────────────────


@pytest.mark.parametrize(
    "confidence,tuned,expected",
    [
        (0.5, None, 0.5),
        (0.7, None, 0.3),
        (0.3, None, 0.5),     # below 0.5 the slider changes nothing
        (0.0, None, 0.5),
        (0.7, 0.42, 0.42),    # tuned threshold ignores the slider
        (0.5, 0.65, 0.65),
    ],
)
def test_effective_p_bad_cutoff(confidence, tuned, expected):
    assert effective_p_bad_cutoff(confidence, tuned) == pytest.approx(expected)


# ── classify_and_filter_mask ─────────────────────────────────────────────────


# P(bad) for cells 1, 2, 3 of the ``labeled_mask_3cells`` fixture.
P_BAD = {1: 0.30, 2: 0.45, 3: 0.60}


class _FakeClassifier:
    def predict_batch(self, crops, batch_size=64):
        # classify_and_filter_mask passes cells in sorted label order.
        assert len(crops) == len(P_BAD)
        p_bad = np.array([P_BAD[k] for k in sorted(P_BAD)], dtype=np.float32)
        probs = np.stack([1.0 - p_bad, p_bad], axis=1)
        return probs.argmax(axis=1), probs


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    pth = tmp_path / "fake.pth"
    pth.write_bytes(b"x")
    import threading

    monkeypatch.setattr(
        cqc, "_cached_classifier",
        lambda *_a, **_k: (_FakeClassifier(), threading.Lock()),
    )
    return pth


def _kept(labeled_mask, fake_model, **kwargs):
    image = np.random.default_rng(0).random((1, *labeled_mask.shape)).astype(np.float32)
    filtered, report = classify_and_filter_mask(
        labeled_mask, image, phase_channel=0, model_path=fake_model,
        use_rules=False, verbose=False, **kwargs,
    )
    return set(np.unique(filtered).tolist()) - {0}, report


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        # Default rule: argmax = good and P(good) >= confidence.
        ({"confidence_threshold": 0.5}, {1, 2}),
        ({"confidence_threshold": 0.6}, {1}),
        ({"confidence_threshold": 0.2}, {1, 2}),
        # Tuned rule: keep iff P(bad) < t. Cell 3 (P(bad) 0.6, argmax "bad")
        # is kept at t = 0.65; the confidence slider plays no part.
        ({"decision_threshold": 0.40}, {1}),
        ({"decision_threshold": 0.50}, {1, 2}),
        ({"decision_threshold": 0.65}, {1, 2, 3}),
        ({"decision_threshold": 0.65, "confidence_threshold": 0.99}, {1, 2, 3}),
        ({"decision_threshold": 0.20}, set()),
    ],
)
def test_filter_rule(labeled_mask_3cells, fake_model, kwargs, expected):
    kept, report = _kept(labeled_mask_3cells, fake_model, **kwargs)
    assert kept == expected
    assert report["removed_cnn"] == 3 - len(expected)
    # The kept set is exactly the cells under the rule's P(bad) cutoff, the
    # same number the model inspector marks on the curves.
    cut = effective_p_bad_cutoff(
        kwargs.get("confidence_threshold", 0.5), kwargs.get("decision_threshold"),
    )
    assert kept == {k for k, p in P_BAD.items() if p < cut}


def test_tuned_rule_reports_the_class_it_decided(labeled_mask_3cells, fake_model):
    _kept_set, report = _kept(labeled_mask_3cells, fake_model, decision_threshold=0.65)
    name, conf = report["details"][3]
    assert name == "good"
    assert conf == pytest.approx(0.40)        # P(good) of the class it was given
    assert report["decision_threshold"] == pytest.approx(0.65)

    _kept_set, report = _kept(labeled_mask_3cells, fake_model, decision_threshold=0.40)
    assert report["details"][2] == ("bad", pytest.approx(0.45))


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.1, float("inf"), float("nan"), "x"])
def test_invalid_decision_threshold_raises(labeled_mask_3cells, fake_model, bad):
    with pytest.raises(ValueError, match="decision_threshold"):
        _kept(labeled_mask_3cells, fake_model, decision_threshold=bad)


# ── Stage options and provenance ──────────────────────────────────────────────


def test_default_classify_signature_unchanged_by_new_field(tmp_path):
    """Outputs signed before decision_threshold existed must stay reusable."""
    from dataclasses import asdict

    src = tmp_path / "in.tif"
    src.write_bytes(b"x")
    opts = api.ClassifyOpts()
    payload = opts.signature_payload()
    assert "decision_threshold" not in payload

    legacy = {k: v for k, v in asdict(opts).items() if k != "decision_threshold"}
    assert payload == legacy
    sig_new = checkpoint_signature(stage="classify", input_path=src, options=opts)
    sig_old = checkpoint_signature(stage="classify", input_path=src, options=legacy)
    assert sig_new["digest"] == sig_old["digest"]

    tuned = api.ClassifyOpts(decision_threshold=0.42)
    assert tuned.signature_payload()["decision_threshold"] == pytest.approx(0.42)
    sig_tuned = checkpoint_signature(stage="classify", input_path=src, options=tuned)
    assert sig_tuned["digest"] != sig_new["digest"]


def test_classify_filter_tiff_passes_decision_threshold(monkeypatch, tmp_path):
    from mycomorph.core.cellpose_pipeline import save_hyperstack

    source = tmp_path / "segmented.tif"
    data = np.zeros((1, 2, 32, 32), dtype=np.uint16)
    data[0, 0, 10:14, 10:14] = 100
    data[0, 1, 10:14, 10:14] = 1
    save_hyperstack(data, source, "c", ["fov_000"], ["Phase", "BinaryMask"], 2.0)
    seen: dict = {}

    def fake_filter(*, labeled_mask, **kwargs):
        seen.update(kwargs)
        return labeled_mask, {"total_cells": 1, "kept": 1, "removed_edge": 0,
                              "removed_debris": 0, "removed_cnn": 0}

    monkeypatch.setattr(cqc, "classify_and_filter_mask", fake_filter)
    messages: list[str] = []
    api.classify_filter_tiff(
        source, tmp_path / "out.tif", phase_channel=0,
        opts=api.ClassifyOpts(model_path=_model_dir(tmp_path, None), decision_threshold=0.42),
        progress_cb=lambda _f, m: messages.append(m),
    )
    assert seen["decision_threshold"] == pytest.approx(0.42)
    assert any("P(bad) < 0.420" in m for m in messages)


def test_live_preview_cache_key_includes_decision_threshold():
    from mycomorph.gui.widgets.live_preview.cache import classify_key

    seg_k = ("segment", "x")
    assert (classify_key(seg_k, api.ClassifyOpts())
            != classify_key(seg_k, api.ClassifyOpts(decision_threshold=0.42)))


# ── GUI: Classify panel and model inspector ───────────────────────────────────


def _metrics(model_pth: Path) -> None:
    """A metrics.json whose ROC thresholds are on P(bad), as both trainers write."""
    thr = [float("inf"), 0.9, 0.7, 0.5, 0.3, 0.1]
    fpr = [0.0, 0.05, 0.10, 0.20, 0.40, 1.0]
    tpr = [0.0, 0.40, 0.60, 0.75, 0.90, 1.0]
    metrics = {
        "binary": {
            "roc": {"fpr": fpr, "tpr": tpr, "thresholds": thr, "auc": 0.85},
            "pr": {"precision": [0.5, 0.6, 0.8, 1.0], "recall": [1.0, 0.9, 0.6, 0.0],
                   "thresholds": [0.1, 0.3, 0.7], "ap": 0.8, "baseline": 0.5},
            "optimal_threshold": 0.5,
        }
    }
    (model_pth.parent / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def _roc_marker(inspector) -> tuple[float, float]:
    ax_roc = inspector._fig.axes[0]
    line = next(ln for ln in ax_roc.lines if ln.get_label().startswith("P(bad) cut"))
    return float(line.get_xdata()[0]), float(line.get_ydata()[0])


def test_inspector_marks_p_bad_cutoff_not_slider_value(qtbot, tmp_path):
    pytest.importorskip("PyQt6")
    from mycomorph.gui.widgets.model_inspector import ModelInspector

    pth = _model_dir(tmp_path, {"decision_threshold": 0.3})
    _metrics(pth)
    insp = ModelInspector()
    qtbot.addWidget(insp)
    insp.set_model(pth)

    # Slider P(good) >= 0.7  ->  reject P(bad) > 0.3  ->  ROC point at thr 0.3
    # (fpr 0.40, tpr 0.90). Plotting the raw 0.7 would land on (0.10, 0.60).
    insp.set_threshold(0.7)
    assert _roc_marker(insp) == pytest.approx((0.40, 0.90))

    # Tuned threshold: the marker sits at t itself.
    insp.set_threshold(0.7, decision_threshold=0.9)
    assert _roc_marker(insp) == pytest.approx((0.05, 0.40))
    assert "P(bad) < 0.300" in insp._stats.text()


def test_classify_panel_offers_tuned_threshold_opt_in(qtbot, tmp_path):
    pytest.importorskip("PyQt6")
    from mycomorph.gui.panels.stage_panels import ClassifyPanel

    pth = _model_dir(tmp_path, {"decision_threshold": 0.42})
    panel = ClassifyPanel()
    qtbot.addWidget(panel)
    panel.preset.setCurrentText("custom…")
    panel.custom_path.setText(str(pth))

    # Offered, shown, but off by default: the default rule is unchanged.
    assert panel.use_model_threshold.isEnabled()
    assert not panel.use_model_threshold.isChecked()
    assert "0.420" in panel.use_model_threshold.text()
    assert panel.opts().decision_threshold is None
    assert panel.confidence.isEnabled()

    panel.use_model_threshold.setChecked(True)
    assert panel.opts().decision_threshold == pytest.approx(0.42)
    assert not panel.confidence.isEnabled()
    assert panel.inspector._decision_threshold == pytest.approx(0.42)

    state = panel.state()
    assert state["use_model_threshold"] is True
    other = ClassifyPanel()
    qtbot.addWidget(other)
    other.restore_state(state)
    assert other.opts().decision_threshold == pytest.approx(0.42)


def test_classify_panel_without_sidecar_keeps_default_rule(qtbot, tmp_path):
    pytest.importorskip("PyQt6")
    from mycomorph.gui.panels.stage_panels import ClassifyPanel

    pth = _model_dir(tmp_path, None)
    panel = ClassifyPanel()
    qtbot.addWidget(panel)
    panel.restore_state({
        "preset": "custom…", "custom_path": str(pth), "use_model_threshold": True,
    })
    assert not panel.use_model_threshold.isEnabled()
    assert not panel.use_model_threshold.isChecked()   # no greyed-out tick
    assert panel.opts().decision_threshold is None
    assert panel.confidence.isEnabled()


def test_switching_to_a_model_without_sidecar_drops_the_opt_in(qtbot, tmp_path):
    pytest.importorskip("PyQt6")
    from mycomorph.gui.panels.stage_panels import ClassifyPanel

    tuned = _model_dir(tmp_path, {"decision_threshold": 0.42})
    plain = tmp_path / "plain" / "best_model.pth"
    plain.parent.mkdir()
    plain.write_bytes(b"x")
    panel = ClassifyPanel()
    qtbot.addWidget(panel)
    panel.preset.setCurrentText("custom…")
    panel.custom_path.setText(str(tuned))
    panel.use_model_threshold.setChecked(True)

    panel.custom_path.setText(str(plain))
    assert not panel.use_model_threshold.isChecked()
    assert panel.opts().decision_threshold is None
    assert panel.confidence.isEnabled()
    assert panel.inspector._decision_threshold is None
