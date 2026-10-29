"""Classifier-model inspector, shared by MycoMorph's Classify page and ModelPrep.

Given a path to a trained model (a CNN ``.pth`` or a feature-model
``.joblib``), looks in the same directory for:
  - ``metrics.json``         → raw ROC/PR curves (live plot with an operating point)
  - ``roc_pr_curves.png``    → fallback static image
  - ``training_config.json`` → summary stats (eval split / acc / AUC / tuned threshold)

Both trainers compute the curves on P(bad) (class "bad" is the positive
class), so the operating point is always a P(bad) cutoff:

- once ``set_threshold`` has been called (MycoMorph's Classify page), the
  cutoff the Classify stage actually applies: the model's tuned
  ``decision_threshold`` when that's in use, otherwise 1 - max(confidence, 0.5)
  for argmax + the P(good) slider;
- until then (ModelPrep's train and Models pages), the model's own decision
  threshold from ``metrics.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from mycomorph.core.cell_quality_classifier import (
    effective_p_bad_cutoff,
    read_decision_threshold,
)


class ModelInspector(QWidget):
    """Renders classifier metadata for the currently selected model."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        empty_title: str = "No classifier selected",
        empty_hint: str = "Rule-based filtering only (edge / debris).",
    ) -> None:
        super().__init__(parent)
        self._empty_title = empty_title
        self._empty_hint = empty_hint
        self._model_dir: Path | None = None
        self._model_path: Path | None = None
        self._metrics: dict | None = None
        # The Classify rule: P(good) slider + optional tuned P(bad) cutoff.
        # Ignored (the model's own threshold is marked) until set_threshold.
        self._threshold = 0.5
        self._decision_threshold: float | None = None
        self._follow_rule = False
        self._canvas = None
        self._fig = None
        self._build_ui()
        self.set_model(None)

    # ------------------------------------------------------------------ setup

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self._header = QLabel()
        self._header.setObjectName("sectionHeader")
        self._header.setWordWrap(True)
        root.addWidget(self._header)

        self._stats = QLabel()
        self._stats.setObjectName("muted")
        self._stats.setWordWrap(True)
        root.addWidget(self._stats)

        # Live plot (matplotlib) — compact (~25% of the original training
        # figure). Imported lazily so a headless/agg-less env still loads the
        # panel — it just falls back to the PNG / text.
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self._fig = Figure(figsize=(4.6, 2.0), tight_layout=True, facecolor="#ffffff")
            self._canvas = FigureCanvasQTAgg(self._fig)
            # Resizable (max 660) so a narrow card doesn't force a wide
            # minimum width on the whole window.
            self._canvas.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            self._canvas.setFixedHeight(220)
            self._canvas.setMinimumWidth(260)
            self._canvas.setMaximumWidth(660)
            root.addWidget(self._canvas, alignment=Qt.AlignmentFlag.AlignLeft)
        except Exception:  # noqa: BLE001
            self._canvas = None

        # Static image fallback — scaled down to match.
        self._png_label = QLabel()
        self._png_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._png_label.setFixedHeight(220)
        self._png_scroll = QScrollArea()
        self._png_scroll.setWidgetResizable(False)
        self._png_scroll.setFrameShape(self._png_scroll.Shape.NoFrame)
        self._png_scroll.setFixedHeight(240)
        self._png_scroll.setMaximumWidth(680)
        self._png_scroll.setWidget(self._png_label)
        root.addWidget(self._png_scroll, alignment=Qt.AlignmentFlag.AlignLeft)

    # ----------------------------------------------------------------- public

    def set_model(self, model_path: Path | None) -> None:
        """Point the inspector at a model file (or None = no model)."""
        if model_path is None:
            self._model_dir = None
            self._model_path = None
            self._metrics = None
            self._header.setText(self._empty_title)
            self._stats.setText(self._empty_hint)
            if self._canvas is not None:
                self._canvas.hide()
            self._png_scroll.hide()
            return

        self._model_path = Path(model_path)
        self._model_dir = self._model_path.parent
        self._load_metadata()
        self._refresh()

    def set_threshold(self, t: float, decision_threshold: float | None = None) -> None:
        """Set the Classify rule to mark on the curves.

        ``t`` is the P(good) confidence slider; ``decision_threshold`` is the
        P(bad) cutoff when the model's tuned threshold is in use (None = the
        default argmax + slider rule).
        """
        self._threshold = float(t)
        self._decision_threshold = decision_threshold
        self._follow_rule = True
        self._refresh_live_plot()

    def marker_cutoff(self) -> float:
        """The P(bad) value the operating-point marker sits at."""
        if self._follow_rule:
            return effective_p_bad_cutoff(self._threshold, self._decision_threshold)
        own = (self._metrics or {}).get("decision_threshold")
        try:
            return float(own)
        except (TypeError, ValueError):
            return effective_p_bad_cutoff(0.5)

    # ---------------------------------------------------------------- loading

    def _load_metadata(self) -> None:
        assert self._model_dir is not None
        self._metrics = None
        metrics_path = self._model_dir / "metrics.json"
        if metrics_path.exists():
            try:
                self._metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self._metrics = None

    # ---------------------------------------------------------------- render

    def _refresh(self) -> None:
        if self._model_dir is None:
            return

        name = self._model_dir.name
        cfg_path = self._model_dir / "training_config.json"
        stats_lines: list[str] = []
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                split = cfg.get("eval_split", "val")
                eval_acc = cfg.get("eval_acc")
                best = cfg.get("best_val_acc")
                auc = cfg.get("roc_auc")
                n_train, n_val, n_test = cfg.get("n_train"), cfg.get("n_val"), cfg.get("n_test")
                n_holdout = cfg.get("n_holdout")
                bal = cfg.get("balanced_acc")
                classes = cfg.get("class_names")
                if eval_acc is not None and eval_acc == eval_acc:
                    # Name the set the number was measured on (the frozen
                    # hold-out, an internal test split, or validation).
                    label = {"holdout": "Hold-out", "test": "Test"}.get(split, "Val")
                    stats_lines.append(f"{label} accuracy: {eval_acc:.1%}")
                    if bal is not None and bal == bal:
                        stats_lines.append(f"balanced {bal:.1%}")
                elif best is not None:
                    stats_lines.append(f"Best val accuracy: {best:.1%}")
                if auc is not None and auc == auc:
                    stats_lines.append(f"AUC {auc:.2f}")
                sizes = [f"{k} {v:,}" for k, v in (("train", n_train), ("val", n_val),
                                                   ("test", n_test), ("hold-out", n_holdout)) if v]
                if sizes:
                    stats_lines.append(" / ".join(sizes))
                if classes:
                    stats_lines.append(f"classes: {', '.join(classes)}")
            except Exception:  # noqa: BLE001
                pass
        tuned = read_decision_threshold(self._model_path)
        if tuned is not None:
            stats_lines.append(f"tuned threshold: keep if P(bad) < {tuned:.3f}")

        self._header.setText(f"Model: {name}")
        self._stats.setText("  ·  ".join(stats_lines) if stats_lines else "")

        # Prefer the live plot if raw curves are available.
        if self._canvas is not None and self._metrics and self._metrics.get("binary"):
            self._canvas.show()
            self._png_scroll.hide()
            self._refresh_live_plot()
        else:
            png = self._model_dir / "roc_pr_curves.png"
            if png.exists():
                pix = QPixmap(str(png))
                pix = pix.scaledToHeight(
                    self._png_label.height(),
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._png_label.setPixmap(pix)
                self._png_label.setFixedWidth(pix.width())
                self._png_scroll.show()
            else:
                self._png_scroll.hide()
            if self._canvas is not None:
                self._canvas.hide()

    def _refresh_live_plot(self) -> None:
        if self._canvas is None or not self._metrics or not self._metrics.get("binary"):
            return
        b = self._metrics["binary"]
        fpr = np.asarray(b["roc"]["fpr"])
        tpr = np.asarray(b["roc"]["tpr"])
        thr_r = np.asarray(b["roc"]["thresholds"])
        auc_v = b["roc"]["auc"]

        prec = np.asarray(b["pr"]["precision"])
        rec = np.asarray(b["pr"]["recall"])
        thr_p = np.asarray(b["pr"]["thresholds"])
        ap_v = b["pr"]["ap"]
        base = b["pr"]["baseline"]

        # Curves are on P(bad), so mark a P(bad) cutoff, never the P(good)
        # slider value itself.
        t = self.marker_cutoff()
        # Nearest operating point on each curve. PR thresholds have
        # len = len(precision) - 1.
        idx_roc = int(np.argmin(np.abs(thr_r - t))) if len(thr_r) else 0
        idx_pr = int(np.argmin(np.abs(thr_p - t))) if len(thr_p) else 0
        idx_pr = min(idx_pr, len(prec) - 1)

        self._fig.clear()
        ax_roc, ax_pr = self._fig.subplots(1, 2)
        for ax in (ax_roc, ax_pr):
            ax.tick_params(labelsize=7)

        ax_roc.plot(fpr, tpr, color="#2166ac", lw=1.2, label=f"AUC {auc_v:.2f}")
        ax_roc.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.4)
        if len(fpr):
            ax_roc.plot(fpr[idx_roc], tpr[idx_roc], "o", ms=5, color="#e0402b",
                        label=f"P(bad) cut {t:.2f}")
        if "optimal_threshold" in b and len(thr_r):
            idx_opt = int(np.argmin(np.abs(thr_r - b["optimal_threshold"])))
            ax_roc.plot(fpr[idx_opt], tpr[idx_opt], "^", ms=5, color="#2ca02c",
                        label=f"opt {b['optimal_threshold']:.2f}")
        ax_roc.set_xlabel("FPR", fontsize=8); ax_roc.set_ylabel("TPR", fontsize=8)
        ax_roc.set_title("ROC (positive = bad)", fontsize=8)
        ax_roc.grid(True, alpha=0.2)
        ax_roc.legend(loc="lower right", fontsize=7, frameon=False)

        ax_pr.plot(rec, prec, color="#b2182b", lw=1.2, label=f"AP {ap_v:.2f}")
        ax_pr.axhline(base, color="k", ls="--", lw=0.7, alpha=0.4)
        if len(prec):
            ax_pr.plot(rec[idx_pr], prec[idx_pr], "o", ms=5, color="#e0402b",
                       label=f"P(bad) cut {t:.2f}")
        ax_pr.set_xlabel("Recall", fontsize=8); ax_pr.set_ylabel("Precision", fontsize=8)
        ax_pr.set_title("PR (positive = bad)", fontsize=8)
        ax_pr.grid(True, alpha=0.2)
        ax_pr.legend(loc="lower left", fontsize=7, frameon=False)

        self._canvas.draw_idle()
