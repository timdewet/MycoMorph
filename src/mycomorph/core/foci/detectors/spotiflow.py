"""Spotiflow (Dominguez Mantes et al., Nature Methods 2025) detector wrapper.

`Paper <https://www.nature.com/articles/s41592-025-02662-x>`__ /
`GitHub <https://github.com/weigertlab/spotiflow>`__.

Uses the pretrained ``general`` model by default (a single set of
weights generalising across spot-detection datasets, including
microscopy / FISH). For best results on bacterial DnaN/ImuB foci the
user could later finetune on their own annotations — out of scope here.

The model is downloaded on first use (~tens of MB) and cached in the
user's HF cache. Detection is GPU-accelerated if PyTorch can see a CUDA
or MPS device; otherwise it falls back to CPU.

Spotiflow returns sub-pixel positions and per-spot probabilities. We
keep its peak ``probability`` as the ``intensity`` field of ``Focus`` so
score-based filtering still works.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .._types import DetectorOpts, Focus
from ._common import build_focus, label_at, normalise_image


class SpotiflowDetector:
    name = "spotiflow"

    def __init__(self) -> None:
        self._model = None

    def _load_model(self, model_name: str):
        try:
            from spotiflow.model import Spotiflow
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "spotiflow not installed. `pip install spotiflow` to use this detector."
            ) from exc

        if self._model is None:
            self._model = Spotiflow.from_pretrained(model_name)
        return self._model

    def fit(self, run_stack):
        # No per-run fitting — the pretrained model is used as-is.
        return None

    def detect(
        self,
        image: np.ndarray,
        labeled_mask: Optional[np.ndarray] = None,
        opts: Optional[DetectorOpts] = None,
    ) -> list[Focus]:
        opts = opts or DetectorOpts()
        model = self._load_model(opts.spotiflow_model)

        norm = normalise_image(image)
        # Spotiflow.predict returns (points, probabilities, _, _) or similar
        # depending on version. We use the common 2-tuple API.
        try:
            points, details = model.predict(
                norm.astype(np.float32),
                min_distance=opts.spotiflow_min_distance,
                exclude_border=False,
                verbose=False,
            )
        except TypeError:
            # Older spotiflow API
            points, details = model.predict(norm.astype(np.float32))

        if points is None or len(points) == 0:
            return []
        # ``points`` is (N, 2) in (y, x) order; ``details`` is a dict that
        # usually has ``probabilities``.
        probs = None
        if isinstance(details, dict):
            probs = details.get("probabilities") or details.get("prob")

        out: list[Focus] = []
        for i, (y, x) in enumerate(points):
            prob = float(probs[i]) if probs is not None else 1.0
            # Spotiflow already gives sub-pixel y, x; pass them through
            # build_focus with refine=False to keep its localisation but
            # still get SNR + cell_label assignment.
            focus = build_focus(
                image,
                float(y), float(x),
                init_sigma=1.5,
                labeled_mask=labeled_mask,
                refine=False,
                snr_min=opts.snr_min,
                intensity_override=prob,
            )
            if focus is None:
                # Fall back: keep the focus even at low SNR but tag it.
                focus = Focus(
                    y=float(y), x=float(x), sigma=1.5,
                    intensity=prob, snr=0.0,
                    cell_label=label_at(labeled_mask, float(y), float(x)),
                )
            out.append(focus)
        return out
