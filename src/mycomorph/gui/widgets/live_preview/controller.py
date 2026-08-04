"""Orchestrator for the live preview.

The :class:`PreviewController` owns:

- a :class:`PreviewCache` of per-FOV stage results
- a single in-flight :class:`PreviewWorker`
- a :class:`QTimer` that debounces a flurry of option-change signals
  into a single render request

Triggers (sample/FOV picked, options changed, tab navigated) call
:meth:`request_render`. The controller decides which stages need to run
based on the live opts vs cached keys, spawns a fresh worker, and
funnels its stage outputs both into the cache and the canvas/panel
callbacks.

Keeps zero references to specific Qt widgets — the panel passes in a
small set of callbacks (``opts_providers``, ``set_phase``, ``set_mask``,
``progress_started`` / ``progress_finished``) so this object stays
testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from .cache import (
    CacheEntry,
    PreviewCache,
)
from .worker import (
    ClassifyPayload,
    FeaturesPayload,
    FocusPayload,
    PreviewWorker,
    RenderRequest,
    SegmentPayload,
)


# How long after the last option-change signal before we kick off a render.
DEFAULT_DEBOUNCE_MS = 300


@dataclass
class OptsProviders:
    """Callbacks the panel installs so the controller can read live opts."""

    focus_opts: Callable[[], Any] = lambda: None
    segment_opts: Callable[[], Any] = lambda: None
    classify_opts: Callable[[], Any] = lambda: None
    features_opts: Callable[[], Any] = lambda: None
    phase_channel: Callable[[], int] = lambda: 0
    # User-edited channel labels from the Input panel ("Phase", "GFP",
    # "RFP", …). Used to label the per-channel overlay rows and the
    # focus payload so the panel doesn't fall back to "C0", "C1".
    channel_labels: Callable[[], Optional[list[str]]] = lambda: None
    # Current segmentation ROI (x0, y0, x1, y1) in image pixel coords,
    # or ``None`` to process the entire image.
    roi: Callable[[], Optional[tuple[int, int, int, int]]] = lambda: None
    # Foci panels' opts — read by the post-processing pass that runs on
    # the main thread after the chain delivers focus/segment payloads.
    # Both default to ``None`` so the controller stays usable from
    # contexts that don't wire the new panels.
    fluor_norm_opts: Callable[[], Any] = lambda: None
    foci_det_opts: Callable[[], Any] = lambda: None
    # Inline-histogram thresholds: feature-name → minimum value. The
    # controller filters cached foci against these before drawing the
    # scatter, so threshold drags don't trigger re-detection.
    foci_thresholds: Callable[[], dict] = lambda: {}
    # Visibility toggle for the foci scatter overlay. Returning False
    # hides the layer without invalidating the cached detection so the
    # user can compare raw image vs. detection at the press of a button.
    foci_visible: Callable[[], bool] = lambda: True


@dataclass
class CanvasSink:
    """Callbacks invoked on the GUI thread as stage results land.

    ``set_phase`` and ``set_image_channels`` are called when focus
    completes. The panel owns the channel rendering logic — given
    raw ``(C, Y, X)`` arrays, it composes them into the canvas's
    per-layer ``ChannelLayer`` records based on the user's per-channel
    visibility / opacity / color settings.

    ``set_features_df`` lands the per-cell DataFrame from the features
    stage; the panel owns the per-cell text-label rendering.
    """

    set_phase: Callable[[Any], None] = lambda _img: None
    set_image_channels: Callable[[Any, Any, int], None] = (
        lambda _channels, _names, _phase_idx: None
    )
    set_mask: Callable[[Any, Any], None] = lambda _mask, _decisions: None
    set_features_df: Callable[[Any], None] = lambda _df: None
    set_labels: Callable[[Any], None] = lambda _labels: None
    clear: Callable[[], None] = lambda: None
    # Foci scatter overlays — used by the fluor_norm and foci_det
    # post-processing passes. Each callable is (name, color_rgba) for
    # add_foci_layer, (name, xs, ys, sizes) for set_foci_layer_data,
    # and (name, visible) for set_foci_layer_visible.
    add_foci_layer: Callable[[str, tuple], None] = lambda _n, _c: None
    set_foci_layer_data: Callable[[str, Any, Any, Any], None] = (
        lambda _n, _x, _y, _s: None
    )
    set_foci_layer_visible: Callable[[str, bool], None] = (
        lambda _n, _v: None
    )
    clear_foci_layers: Callable[[], None] = lambda: None
    # Push the detected foci's features DataFrame back to the panel so
    # the inline histograms can re-bin. Called once per detection pass
    # (not on threshold drags — those are pure scatter filters).
    set_foci_features: Callable[[Any], None] = lambda _df: None


@dataclass
class ProgressSink:
    """Callbacks for spinner/progress UI on the GUI thread."""

    started: Callable[[str], None] = lambda _stage: None
    finished: Callable[[], None] = lambda: None
    failed: Callable[[str, str], None] = lambda _stage, _msg: None


class PreviewController(QObject):
    """Owns cache + worker + debounce timer for the live preview."""

    # Forwarded to the panel for high-level UI feedback.
    renderStateChanged = pyqtSignal(bool)   # True when work is in flight

    def __init__(
        self,
        opts: OptsProviders,
        canvas: CanvasSink,
        progress: ProgressSink,
        debounce_ms: int = DEFAULT_DEBOUNCE_MS,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._opts = opts
        self._canvas = canvas
        self._progress = progress

        self._cache = PreviewCache()
        self._worker: Optional[PreviewWorker] = None
        self._pending_request: Optional[RenderRequest] = None

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(debounce_ms)
        self._debounce.timeout.connect(self._fire_pending)

        # Selection state — driven by panel.set_selection.
        self._tab: str = "segment"
        self._sample_path: Optional[Path] = None
        self._fov_index: int = 0
        # Tracks foci scatter-layer names currently in use so a render
        # pass on the foci_det tab can wipe stale layers (e.g. when the
        # user unticks a detector mid-session).
        self._foci_scatter_names: set[str] = set()

    # ----------------------------------------------------------------- API

    def set_selection(self, tab: str, sample_path: Optional[Path],
                      fov_index: int) -> None:
        """Update which FOV the controller is rendering for.

        On a real sample/FOV change we eagerly wipe the stale mask
        boundary overlay and per-cell labels so the canvas doesn't
        show last-FOV's segmentation while the new one is still
        processing. The phase plane is left in place until the new
        focus payload lands — clearing it would just produce a flash
        of black for the typically-brief focus step.
        """
        new_path = Path(sample_path) if sample_path is not None else None
        new_fov = int(fov_index)
        fov_changed = (new_path != self._sample_path) or (new_fov != self._fov_index)
        prev_tab = self._tab
        self._tab = tab
        self._sample_path = new_path
        self._fov_index = new_fov
        if fov_changed:
            self._canvas.set_mask(None, None)
            self._canvas.set_features_df(None)
        # Leaving a foci tab clears its scatter overlays so they don't
        # linger over the segment / features view.
        if prev_tab in ("fluor_norm", "foci_det") and tab not in (
            "fluor_norm", "foci_det"
        ):
            self._canvas.clear_foci_layers()
        self.request_render(reason="selection")
        # Repaint cached data immediately so foci-tab arrivals don't have
        # to wait for the worker chain (cache hits → worker emits nothing).
        if tab in ("fluor_norm", "foci_det"):
            self._redraw_foci_overlays()

    def request_render(self, reason: str = "") -> None:
        """Schedule a render, debounced; cancels any in-flight worker."""
        if self._sample_path is None:
            self._cancel_worker()
            self._canvas.clear()
            return
        # Build the request now so it captures the live opts at the moment
        # of the trigger, not at fire time. (If the user keeps moving a
        # slider, each trigger overwrites the pending request.)
        self._pending_request = self._build_request()
        self._debounce.start()

    def repaint_cached(self) -> None:
        """Repaint the canvas from the cached entry for the current FOV.

        Called after a UI toggle (e.g. "Show classification colors") so
        the change shows up even if no stage actually re-runs.
        """
        if self._sample_path is None:
            self._canvas.clear()
            return
        entry = self._cache.get(self._sample_path, self._fov_index)
        if entry.phase is not None:
            self._canvas.set_phase(entry.phase)
        if entry.image_channels is not None:
            phase_idx = (
                entry.resolved_phase_channel
                if entry.resolved_phase_channel is not None
                else (self._opts.phase_channel() if isinstance(self._opts.phase_channel(), int) else 0)
            )
            self._canvas.set_image_channels(
                entry.image_channels,
                entry.channel_names or [],
                int(phase_idx),
            )
        if entry.mask is not None:
            self._canvas.set_mask(entry.mask, entry.decisions)
        else:
            self._canvas.set_mask(None, None)
        if entry.features_df is not None:
            self._canvas.set_features_df(entry.features_df)

    def invalidate_cache(self, levels: tuple[str, ...] = ("segment",)) -> None:
        """Drop selected stage results across all cache entries.

        Currently used to reset cache from external "channels changed"
        / "phase channel changed" events. Most stage invalidation is
        implicit via the keying — set_selection / option changes pick a
        new key and the cache miss handles the rest.
        """
        for entry in list(self._cache._entries.values()):
            for lvl in levels:
                if lvl == "focus":
                    entry.focus_key = None
                    entry.phase = None
                    entry.image_channels = None
                if lvl == "segment":
                    entry.segment_key = None
                    entry.mask = None
                if lvl == "classify":
                    entry.classify_key = None
                    entry.decisions = None
                if lvl == "features":
                    entry.features_key = None
                    entry.features_df = None

    # ------------------------------------------------------------ internals

    def _build_request(self) -> RenderRequest:
        # Run the full chain regardless of which tab the user is on,
        # so segmentation + classification appear on the Focus tab
        # too — the user shouldn't have to switch tabs to see whether
        # their focus options yield masks worth keeping. The cache
        # ensures that tweaks to (say) only the focus options don't
        # silently re-run segment + classify on every keystroke.
        target = "features" if self._tab == "features" else "classify"
        # Pick the focus shim based on the sample's file type:
        # focused TIFFs on disk → use the cheap disk-load shim,
        # raw CZIs → run the in-memory single-FOV focus chain. This
        # lets the user preview segmentation off raw CZIs before
        # they've ever run the real Focus stage.
        use_disk_focus = self._is_focused_tiff(self._sample_path)

        classify_opts = self._opts.classify_opts() if target in ("classify", "features") else None
        # ``phase_channel`` may be ``None`` ("auto") or a string label —
        # the worker resolves it to a concrete int from the image data
        # the first time focus loads channels for a given sample.
        raw_phase = self._opts.phase_channel()
        phase_channel: Any = raw_phase if isinstance(raw_phase, (int, str)) else None
        return RenderRequest(
            sample_path=self._sample_path,
            fov_index=self._fov_index,
            target_stage=target,
            phase_channel=phase_channel,
            focus_opts=self._opts.focus_opts(),
            segment_opts=self._opts.segment_opts(),
            classify_opts=classify_opts,
            features_opts=self._opts.features_opts() if target == "features" else None,
            use_disk_focus=use_disk_focus,
            channel_labels=self._opts.channel_labels(),
            roi=self._opts.roi(),
            cached_entry=self._cache.get(self._sample_path, self._fov_index),
        )

    def _effective_phase_index(self, payload: FocusPayload) -> int:
        """Pick the phase channel index to use for canvas rendering.

        Prefers the worker's resolved value (auto-detection result),
        then any explicit int the user picked in the InputPanel,
        finally falls back to 0.
        """
        if payload.resolved_phase_channel is not None:
            return int(payload.resolved_phase_channel)
        raw = self._opts.phase_channel()
        if isinstance(raw, int):
            return raw
        return 0

    @staticmethod
    def _is_focused_tiff(path: Optional[Path]) -> bool:
        """True when the sample is a focused TIFF on disk (vs. a raw CZI).

        The two source types use different focus shims in the worker —
        TIFFs are loaded via ``load_hyperstack`` (cheap) while CZIs go
        through the in-memory single-FOV focus chain.
        """
        if path is None:
            return True
        suffix = path.suffix.lower()
        return suffix in (".tif", ".tiff")

    def _fire_pending(self) -> None:
        if self._pending_request is None:
            return
        request = self._pending_request
        self._pending_request = None
        self._cancel_worker()
        self._start_worker(request)

    def _cancel_worker(self) -> None:
        if self._worker is None:
            return
        self._worker.request_stop()
        # Don't block waiting for the worker — Qt will dispatch its
        # ``cancelled`` / ``chainFinished`` signal when the cellpose call
        # comes back. Detach so a new worker can start.
        try:
            self._worker.stageStarted.disconnect(self._on_stage_started)
            self._worker.stageFinished.disconnect(self._on_stage_finished)
            self._worker.stageFailed.disconnect(self._on_stage_failed)
            self._worker.chainFinished.disconnect(self._on_chain_finished)
            self._worker.cancelled.disconnect(self._on_cancelled)
        except (TypeError, RuntimeError):
            # Already disconnected or deleted; ignore.
            pass
        self._worker = None

    def _start_worker(self, request: RenderRequest) -> None:
        worker = PreviewWorker(request, parent=self)
        worker.stageStarted.connect(self._on_stage_started)
        worker.stageFinished.connect(self._on_stage_finished)
        worker.stageFailed.connect(self._on_stage_failed)
        worker.chainFinished.connect(self._on_chain_finished)
        worker.cancelled.connect(self._on_cancelled)
        self._worker = worker
        self.renderStateChanged.emit(True)
        worker.start()

    # ------------------------------------------------------------- slots

    def _on_stage_started(self, stage: str) -> None:
        self._progress.started(stage)

    def _on_stage_finished(self, stage: str, payload: object) -> None:
        # Land the result both in the cache and on the canvas.
        if self._sample_path is None:
            return
        entry = self._cache.get(self._sample_path, self._fov_index)
        if isinstance(payload, FocusPayload):
            entry.focus_key = payload.key
            entry.phase = payload.phase
            entry.image_channels = payload.image_channels
            entry.channel_names = payload.channel_names
            if payload.resolved_phase_channel is not None:
                entry.resolved_phase_channel = int(payload.resolved_phase_channel)
            self._canvas.set_phase(payload.phase)
            self._canvas.set_image_channels(
                self._channels_for_canvas(entry.image_channels),
                payload.channel_names or [],
                # Prefer the worker-resolved index (correct when the
                # InputPanel is set to "auto"), falling back to whatever
                # the user explicitly picked.
                self._effective_phase_index(payload),
            )
        elif isinstance(payload, SegmentPayload):
            entry.segment_key = payload.key
            entry.mask = payload.mask
            entry.n_cells = payload.n_cells
            self._canvas.set_mask(payload.mask, entry.decisions)
            # Foci detection runs on the main thread after segment because
            # it needs both the fluorescence channels and the labeled
            # mask. Cheap for ``dog``/``log``/``wavelet``; can be sluggish
            # for ``decon_bm3d_wavelet`` on big FOVs — the
            # user toggles detectors in the panel to manage cost.
            if self._tab == "foci_det":
                self._run_foci_detection_preview(entry)
        elif isinstance(payload, ClassifyPayload):
            entry.classify_key = payload.key
            entry.decisions = payload.decisions
            # Repaint the mask so colors update with classification.
            self._canvas.set_mask(entry.mask, entry.decisions)
        elif isinstance(payload, FeaturesPayload):
            entry.features_key = payload.key
            entry.features_df = payload.df
            self._canvas.set_features_df(payload.df)

    # ─────────────────────────────────────────────────────────────────
    # Foci-stages preview: post-processing after the worker chain lands
    # ─────────────────────────────────────────────────────────────────

    def _channels_for_canvas(self, image_channels):
        """Return the channel stack to display on the canvas.

        On the ``fluor_norm`` AND ``foci_det`` tabs we substitute each
        fluorescence channel with its normalised version — so the foci
        detector sees (and the user sees) the same image the full
        pipeline will run detection against. On every other tab we pass
        the raw channels straight through. Phase + mask are never
        touched. Every selected fluor channel is normalised, not just
        the first — so multi-channel foci detection stays consistent.
        """
        if image_channels is None or self._tab not in (
            "fluor_norm", "foci_det",
        ):
            return image_channels
        opts = self._opts.fluor_norm_opts()
        if opts is None:
            return image_channels
        method = getattr(opts, "method", "none")
        if method in ("", "none"):
            return image_channels
        try:
            from mycomorph.core.foci.normalise import (
                NormaliserOpts, build, requires_run_level_fit,
            )
        except Exception:  # noqa: BLE001
            return image_channels
        if requires_run_level_fit(method):
            # BaSiC needs every FOV of a channel to fit; single-FOV
            # preview can't do that. Show raw and let the user pick a
            # per-FOV method to iterate on.
            return image_channels

        norm_opts = NormaliserOpts(
            tophat_radius_px=getattr(opts, "tophat_radius_px", 5),
            gaussian_lp_sigma=getattr(opts, "gaussian_lp_sigma", None),
            rl_iterations=getattr(opts, "rl_iterations", 30),
            rl_psf_sigma=getattr(opts, "rl_psf_sigma", 1.5),
            bm3d_sigma=getattr(opts, "bm3d_sigma", None),
        )
        import numpy as np  # noqa: PLC0415 — kept lazy with the foci import
        try:
            normaliser = build(method, norm_opts)
        except Exception:  # noqa: BLE001
            return image_channels
        # Channel indices: skip phase (and don't touch the mask if it
        # ever lands here — image_channels at this stage is C×Y×X with
        # no mask appended yet, but be defensive).
        phase_idx = self._opts.phase_channel()
        phase_idx = int(phase_idx) if isinstance(phase_idx, int) else 0
        apply_to = getattr(opts, "apply_to_channels", None) or []
        n_ch = image_channels.shape[0]
        if not apply_to:
            apply_to = [c for c in range(n_ch) if c != phase_idx]
        out = np.array(image_channels, dtype=np.float32, copy=True)
        for c in apply_to:
            if not (0 <= c < n_ch):
                continue
            try:
                out[c] = normaliser.apply(image_channels[c])
            except Exception:  # noqa: BLE001
                pass
        return out

    def _run_foci_detection_preview(self, entry: CacheEntry) -> None:
        """Detect foci on the visible fluor channel and cache the
        features DataFrame on the cache entry. Then push the features
        to the panel (for inline histograms) and run the threshold
        filter to draw the scatter layer.

        Cache key: detector + opts + channel + fluor-norm opts +
        upstream segment key. A threshold drag does NOT invalidate this
        cache — :meth:`apply_thresholds_only` re-uses the cached
        DataFrame and skips detection.
        """
        opts = self._opts.foci_det_opts()
        if opts is None or entry.image_channels is None or entry.mask is None:
            self._canvas.clear_foci_layers()
            self._canvas.set_foci_features(None)
            return

        detector_keys = list(getattr(opts, "detector_keys", []) or [])
        if not detector_keys:
            self._canvas.clear_foci_layers()
            self._canvas.set_foci_features(None)
            entry.foci_df = None
            entry.foci_key = None
            return
        det_key = str(detector_keys[0])  # single-detector GUI

        try:
            from mycomorph.core.foci.detectors import REGISTRY as DET_REGISTRY
            from mycomorph.core.foci._types import DetectorOpts as _DO
            from mycomorph.core.foci.features import features_dataframe
        except Exception:  # noqa: BLE001
            return

        phase_idx = self._opts.phase_channel()
        phase_idx = int(phase_idx) if isinstance(phase_idx, int) else 0
        target_channels = self._target_fluor_channels(
            entry.image_channels, opts, phase_idx,
        )
        if not target_channels:
            self._wipe_all_foci_layers()
            self._canvas.set_foci_features(None)
            return

        # Optionally apply the fluor-norm preview transform first so
        # detectors see the same input they would in the full pipeline.
        channels = self._channels_for_canvas(entry.image_channels)
        mask = entry.mask
        det_opts = getattr(opts, "detector_opts", None) or _DO()

        # Cache miss → run detection + features on every selected channel
        # and stack the results. Otherwise re-use the cached DataFrame so
        # threshold-only changes are instant.
        from .cache import foci_preview_key
        seg_k = entry.segment_key
        fluor_opts = self._opts.fluor_norm_opts()
        new_key = foci_preview_key(
            seg_k, det_key, tuple(target_channels), det_opts, fluor_opts,
        )
        if entry.foci_key != new_key or entry.foci_df is None:
            if det_key not in DET_REGISTRY:
                entry.foci_df = None
                entry.foci_key = new_key
                self._wipe_all_foci_layers()
                self._canvas.set_foci_features(None)
                return
            try:
                import pandas as pd  # noqa: PLC0415
                per_channel: list = []
                for ch in target_channels:
                    detector = DET_REGISTRY[det_key]()
                    image = channels[ch]
                    foci = detector.detect(image, mask, det_opts)
                    df_ch = features_dataframe(
                        image, mask, foci, detector=det_key,
                    )
                    if not df_ch.empty:
                        df_ch["channel_index"] = int(ch)
                    per_channel.append(df_ch)
                df = (
                    pd.concat(per_channel, ignore_index=True)
                    if per_channel else pd.DataFrame()
                )
            except Exception:  # noqa: BLE001
                entry.foci_df = None
                entry.foci_key = new_key
                self._wipe_all_foci_layers()
                self._canvas.set_foci_features(None)
                return
            entry.foci_df = df
            entry.foci_key = new_key

        # Push pooled features to the panel histograms (across channels).
        self._canvas.set_foci_features(entry.foci_df)

        # Filter + paint scatter (one layer per channel).
        self._paint_scatter_from_cache(entry, det_key)

    # Fallback palette used only when a channel has no display label
    # (so we can't infer its colour from the name). Picked to be
    # visually distinct in case multiple channels lack labels.
    _FALLBACK_PALETTE: list[tuple[int, int, int, int]] = [
        (255, 235,  60, 230),   # yellow
        (200,  90, 255, 230),   # magenta
        ( 80, 230, 230, 230),   # cyan
        (255, 110,  80, 230),   # orange
    ]

    def _layer_name(self, det_key: str, channel_index: int) -> str:
        return f"{det_key}@ch{int(channel_index)}"

    def _channel_color_rgba(
        self, channel_index: int, fallback_slot: int = 0,
    ) -> tuple[int, int, int, int]:
        """Map a fluor channel index to a scatter colour that matches
        the channel's display LUT (DAPI→blue, GFP→green, mCherry→red,
        Cy5→magenta). Channels without an inferrable label fall back to
        a small distinct palette indexed by their position in the
        selected-channels list.
        """
        labels = self._opts.channel_labels() or []
        name = (
            labels[channel_index]
            if 0 <= channel_index < len(labels) else ""
        )
        from .canvas import color_for_channel_name, rgb_for_channel_name
        # ``color_for_channel_name`` returns "white" for un-matched
        # names; we treat that as "no inferred colour" and use the
        # fallback palette so multi-channel runs stay distinguishable.
        if color_for_channel_name(name) == "white":
            return self._FALLBACK_PALETTE[
                fallback_slot % len(self._FALLBACK_PALETTE)
            ]
        r, g, b = rgb_for_channel_name(name)
        return (r, g, b, 230)

    def _wipe_all_foci_layers(self) -> None:
        """Empty every registered foci scatter layer in the canvas. Used
        when there's nothing to draw (no detector, no channels, etc.)."""
        for name in list(self._foci_scatter_names):
            self._canvas.set_foci_layer_data(name, [], [], [])
        self._foci_scatter_names = set()

    def _paint_scatter_from_cache(
        self, entry: CacheEntry, det_key: str,
    ) -> None:
        """Apply current thresholds to ``entry.foci_df`` and paint one
        scatter layer per channel. Cheap — used by both the
        detect-then-filter path and the threshold-only hot path.
        """
        from ..foci_filter_io import compute_pass_mask
        df = entry.foci_df
        try:
            visible = bool(self._opts.foci_visible())
        except Exception:  # noqa: BLE001
            visible = True

        if df is None or df.empty:
            self._wipe_all_foci_layers()
            return
        thr = self._opts.foci_thresholds() or {}

        # Backward-compat: if rows aren't tagged with channel_index
        # (e.g. legacy cache entries), treat them as one synthetic
        # channel so we still paint something.
        if "channel_index" not in df.columns:
            df = df.assign(channel_index=-1)

        expected_layers: set[str] = set()
        # Sort channels for deterministic colour assignment.
        for slot, (ch, group_df) in enumerate(
            df.groupby("channel_index", sort=True),
        ):
            ch_int = int(ch)
            layer = self._layer_name(det_key, ch_int)
            expected_layers.add(layer)
            color = self._channel_color_rgba(ch_int, fallback_slot=slot)
            self._canvas.add_foci_layer(layer, color)
            if thr:
                pass_mask = compute_pass_mask(group_df, thr)
                passed = group_df.loc[pass_mask]
            else:
                passed = group_df
            xs = passed["x"].astype(float).tolist()
            ys = passed["y"].astype(float).tolist()
            self._canvas.set_foci_layer_data(
                layer, xs, ys, [12] * len(xs),
            )
            self._canvas.set_foci_layer_visible(layer, visible)

        # Wipe stale per-channel layers (e.g. user just unticked a
        # channel) so they don't linger on the canvas.
        for old_name in list(self._foci_scatter_names):
            if old_name not in expected_layers:
                self._canvas.set_foci_layer_data(old_name, [], [], [])
        self._foci_scatter_names = expected_layers

    def apply_thresholds_only(self) -> None:
        """Re-paint the scatter from the cached foci DataFrame using
        current thresholds. No detection re-run; intended for the
        threshold-drag hot path.
        """
        if self._sample_path is None or self._tab != "foci_det":
            return
        entry = self._cache.get(self._sample_path, self._fov_index)
        if entry.foci_df is None:
            return
        opts = self._opts.foci_det_opts()
        if opts is None:
            return
        keys = list(getattr(opts, "detector_keys", []) or [])
        if not keys:
            return
        self._paint_scatter_from_cache(entry, str(keys[0]))

    @staticmethod
    def _target_fluor_channels(
        image_channels, opts, phase_idx: int,
    ) -> list[int]:
        """Every fluor channel the detector should run on. Respects
        ``apply_to_channels`` when the user has narrowed the list;
        otherwise returns every channel that isn't phase. Order matches
        the user's selection (or natural channel order) so palette
        colours stay stable across renders."""
        apply_to = list(getattr(opts, "apply_to_channels", None) or [])
        n_ch = image_channels.shape[0]
        if not apply_to:
            return [c for c in range(n_ch) if c != phase_idx]
        return [c for c in apply_to if 0 <= c < n_ch and c != phase_idx]

    def _on_stage_failed(self, stage: str, msg: str) -> None:
        self._progress.failed(stage, msg)

    def _on_chain_finished(self) -> None:
        self._progress.finished()
        self.renderStateChanged.emit(False)
        # After the chain settles (cached or freshly computed), re-apply
        # the foci-tab post-processing on top — the worker doesn't emit
        # cached stage payloads, so cache hits would otherwise show the
        # raw image with no foci overlays / no normalisation applied.
        if self._tab in ("fluor_norm", "foci_det"):
            self._redraw_foci_overlays()
        # If the user changed something while we were running, fire
        # again now (debounced re-trigger handled elsewhere).
        if self._pending_request is not None:
            self._fire_pending()

    def _redraw_foci_overlays(self) -> None:
        """Push fluor-norm / foci-det post-processing onto cached data.

        Called on tab arrival (so cache hits paint instantly) and after
        the chain finishes (so worker re-runs flush the new transforms /
        detections). No-op when no FOV is selected or when there's no
        cached focus payload yet.
        """
        if self._sample_path is None:
            return
        entry = self._cache.get(self._sample_path, self._fov_index)
        if entry.image_channels is not None:
            phase_idx = (
                entry.resolved_phase_channel
                if entry.resolved_phase_channel is not None
                else (
                    self._opts.phase_channel()
                    if isinstance(self._opts.phase_channel(), int)
                    else 0
                )
            )
            self._canvas.set_image_channels(
                self._channels_for_canvas(entry.image_channels),
                entry.channel_names or [],
                int(phase_idx),
            )
        if self._tab == "foci_det" and entry.mask is not None:
            self._run_foci_detection_preview(entry)
        elif self._tab == "fluor_norm":
            # Make sure stale foci scatter doesn't linger.
            self._canvas.clear_foci_layers()

    def _on_cancelled(self) -> None:
        self._progress.finished()
        self.renderStateChanged.emit(False)
