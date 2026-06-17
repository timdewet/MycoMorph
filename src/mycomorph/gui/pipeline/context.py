"""RunContext — immutable-ish bag of state passed from the GUI to the runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from mycomorph.core.api import ClassifyOpts, ExtractOpts, FocusOpts, SegmentOpts
from mycomorph.core.foci import DetectorOpts

from .layout import PlateLayout


@dataclass
class FluorescentNormalisationOpts:
    """Configuration for the Fluorescent Normalisation pipeline stage."""

    # Key in NORMALISER_REGISTRY: "none", "tophat", "gaussian_lp",
    # "richardson_lucy", "bm3d", "decon_bm3d", "basic".
    method: str = "tophat"

    # NormaliserOpts knobs the panel will eventually expose. The stage
    # builds a NormaliserOpts from these at run time.
    tophat_radius_px: int = 5
    gaussian_lp_sigma: Optional[float] = None       # None → auto
    rl_iterations: int = 30
    rl_psf_sigma: float = 1.5
    bm3d_sigma: Optional[float] = None              # None → MAD-estimate

    # Channel selection. None → every channel that isn't phase or mask.
    apply_to_channels: Optional[list[int]] = None


@dataclass
class FociDetectionOpts:
    """Configuration for the Foci Detection pipeline stage."""

    # REGISTRY key for the detector. The GUI restricts this to one key
    # at a time (``wavelet`` by default — a strong baseline for dim
    # signal after the Fluorescent Normalisation stage). The shape stays
    # ``list[str]`` for forward compatibility with multi-detector runs
    # driven from CLI / scripts.
    detector_keys: list[str] = field(default_factory=lambda: ["wavelet"])

    # Shared detector parameters; individual detectors ignore fields that
    # don't apply to them.
    detector_opts: DetectorOpts = field(default_factory=DetectorOpts)

    # Channel selection. None → every channel that isn't phase or mask.
    apply_to_channels: Optional[list[int]] = None


@dataclass
class EmbeddingOpts:
    """Configuration for the optional CNN Embeddings pipeline stage."""

    model_type: str = "ResNet-18"           # "ResNet-18" | "Lightweight"
    model_source: str = "Auto (train/fine-tune)"
    model_path: str = ""                    # only used when model_source = "Use existing model"
    in_channels: int = 1                    # number of non-mask image channels
    # Append the segmentation mask as an additional input channel. Default
    # ON to match the Mtb reference (phase + ParB + mask).
    include_mask: bool = True
    epochs: int = 50
    batch_size: int = 256
    include_library_crops: bool = True
    library_dir: Optional[Path] = None
    species: str = ""
    # When True, the stage runs without a current-run input — it pulls only
    # library crops, trains a model, and re-extracts embeddings for the whole
    # library. Used by the "Train embeddings" input mode.
    train_only: bool = False


@dataclass
class RunContext:
    czi_path: Path
    output_dir: Path
    layout: PlateLayout
    # Multi-CZI plate support: when the user has selected several CZIs that
    # together compose one logical plate (e.g. row-band split across files),
    # ``czi_paths`` carries the full list. Single-CZI runs leave this as a
    # one-element list and ``czi_path`` keeps its meaning for legacy code.
    czi_paths: list[Path] = field(default_factory=list)

    # Stage enables
    do_split: bool = True
    do_focus: bool = True
    do_segment: bool = True
    do_classify: bool = True
    do_features: bool = False
    do_fluorescent_normalisation: bool = False
    do_foci_detection: bool = False
    do_embeddings: bool = False

    # Per-stage options
    focus_opts: FocusOpts = field(default_factory=FocusOpts)
    segment_opts: SegmentOpts = field(default_factory=SegmentOpts)
    classify_opts: ClassifyOpts = field(default_factory=ClassifyOpts)
    features_opts: ExtractOpts = field(default_factory=ExtractOpts)
    fluorescent_normalisation_opts: FluorescentNormalisationOpts = field(
        default_factory=FluorescentNormalisationOpts
    )
    foci_detection_opts: FociDetectionOpts = field(
        default_factory=FociDetectionOpts
    )
    embeddings_opts: EmbeddingOpts = field(default_factory=EmbeddingOpts)

    # Channel handling (resolved early so all downstream stages agree)
    phase_channel: Optional[int] = None
    channel_labels: Optional[list[str]] = None

    # Reuse-existing flag — set by the runner so stages can do per-input
    # reuse logic (e.g. a multi-CZI plate where one CZI's wells are already
    # processed but a newly-added CZI still needs to run).
    reuse_existing: bool = False

    # ------------------------------------------------------------------ paths

    @property
    def all_czi_paths(self) -> list[Path]:
        """Canonical accessor: returns the full list of CZI paths whether
        the run was constructed with the legacy single-path or the new
        ``czi_paths`` list. Stages should call this rather than touching
        either field directly."""
        if self.czi_paths:
            return list(self.czi_paths)
        if self.czi_path:
            return [self.czi_path]
        return []

    @property
    def split_dir(self) -> Path:
        return self.output_dir / "01_split"

    @property
    def focus_dir(self) -> Path:
        # Split is skipped when Focus is on (Focus produces the per-well output
        # directly). In that case we merge the two stage roles into a single
        # folder so the output hierarchy doesn't advertise a missing step.
        if self.do_split and self.do_focus:
            return self.output_dir / "01_split_and_focused"
        return self.output_dir / "01_focus"

    @property
    def segment_dir(self) -> Path:
        return self.output_dir / "02_segment"

    @property
    def classify_dir(self) -> Path:
        return self.output_dir / "03_classify"

    @property
    def features_dir(self) -> Path:
        return self.output_dir / "04_features"

    @property
    def fluorescent_normalisation_dir(self) -> Path:
        return self.output_dir / "04b_fluorescent_normalisation"

    @property
    def foci_detection_dir(self) -> Path:
        return self.output_dir / "04c_foci_detection"

    @property
    def foci_filtered_dir(self) -> Path:
        # Filtered subsets of the foci-detection parquets written by the
        # FociFilterDialog when the user picks per-feature thresholds.
        return self.output_dir / "04d_foci_filtered"

    @property
    def foci_labels_dir(self) -> Path:
        # Used by the future FociLabelDialog (Slice 4). Defined now so
        # downstream code can reference it without a second context.py edit.
        return self.output_dir / "05_foci_labels"

    @property
    def foci_filters_dir(self) -> Path:
        # Per-session threshold JSONs written alongside the filtered
        # parquets so the chosen thresholds are auditable.
        return self.output_dir / "05_foci_filters"

    @property
    def embeddings_dir(self) -> Path:
        return self.output_dir / "05_embeddings"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "run_manifest.json"


@dataclass
class BulkRunContext:
    """Run context for non-plate (Single file or Bulk) modes.

    A list of (czi_path, label-fields) entries that all share the same
    output directory and stage options. The Focus stage is invoked once
    per entry; Segment and Classify run once each over the resulting
    per-CZI TIFFs.
    """

    czi_entries: list[dict[str, Any]]
    output_dir: Path

    # Stage enables (Split is unused in bulk mode — each CZI is one sample).
    do_focus: bool = True
    do_segment: bool = True
    do_classify: bool = True
    do_features: bool = False
    do_fluorescent_normalisation: bool = False
    do_foci_detection: bool = False
    do_embeddings: bool = False

    # Per-stage options (shared across all CZIs in the batch)
    focus_opts: FocusOpts = field(default_factory=FocusOpts)
    segment_opts: SegmentOpts = field(default_factory=SegmentOpts)
    classify_opts: ClassifyOpts = field(default_factory=ClassifyOpts)
    features_opts: ExtractOpts = field(default_factory=ExtractOpts)
    fluorescent_normalisation_opts: FluorescentNormalisationOpts = field(
        default_factory=FluorescentNormalisationOpts
    )
    foci_detection_opts: FociDetectionOpts = field(
        default_factory=FociDetectionOpts
    )
    embeddings_opts: EmbeddingOpts = field(default_factory=EmbeddingOpts)

    phase_channel: Optional[int] = None
    channel_labels: Optional[list[str]] = None

    # ── duck-type compatibility with RunContext ──────────────────────
    # SegmentStage and ClassifyStage read these; provide stable values so
    # the existing stage code can be reused without branching on context
    # type.
    do_split: bool = False     # bulk mode never splits
    reuse_existing: bool = False   # set by BulkPipelineRunner from its own flag

    @property
    def split_dir(self) -> Path:
        # Path that never exists, so SegmentStage's split-fallback skips it.
        return self.output_dir / "_split_unused_in_bulk_mode"

    # Same flat directory layout as the plate flow so downstream tools
    # don't have to special-case anything.
    @property
    def focus_dir(self) -> Path:
        return self.output_dir / "01_focus"

    @property
    def segment_dir(self) -> Path:
        return self.output_dir / "02_segment"

    @property
    def classify_dir(self) -> Path:
        return self.output_dir / "03_classify"

    @property
    def features_dir(self) -> Path:
        return self.output_dir / "04_features"

    @property
    def fluorescent_normalisation_dir(self) -> Path:
        return self.output_dir / "04b_fluorescent_normalisation"

    @property
    def foci_detection_dir(self) -> Path:
        return self.output_dir / "04c_foci_detection"

    @property
    def foci_filtered_dir(self) -> Path:
        return self.output_dir / "04d_foci_filtered"

    @property
    def foci_labels_dir(self) -> Path:
        return self.output_dir / "05_foci_labels"

    @property
    def foci_filters_dir(self) -> Path:
        return self.output_dir / "05_foci_filters"

    @property
    def embeddings_dir(self) -> Path:
        return self.output_dir / "05_embeddings"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "run_manifest.json"
