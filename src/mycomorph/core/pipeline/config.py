"""Pipeline runs described by a JSON config (``mycomorph-cli run --config``).

A config says what the GUI's Input, Plate layout, stage and Run pages say
between them, in the form ``run_manifest.json`` records a run::

    {
      "mode": "plate",
      "czi_paths": ["plate1.czi"],
      "plate_layout_csv": "plate_layout.csv",
      "output_dir": "results",
      "stages": [
        {"name": "Split"},
        {"name": "Focus", "opts": {"mode": "edf"}},
        {"name": "Segment", "opts": {"gpu": false}},
        {"name": "Classify", "opts": {"model_path": "mtb-2026-06"}},
        {"name": "Features"}
      ]
    }

``mode`` is ``"plate"`` (the GUI's Single plate) or ``"bulk"`` (Single file
and Multiple files), which takes ``bulk_entries_csv`` or inline ``entries``
instead of the CZIs and layout. Every stage listed in ``stages`` runs unless
it has ``"enabled": false``; such an entry only supplies options that other
stages read. Options left out keep the defaults the GUI starts with.
``phase_channel`` and ``channel_labels`` are optional: when absent they are
read from the first input file, as the Input page does.

Every manifest holds the run it records in a ``config`` block of this form,
so a manifest can be passed as the config to repeat its run. Manifests written
before that block existed are read from their top-level keys.

Relative paths are resolved against the folder holding the config file (and,
for paths inside an entries CSV, the folder holding that CSV).
"""

from __future__ import annotations

import json
import types
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union, get_args, get_origin, get_type_hints

from .bulk_layout import COLUMNS as ENTRY_COLUMNS
from .bulk_layout import BulkLayout
from .context import BulkRunContext, RunContext
from .layout import PlateLayout
from .runner import STAGE_FLAGS, STAGE_OPTS_ATTR, BulkRun, PlateRun

STAGE_NAMES = tuple(name for name, _ in STAGE_FLAGS)

_CONFIG_KEYS = {
    "mode", "czi", "czi_paths", "plate_layout_csv", "bulk_entries_csv",
    "entries", "output_dir", "phase_channel", "channel_labels", "stages",
}
# Bookkeeping a manifest carries next to the run description.
_MANIFEST_KEYS = {"started_at", "finished_at", "software", "stopped_early", "config"}
_STAGE_KEYS = {"name", "enabled", "opts"}
_MANIFEST_STAGE_KEYS = {"reused", "stopped", "skipped", "elapsed_s", "outputs"}


class ConfigError(ValueError):
    """The config can't be turned into a run; the message says what to fix."""


# ─────────────────────────────────────────────────────────────────────────────
# Shared with the GUI's pre-run check
# ─────────────────────────────────────────────────────────────────────────────

def missing_upstream_outputs(out: Path, enables: dict[str, bool]) -> list[str]:
    """Stages that are enabled without their input: the stage that makes it
    is off and no earlier output is on disk. Empty when the run can start."""
    def has_files(d: Path) -> bool:
        return d.exists() and any(d.iterdir())

    blocking: list[str] = []
    if enables.get("Classify") and not enables.get("Segment"):
        seg_dir = out / "02_segment"
        if not has_files(seg_dir):
            blocking.append(
                "Classify is enabled but Segment is disabled and no "
                f"segment output exists at {seg_dir}."
            )

    # Split alone produces unfocused per-sample TIFFs (single-plane
    # inputs need no focus picking), so it also satisfies Segment.
    if (enables.get("Segment") and not enables.get("Focus")
            and not enables.get("Split")):
        focus_candidates = [
            out / "01_split_and_focused",
            out / "01_focus",
            out / "01_split",
        ]
        if not any(has_files(d) for d in focus_candidates):
            blocking.append(
                "Segment is enabled but Focus and Split are both "
                "disabled, and no earlier output exists in any of:\n  "
                + "\n  ".join(str(d) for d in focus_candidates)
            )

    if enables.get("Features") and not (
        enables.get("Segment") or enables.get("Classify")
    ):
        features_candidates = [out / "03_classify", out / "02_segment"]
        if not any(has_files(d) for d in features_candidates):
            blocking.append(
                "Features is enabled but neither Segment nor Classify is "
                "enabled, and no upstream output exists in any of:\n  "
                + "\n  ".join(str(d) for d in features_candidates)
            )
    return blocking


# ─────────────────────────────────────────────────────────────────────────────
# The plan: a context ready to run
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunPlan:
    """A run built from a config.

    ``ctx`` is the same context object the GUI hands its runner. ``notes``
    are things worth telling the user about how the config was read.
    """

    mode: str                                  # "plate" | "bulk"
    ctx: RunContext | BulkRunContext
    notes: list[str] = field(default_factory=list)
    bulk_layout: Optional[BulkLayout] = None   # bulk mode: the entries table

    @property
    def enables(self) -> dict[str, bool]:
        return {name: bool(getattr(self.ctx, flag)) for name, flag in STAGE_FLAGS}

    @property
    def enabled_stages(self) -> list[str]:
        return [name for name, on in self.enables.items() if on]

    def preflight(self, *, reuse_existing: bool = True) -> tuple[list[str], list[str]]:
        """The checks the GUI makes when Run is pressed: ``(blocking, warnings)``."""
        ctx = self.ctx
        out = ctx.output_dir
        blocking = missing_upstream_outputs(out, self.enables)
        if self.bulk_layout is not None:
            blocking.extend(self.bulk_layout.validate())
        elif ctx.do_split or ctx.do_focus:
            # What the Split and Focus stages check as the run starts.
            blocking.extend(
                f"CZI file not found: {p}" for p in ctx.all_czi_paths if not p.exists()
            )
            if not ctx.do_focus:
                blocking.extend(ctx.layout.validate())

        warnings: list[str] = []
        if ctx.do_segment and ctx.segment_opts.gpu and not _gpu_available():
            warnings.append(
                "GPU requested for segmentation but no CUDA or MPS device "
                "was found; the run will fall back to CPU (much slower)."
            )
        if not reuse_existing and out.exists() and any(out.iterdir()):
            warnings.append(
                f"Output directory already contains files and reuse is off. "
                f"Existing files in {out} may be overwritten."
            )
        return blocking, warnings

    def make_run(self, *, reuse_existing: bool = True) -> PlateRun | BulkRun:
        """The run object: connect to its signals, then call ``execute()``."""
        if self.mode == "plate":
            return PlateRun(self.ctx, reuse_existing=reuse_existing)
        return BulkRun(self.ctx, reuse_existing=reuse_existing)


def _gpu_available() -> bool:
    try:
        from mycomorph.modelprep.core.device import gpu_available
        return bool(gpu_available())
    except Exception:  # noqa: BLE001
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Reading a config
# ─────────────────────────────────────────────────────────────────────────────

def load_run_plan(
    config_path: Path, *, output_dir: Optional[Path] = None,
) -> RunPlan:
    """Read a config file (or a ``run_manifest.json``) into a :class:`RunPlan`.

    ``output_dir`` replaces the config's own, e.g. to repeat a manifest's run
    in a new folder.
    """
    config_path = Path(config_path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ConfigError(f"Cannot read {config_path}: {e}") from e
    except ValueError as e:
        raise ConfigError(f"{config_path} is not valid JSON: {e}") from e
    return plan_from_config(
        data, base_dir=config_path.resolve().parent, output_dir=output_dir,
    )


def plan_from_config(
    data: Any, *, base_dir: Path, output_dir: Optional[Path] = None,
) -> RunPlan:
    """Build a :class:`RunPlan` from parsed config (or manifest) JSON."""
    if not isinstance(data, dict):
        raise ConfigError("The config must be a JSON object.")
    base_dir = Path(base_dir)

    # A manifest carries its run in a "config" block. Older manifests
    # don't, and are read from their top-level keys like a hand-written one.
    from_manifest = isinstance(data.get("config"), dict)
    block = data["config"] if from_manifest else data
    unknown = sorted(set(block) - _CONFIG_KEYS - _MANIFEST_KEYS)
    if unknown:
        raise ConfigError(
            f"Unknown config key(s): {', '.join(unknown)}. "
            f"Known keys: {', '.join(sorted(_CONFIG_KEYS))}."
        )
    notes: list[str] = []

    mode = block.get("mode")
    if mode is None:
        mode = "bulk" if ("bulk_entries_csv" in block or "entries" in block) else "plate"
    if mode not in ("plate", "bulk"):
        raise ConfigError(f'"mode" must be "plate" or "bulk", not {mode!r}.')

    if output_dir is not None:
        out = Path(output_dir).expanduser().resolve()
    elif block.get("output_dir"):
        out = _resolve_path(_string(block["output_dir"], "output_dir"), base_dir)
    else:
        raise ConfigError('The config needs an "output_dir" (or pass --output-dir).')

    enables, stage_opts = _read_stages(block, mode, legacy=not from_manifest)

    bulk_layout: Optional[BulkLayout] = None
    if mode == "plate":
        czi_paths = _plate_inputs(block, base_dir)
        layout = _plate_layout(block, czi_paths, base_dir, notes)
        first_input: Optional[Path] = czi_paths[0]
    else:
        bulk_layout = _bulk_entries(block, base_dir)
        entries = bulk_layout.active_rows().to_dict(orient="records")
        first_input = Path(str(entries[0]["czi_path"])) if entries else None

    labels, phase = _channels(block, first_input, notes)

    opts = _build_stage_opts(stage_opts, base_dir, phase)
    flags = {flag: enables[name] for name, flag in STAGE_FLAGS}
    common = dict(
        output_dir=out, phase_channel=phase, channel_labels=labels, **flags, **opts,
    )
    ctx: RunContext | BulkRunContext
    if mode == "plate":
        ctx = RunContext(
            czi_path=czi_paths[0], czi_paths=list(czi_paths), layout=layout, **common,
        )
    else:
        ctx = BulkRunContext(czi_entries=entries, **common)
    return RunPlan(mode=mode, ctx=ctx, notes=notes, bulk_layout=bulk_layout)


# ── stages ───────────────────────────────────────────────────────────────────

def _stage_key(name: Any) -> str:
    key = " ".join(str(name).replace("_", " ").replace("-", " ").lower().split())
    return key.replace("normalization", "normalisation")


_STAGE_BY_KEY = {_stage_key(name): name for name in STAGE_NAMES}


def _read_stages(
    block: dict, mode: str, *, legacy: bool,
) -> tuple[dict[str, bool], dict[str, dict]]:
    """``(enabled flag per stage, raw options per listed stage)``."""
    stages = block.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ConfigError(
            'The config needs a "stages" list, e.g. '
            '[{"name": "Focus"}, {"name": "Segment"}]. '
            f"Stages: {', '.join(STAGE_NAMES)}."
        )
    enables = {name: False for name in STAGE_NAMES}
    raw_opts: dict[str, dict] = {}
    seen: set[str] = set()
    for i, entry in enumerate(stages):
        if isinstance(entry, str):
            entry = {"name": entry}
        if not isinstance(entry, dict) or "name" not in entry:
            raise ConfigError(f'stages[{i}] must be an object with a "name".')
        name = _STAGE_BY_KEY.get(_stage_key(entry["name"]))
        if name is None:
            raise ConfigError(
                f"stages[{i}]: unknown stage {entry['name']!r}. "
                f"Stages: {', '.join(STAGE_NAMES)}."
            )
        if name in seen:
            raise ConfigError(f"Stage {name!r} is listed twice.")
        seen.add(name)
        unknown = sorted(set(entry) - _STAGE_KEYS - _MANIFEST_STAGE_KEYS)
        if unknown:
            raise ConfigError(
                f"Stage {name!r}: unknown key(s) {', '.join(unknown)}. "
                'A stage entry takes "name", "opts" and "enabled".'
            )
        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ConfigError(f'Stage {name!r}: "enabled" must be true or false.')
        enables[name] = enabled
        opts = entry.get("opts") or {}
        if not isinstance(opts, dict):
            raise ConfigError(f'Stage {name!r}: "opts" must be an object.')
        if opts and name not in STAGE_OPTS_ATTR:
            raise ConfigError(f"Stage {name!r} takes no options.")
        raw_opts[name] = opts

    # A manifest from before the "config" block lists only the stages that
    # ran, and Focus runs in place of a ticked Split. In plate mode that tick
    # decides the folder name, so recover it from where Focus wrote.
    if legacy and mode == "plate" and enables["Focus"] and "Split" not in seen:
        focus = next(e for e in stages if isinstance(e, dict)
                     and _STAGE_BY_KEY.get(_stage_key(e.get("name"))) == "Focus")
        outputs = focus.get("outputs") or []
        if any(Path(str(p)).parent.name == "01_split_and_focused" for p in outputs):
            enables["Split"] = True

    if not any(enables.values()):
        raise ConfigError("No stage is enabled, so there is nothing to run.")
    return enables, raw_opts


def _build_stage_opts(raw: dict[str, dict], base_dir: Path, phase: int) -> dict:
    """Options dataclasses for the context, keyed by its field names."""
    hints = get_type_hints(RunContext)
    built: dict[str, Any] = {}
    for name, attr in STAGE_OPTS_ATTR.items():
        data = dict(raw.get(name) or {})
        if name == "Classify" and isinstance(data.get("model_path"), str):
            data["model_path"] = _classifier_path(data["model_path"], base_dir)
        opts = _dataclass_from(hints[attr], data, name, base_dir)
        # The Input page's phase channel also drives Focus, unless the
        # config sets Focus's own.
        if name == "Focus" and "phase_channel" not in data:
            opts.phase_channel = phase
        built[attr] = opts
    return built


def _classifier_path(value: str, base_dir: Path) -> str:
    """A bundled preset's name, or a path to a ``.pth``, as an existing path."""
    from mycomorph.core.api import PRESET_MODELS, resolve_classifier_preset

    if value in PRESET_MODELS:
        return str(resolve_classifier_preset(value))
    path = _resolve_path(value, base_dir)
    if not path.is_file():
        raise ConfigError(
            f"Classify: model_path {value!r} is neither an existing file nor a "
            f"bundled preset ({', '.join(sorted(PRESET_MODELS))}). "
            "Use null for the rule-based filter alone."
        )
    return str(path)


# ── inputs ───────────────────────────────────────────────────────────────────

def _plate_inputs(block: dict, base_dir: Path) -> list[Path]:
    raw = block.get("czi_paths")
    if not raw and block.get("czi"):
        raw = [block["czi"]]
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise ConfigError(
            'Plate mode needs "czi_paths": the plate\'s CZI file(s). '
            'For single files or a table of files use "mode": "bulk".'
        )
    return [_resolve_path(_string(p, "czi_paths"), base_dir) for p in raw]


def _plate_layout(
    block: dict, czi_paths: list[Path], base_dir: Path, notes: list[str],
) -> PlateLayout:
    """The layout as the Plate layout page builds it: wells and scenes from
    the CZIs, labels from the CSV."""
    if not block.get("plate_layout_csv"):
        raise ConfigError(
            'Plate mode needs "plate_layout_csv": a CSV with a "well" column and '
            '"condition", "reporter", "mutant_or_drug" and "replica" labels.'
        )
    csv_path = _resolve_path(_string(block["plate_layout_csv"], "plate_layout_csv"), base_dir)
    if not csv_path.is_file():
        raise ConfigError(f"Plate layout CSV not found: {csv_path}")
    try:
        imported = PlateLayout.from_csv(csv_path)
    except Exception as e:  # noqa: BLE001
        raise ConfigError(f"Cannot read plate layout {csv_path}: {e}") from e

    existing = [p for p in czi_paths if p.exists()]
    if existing:
        conflicts: list[str] = []
        live = PlateLayout.from_czis(
            existing,
            on_conflict=lambda well, kept, skipped: conflicts.append(
                f"well {well} appears in {kept} and {skipped}; keeping {kept}"
            ),
        )
        if any(len(s) > 0 for s in live.df["scene_indices"]):
            notes.extend(conflicts)
            labelled = set(imported.active_rows()["well"].astype(str))
            unmatched = sorted(labelled - set(live.df["well"].astype(str)))
            if unmatched:
                notes.append(
                    f"{len(unmatched)} layout well(s) have no scenes in the "
                    f"CZI(s) and are skipped: {', '.join(unmatched[:8])}"
                    + ("…" if len(unmatched) > 8 else "")
                )
            live.merge_labels_from(imported)
            return live
    # No readable plate CZI (e.g. a re-run from Segment onwards after the
    # raw files moved): the CSV's own wells and scene columns stand.
    return imported


def _bulk_entries(block: dict, base_dir: Path) -> BulkLayout:
    has_csv = bool(block.get("bulk_entries_csv"))
    has_inline = block.get("entries") is not None
    if has_csv == has_inline:
        raise ConfigError(
            'Bulk mode needs one of "bulk_entries_csv" (a CSV with columns '
            f'{", ".join(ENTRY_COLUMNS)}) or an inline "entries" list.'
        )
    if has_csv:
        csv_path = _resolve_path(_string(block["bulk_entries_csv"], "bulk_entries_csv"), base_dir)
        if not csv_path.is_file():
            raise ConfigError(f"Entries CSV not found: {csv_path}")
        try:
            layout = BulkLayout.from_csv(csv_path)
        except Exception as e:  # noqa: BLE001
            raise ConfigError(f"Cannot read entries table {csv_path}: {e}") from e
        paths_base = csv_path.parent
    else:
        import pandas as pd

        rows = block["entries"]
        if not isinstance(rows, list) or not rows:
            raise ConfigError('"entries" must be a non-empty list of objects.')
        clean = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict) or not row.get("czi_path"):
                raise ConfigError(f'entries[{i}] must be an object with a "czi_path".')
            unknown = sorted(set(row) - set(ENTRY_COLUMNS))
            if unknown:
                raise ConfigError(
                    f"entries[{i}]: unknown key(s) {', '.join(unknown)}. "
                    f"Keys: {', '.join(ENTRY_COLUMNS)}."
                )
            clean.append({c: str(row.get(c, "") or "") for c in ENTRY_COLUMNS})
        layout = BulkLayout(df=pd.DataFrame(clean, columns=ENTRY_COLUMNS))
        paths_base = base_dir
    layout.df["czi_path"] = [
        str(_resolve_path(str(p), paths_base)) if str(p).strip() else ""
        for p in layout.df["czi_path"]
    ]
    if layout.active_rows().empty:
        raise ConfigError("No entry has a condition assigned, so there is nothing to process.")
    return layout


def _channels(
    block: dict, first_input: Optional[Path], notes: list[str],
) -> tuple[Optional[list[str]], int]:
    """``(channel_labels, phase_channel)``, read from the first input file
    when the config doesn't give them, as the Input page does."""
    from mycomorph.core.focus import io_source

    labels: Optional[list[str]]
    if "channel_labels" in block:
        labels = block["channel_labels"]
        if labels is not None and not (
            isinstance(labels, list) and all(isinstance(x, str) for x in labels)
        ):
            raise ConfigError('"channel_labels" must be a list of names, or null.')
    else:
        labels = None
        if first_input is not None and first_input.is_file():
            try:
                labels = list(io_source.channel_names(first_input)) or None
            except Exception as e:  # noqa: BLE001
                notes.append(f"Could not read channel names from {first_input.name}: {e}")
            if labels:
                notes.append(
                    f"Channel labels read from {first_input.name}: {', '.join(labels)}"
                )

    phase = block.get("phase_channel")
    if phase is None:
        phase = io_source.guess_phase_channel(labels) if labels else 0
        if labels:
            notes.append(f"Phase channel guessed from the labels: {phase} ({labels[phase]})")
    elif isinstance(phase, bool) or not isinstance(phase, int) or phase < 0:
        raise ConfigError('"phase_channel" must be a channel index (0, 1, …).')
    if labels and phase >= len(labels):
        raise ConfigError(
            f'"phase_channel" is {phase} but there are only {len(labels)} channel labels.'
        )
    return labels, phase


# ── JSON values → option dataclasses ─────────────────────────────────────────

def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f'"{where}" must be a path.')
    return value


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


def _dataclass_from(cls: type, data: dict, where: str, base_dir: Path) -> Any:
    hints = get_type_hints(cls)
    known = [f.name for f in fields(cls) if f.init]
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(
            f"{where}: unknown option(s) {', '.join(unknown)}. "
            f"Options: {', '.join(known)}."
        )
    return cls(**{
        key: _coerce(value, hints[key], f"{where}.{key}", base_dir)
        for key, value in data.items()
    })


def _coerce(value: Any, tp: Any, where: str, base_dir: Path) -> Any:
    """``value`` as the option's declared type, e.g. JSON lists back to tuples
    and whole numbers to floats, so options compare (and hash into provenance
    signatures) exactly as the GUI's do."""
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        members = get_args(tp)
        if value is None and type(None) in members:
            return None
        for member in members:
            if member is type(None):
                continue
            try:
                return _coerce(value, member, where, base_dir)
            except ConfigError:
                continue
        raise ConfigError(f"{where}: expected {_describe(tp)}, got {value!r}.")
    if is_dataclass(tp) and isinstance(tp, type):
        if not isinstance(value, dict):
            raise ConfigError(f"{where}: expected an object, got {value!r}.")
        return _dataclass_from(tp, value, where, base_dir)

    ok: bool
    converted: Any = value
    if tp is Path:
        ok = isinstance(value, str)
        if ok:
            converted = _resolve_path(value, base_dir)
    elif origin is tuple or origin is list:
        ok = isinstance(value, (list, tuple))
        if ok:
            members = get_args(tp)
            if origin is tuple and not (len(members) == 2 and members[1] is Ellipsis):
                if len(value) != len(members):
                    raise ConfigError(
                        f"{where}: expected {len(members)} values, got {len(value)}."
                    )
                item_types = list(members)
            else:
                item_types = [members[0] if members else Any] * len(value)
            converted = origin(
                _coerce(v, t, f"{where}[{i}]", base_dir)
                for i, (v, t) in enumerate(zip(value, item_types))
            )
    elif tp is bool:
        ok = isinstance(value, bool)
    elif tp is int:
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif tp is float:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        if ok:
            converted = float(value)
    elif tp is str:
        ok = isinstance(value, str)
    else:
        ok = True
    if not ok:
        raise ConfigError(f"{where}: expected {_describe(tp)}, got {value!r}.")
    return converted


def _describe(tp: Any) -> str:
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        return " or ".join(_describe(m) for m in get_args(tp))
    if tp is type(None):
        return "null"
    if tp is Path:
        return "a path"
    if origin is tuple or origin is list:
        return "a list"
    return {bool: "true or false", int: "a whole number", float: "a number",
            str: "text"}.get(tp, getattr(tp, "__name__", str(tp)))


# ─────────────────────────────────────────────────────────────────────────────
# Running, with progress as text
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunOutcome:
    status: str                            # "finished" | "stopped" | "failed"
    manifest_path: Optional[Path] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "finished"


def execute(
    run: PlateRun | BulkRun,
    *,
    log: Optional[Callable[[str], None]] = None,
    progress: bool = True,
) -> RunOutcome:
    """Execute ``run`` in the calling thread and report how it ended.

    ``log`` receives one line per event; ``progress=False`` leaves out the
    per-step messages inside each stage.
    """
    emit = log or (lambda _line: None)
    outcome = RunOutcome(status="failed", error="The run ended without reporting a result.")
    last: dict[str, str] = {}

    def on_progress(name: str, fraction: float, message: str) -> None:
        if not progress or not message or last.get(name) == message:
            return
        last[name] = message
        emit(f"[{name} {max(0.0, min(1.0, fraction)) * 100:3.0f}%] {message}")

    def on_finished(manifest: Path) -> None:
        outcome.status, outcome.manifest_path, outcome.error = "finished", Path(manifest), ""
        emit(f"Run complete. Manifest: {manifest}")

    def on_cancelled(manifest: Path) -> None:
        outcome.status, outcome.manifest_path, outcome.error = "stopped", Path(manifest), ""
        emit(f"Run stopped. Manifest: {manifest}")

    def on_failed(error: str) -> None:
        outcome.status, outcome.error = "failed", error
        emit(f"Run failed: {error}")

    run.stagesPlanned.connect(lambda names: emit("Stages: " + ", ".join(names)))
    run.stageStarted.connect(lambda name: emit(f"== {name} =="))
    run.stageProgress.connect(on_progress)
    run.stageFinished.connect(lambda name, n: emit(f"{name} finished: {n} output(s)"))
    run.stageCancelled.connect(lambda name: emit(f"{name} stopped"))
    run.runFinished.connect(on_finished)
    run.runCancelled.connect(on_cancelled)
    run.runFailed.connect(on_failed)
    run.execute()
    return outcome
