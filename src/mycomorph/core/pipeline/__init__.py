"""The pipeline behind both front-ends: stage adapters, run contexts, layouts
and the run logic, with no Qt.

- :mod:`.stages` wraps the ``mycomorph.core.api`` stage functions with reuse,
  provenance sidecars and file handling.
- :mod:`.runner` walks the enabled stages and writes ``run_manifest.json``.
  The GUI runs it on a ``QThread`` (``mycomorph.gui.pipeline.runner``);
  ``mycomorph-cli run`` runs it in the terminal.
- :mod:`.config` builds a run from a JSON config or an earlier manifest.

Submodules are imported on use, so importing this package stays cheap.
"""
