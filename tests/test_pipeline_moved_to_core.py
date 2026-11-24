"""The pipeline's stage adapters, run contexts and layouts moved from
``mycomorph.gui.pipeline`` to ``mycomorph.core.pipeline``. The old module
names stay importable as aliases of the moved modules."""

from __future__ import annotations

import importlib

import pytest

MOVED = ("bulk_layout", "cancellation", "context", "layout", "stages")


@pytest.mark.parametrize("name", MOVED)
def test_old_module_name_is_the_core_module(name: str):
    old = importlib.import_module(f"mycomorph.gui.pipeline.{name}")
    new = importlib.import_module(f"mycomorph.core.pipeline.{name}")
    assert old is new


def test_old_names_work_in_every_import_form():
    import mycomorph.gui.pipeline.stages as by_path
    from mycomorph.core.pipeline import stages as core_stages
    from mycomorph.gui.pipeline import stages as by_attribute
    from mycomorph.gui.pipeline.context import RunContext
    from mycomorph.gui.pipeline.stages import SegmentStage

    assert by_path is by_attribute is core_stages
    assert SegmentStage is core_stages.SegmentStage
    assert RunContext.__module__ == "mycomorph.core.pipeline.context"
