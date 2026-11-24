"""Old home of the pipeline modules, kept as aliases.

The stage adapters, run contexts, layouts, cancellation and runner moved to
:mod:`mycomorph.core.pipeline`. The old module names are aliases of the moved
modules, so ``from mycomorph.gui.pipeline.stages import SegmentStage`` and the
like keep working; new code should import from ``mycomorph.core.pipeline``.
"""

import sys

from mycomorph.core.pipeline import (
    bulk_layout,
    cancellation,
    context,
    layout,
    runner,
    stages,
)

for _module in (bulk_layout, cancellation, context, layout, runner, stages):
    sys.modules[f"{__name__}.{_module.__name__.rpartition('.')[2]}"] = _module
del _module
