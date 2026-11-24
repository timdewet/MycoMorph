"""The GUI's side of the pipeline: :mod:`.runner`, the Qt runners.

Everything else that used to live here (stage adapters, run contexts,
layouts, cancellation) has no Qt in it and moved to
:mod:`mycomorph.core.pipeline`, where ``mycomorph-cli run`` uses it without
importing the GUI. The old module names are kept as aliases of the moved
modules, so ``from mycomorph.gui.pipeline.stages import SegmentStage`` and the
like keep working; new code should import from ``mycomorph.core.pipeline``.
"""

import sys

from mycomorph.core.pipeline import bulk_layout, cancellation, context, layout, stages

for _module in (bulk_layout, cancellation, context, layout, stages):
    sys.modules[f"{__name__}.{_module.__name__.rpartition('.')[2]}"] = _module
del _module
