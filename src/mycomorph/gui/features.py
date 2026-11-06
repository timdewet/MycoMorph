"""Feature switches for the desktop app.

The CNN embeddings work (the Embeddings stage and page, the "Train
embeddings" input mode, and the CNN embedding views on the Analysis page) is
not part of the 0.7 release: the application note covers focus, segmentation,
the quality filter, measurements and foci. The code stays, but the app hides
it unless ``MYCOMORPH_EMBEDDINGS=1`` is set in the environment.
"""

from __future__ import annotations

import os

EMBEDDINGS_ENV = "MYCOMORPH_EMBEDDINGS"


def embeddings_enabled() -> bool:
    """True when the CNN embeddings stage, page and views should be shown."""
    return os.environ.get(EMBEDDINGS_ENV, "").strip() == "1"
