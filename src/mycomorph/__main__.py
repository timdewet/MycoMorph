"""``python -m mycomorph`` and the desktop bundle's entry point.

Opens MycoMorph's GUI; ``--modelprep`` opens ModelPrep (the cell-quality
labeller and trainer) instead. See ``mycomorph.gui.app``.
"""

from mycomorph.gui.app import main

if __name__ == "__main__":
    raise SystemExit(main())
