"""Commodity Open 2026 dataset builders. One module per build lane; see SPEC.md."""

import os

#: Environment variable that names the dataset build tree.
BUILD_ROOT_ENV = "SWING_BUILD_ROOT"

#: Root of the build tree: ``raw/`` captures, ``build/<lane>/`` lane outputs and assembled
#: ``pack/<version>/`` releases. Every lane derives its paths from this. Point
#: ``SWING_BUILD_ROOT`` at your own tree; without it the lanes use ``./swing-dataset`` under the
#: working directory. Reading the pack does not need it: ``swing_dataset.pack.pit.PointInTime``
#: takes the pack directory as an argument.
BUILD_ROOT = os.environ.get(BUILD_ROOT_ENV) or "./swing-dataset"
