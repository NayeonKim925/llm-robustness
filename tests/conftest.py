"""Pytest bootstrap.

When the package is installed (``pip install -e .``) this file is a no-op. As a
courtesy it also makes ``src/`` importable when tests are run from a bare
checkout without installing, so the test files themselves stay import-hack-free.
"""

import os
import sys

_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
