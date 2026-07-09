"""Import path setup for tests run with ``unittest discover -s tests``."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

path_text = str(PROJECT_ROOT)
if path_text not in sys.path:
    sys.path.insert(0, path_text)
