"""Test package path setup for root modules."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

path_text = str(PROJECT_ROOT)
if path_text not in sys.path:
    sys.path.insert(0, path_text)

import market_edge  # noqa: E402,F401  (bundles v11/v13-v27/v28-v38/v39; populates sys.modules)
