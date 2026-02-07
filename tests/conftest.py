from __future__ import annotations

import sys
from pathlib import Path

# Ensure `src/` is on sys.path so tests work without installing the package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
