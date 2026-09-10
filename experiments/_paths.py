"""Repo-root anchoring for the experiment entrypoints.

Importing this module puts the repository root on ``sys.path`` (so ``src`` is
importable) and exposes the canonical input locations. Anchoring on ``__file__``
rather than the process working directory lets the scripts run from anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "configs" / "config.yaml"
DATA = ROOT / "data"
PROMPTS = ROOT / "prompts"


def data_dir(configured: str | None = None) -> Path:
    """Resolve the configured data directory against the repo root."""
    if not configured:
        return DATA
    p = Path(configured)
    return p if p.is_absolute() else ROOT / p
