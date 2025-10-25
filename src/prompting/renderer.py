from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def render_prompt(path: Path, **kwargs: Any) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Prompt YAML must be a mapping with 'system' and 'user'")

    def _render(text: str) -> str:
        out = text
        for k, v in kwargs.items():
            out = out.replace(f"{{{{ {k} }}}}", str(v))
        return out

    system = _render(str(data.get("system", "")))
    user = _render(str(data.get("user", "")))
    return {"system": system, "user": user}


