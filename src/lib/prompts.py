from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def render_prompt(path: Path, **kwargs: Any) -> Dict[str, str]:
    """Render a prompt template from markdown files."""
    
    def _render(text: str) -> str:
        out = text
        for k, v in kwargs.items():
            out = out.replace(f"{{{{ {k} }}}}", str(v))
        return out

    with path.open("r", encoding="utf-8") as f:
        content = f.read()
    
    user_text = _render(content)
    return {"user": user_text}


