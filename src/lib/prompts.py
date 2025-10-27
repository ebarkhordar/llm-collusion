from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def render_prompt(path: Path, **kwargs: Any) -> Dict[str, str]:
    """Render a prompt template from YAML or markdown files."""
    
    def _render(text: str) -> str:
        out = text
        for k, v in kwargs.items():
            out = out.replace(f"{{{{ {k} }}}}", str(v))
        return out

    # Handle markdown files
    if path.suffix == ".md":
        with path.open("r", encoding="utf-8") as f:
            content = f.read()
        user_text = _render(content)
        return {"user": user_text}

    # Handle YAML files
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Support README-style single-block prompts or legacy dict formats
    if isinstance(data, str):
        user_text = _render(data)
        return {"user": user_text}

    if isinstance(data, dict):
        # Prefer a single 'prompt' or 'user' field; if 'system' exists, merge it for backward compatibility
        raw_prompt = str(data.get("prompt", data.get("user", "")))
        system_text = str(data.get("system", ""))
        merged = (system_text.strip() + "\n\n" + raw_prompt.strip()).strip() if (system_text or raw_prompt) else ""
        user_text = _render(merged or raw_prompt)
        return {"user": user_text}

    raise ValueError("Prompt file must be a YAML string or a mapping with 'prompt'/'user' (system optional)")


