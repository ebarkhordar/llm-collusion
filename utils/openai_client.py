from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import requests
from rich.console import Console

# optional: load .env if available
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

console = Console()


@dataclass
class OpenRouterClient:
    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not self.api_key:
            console.print("[yellow]WARNING[/]: OPENROUTER_API_KEY not set; requests will fail.")

    def generate_code(self, prompt: str, model: str, temperature: float = 0.2) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # OpenRouter recommends adding identification headers when possible
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        title = os.getenv("OPENROUTER_X_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title

        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a coding assistant. Return only code unless instructed otherwise."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        resp = requests.post(url, json=body, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            console.print(f"[red]Unexpected response format[/]: {data}")
            raise e
