from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List, Dict

import requests
from rich.console import Console
import time
from random import random

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

    def generate_code(
        self,
        model: str,
        temperature: float = 0.0,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = 2000,
        max_retries: int = 5,
        initial_backoff_s: float = 1.0,
    ) -> str:
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

        body: Dict[str, object] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        attempt = 0
        backoff = initial_backoff_s
        while True:
            try:
                resp = requests.post(url, json=body, headers=headers, timeout=120)
                # Retry on 429/5xx
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:  # noqa: BLE001
                attempt += 1
                if attempt > max_retries:
                    console.print(f"[red]OpenRouter request failed after retries[/]: {e}")
                    raise
                # jittered exponential backoff
                sleep_s = backoff * (1.0 + 0.25 * random())
                time.sleep(sleep_s)
                backoff *= 2


