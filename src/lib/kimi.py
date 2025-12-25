from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from random import random
from typing import Optional, List, Dict, Any

import requests
from rich.console import Console

# optional: load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

console = Console()


@dataclass
class KimiClient:
    """
    Kimi K2 client using Vertex AI's OpenAI-compatible API.
    
    Kimi K2 Thinking is Moonshot AI's most capable open-source thinking model:
    - 1T total params, 32B active (MoE architecture)
    - 256K context window
    - Native INT4 quantization for 2x speed
    - Deep thinking with tool orchestration (200-300 sequential tool calls)
    
    Authentication: Uses Google Cloud service account credentials.
    
    Required:
    - GOOGLE_CLOUD_PROJECT: Your GCP project ID
    - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON
    """
    
    project_id: Optional[str] = None
    region: str = "global"  # Kimi K2 uses global region
    credentials_path: Optional[str] = None
    _credentials: Any = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        # Get project ID from environment if not provided
        if not self.project_id:
            self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        
        # Get credentials path
        if not self.credentials_path:
            self.credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        if not self.project_id:
            console.print("[yellow]WARNING[/]: GOOGLE_CLOUD_PROJECT not set; requests may fail.")
        
        if not self.credentials_path:
            console.print("[yellow]WARNING[/]: GOOGLE_APPLICATION_CREDENTIALS not set; requests may fail.")
        
        self._init_credentials()
    
    def _init_credentials(self) -> None:
        """Initialize Google Cloud credentials."""
        try:
            from google.oauth2 import service_account
            
            SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
            self._credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=SCOPES
            )
            console.print(f"[green]✓[/] Kimi client initialized (project: {self.project_id}, region: {self.region})")
        except Exception as e:
            console.print(f"[red]ERROR[/]: Failed to initialize Kimi client: {e}")
            raise
    
    def _get_access_token(self) -> str:
        """Get a fresh access token."""
        import google.auth.transport.requests
        
        auth_req = google.auth.transport.requests.Request()
        self._credentials.refresh(auth_req)
        return self._credentials.token
    
    def _get_endpoint_url(self) -> str:
        """Get the Vertex AI endpoint URL for Kimi."""
        return f"https://aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/endpoints/openapi/chat/completions"
    
    def generate_code(
        self,
        model: str = "moonshotai/kimi-k2-thinking-maas",
        temperature: float = 0.0,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 4000,  # Higher default for thinking models
        max_retries: int = 5,
        initial_backoff_s: float = 2.0,
        json_mode: bool = False,
    ) -> str:
        """
        Generate code using Kimi K2 Thinking model via Vertex AI.
        
        Note: Kimi K2 is a "thinking" model that reasons step-by-step.
        It needs more tokens to complete thinking and produce final answer.
        
        Args:
            model: Kimi model name:
                - "moonshotai/kimi-k2-thinking-maas" (1T MoE, 32B active)
            temperature: Sampling temperature (0.0 = deterministic)
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens (default 4000 for thinking + answer)
            max_retries: Number of retry attempts
            initial_backoff_s: Initial backoff time in seconds
            json_mode: Whether to request JSON output
            
        Returns:
            Generated text content (final answer, not reasoning)
        """
        messages = messages or []
        
        # Normalize model name - add moonshotai/ prefix if not present
        if not model.startswith("moonshotai/"):
            model = f"moonshotai/{model}"
        
        # Build request payload (OpenAI-compatible format)
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        
        if temperature > 0:
            payload["temperature"] = temperature
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        
        url = self._get_endpoint_url()
        
        attempt = 0
        backoff = initial_backoff_s
        
        while True:
            try:
                token = self._get_access_token()
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                }
                
                response = requests.post(url, headers=headers, json=payload, timeout=300)  # Longer timeout for thinking
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("choices") and len(result["choices"]) > 0:
                        message = result["choices"][0].get("message", {})
                        # Kimi K2 Thinking has both content and reasoning_content
                        content = message.get("content")
                        if content:
                            return content.strip()
                        # Fall back to reasoning_content if content is empty
                        reasoning = message.get("reasoning_content")
                        if reasoning:
                            return reasoning.strip()
                        raise ValueError("Empty content in Kimi response")
                    else:
                        raise ValueError("Empty response from Kimi")
                else:
                    raise Exception(f"HTTP {response.status_code}: {response.text[:500]}")
                    
            except Exception as e:
                attempt += 1
                error_str = str(e).lower()
                
                # Check if it's a retryable error
                is_retryable = any(x in error_str for x in [
                    "429", "quota", "rate", "resource exhausted",
                    "500", "502", "503", "504", "unavailable", "timeout",
                    "overloaded"
                ])
                
                if attempt > max_retries or not is_retryable:
                    console.print(f"[red]Kimi request failed[/]: {e}")
                    raise
                
                # Jittered exponential backoff
                sleep_s = backoff * (1.0 + 0.25 * random())
                console.print(f"[yellow]Retry {attempt}/{max_retries}[/] after {sleep_s:.1f}s: {e}")
                time.sleep(sleep_s)
                backoff *= 2
    
    def list_models(self) -> List[str]:
        """List available Kimi models on Vertex AI."""
        return [
            "moonshotai/kimi-k2-thinking-maas",  # 1T MoE, 32B active, thinking model
        ]


# Convenience function
def create_kimi_client(
    project_id: Optional[str] = None,
    region: str = "global",
    credentials_path: Optional[str] = None,
) -> KimiClient:
    """
    Create a Kimi client with the specified configuration.
    
    Args:
        project_id: GCP project ID (or set GOOGLE_CLOUD_PROJECT env var)
        region: Vertex AI region (default: global for Kimi K2)
        credentials_path: Path to service account JSON file
        
    Returns:
        Configured KimiClient instance
    """
    return KimiClient(
        project_id=project_id,
        region=region,
        credentials_path=credentials_path,
    )

