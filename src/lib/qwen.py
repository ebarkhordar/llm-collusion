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
class QwenClient:
    """
    Qwen client using Vertex AI's OpenAI-compatible API.
    
    Qwen3 Coder is Alibaba Cloud's advanced agentic code model:
    - 480B total params, 35B active (MoE architecture)
    - 256K native context, extensible to 1M tokens
    - Specialized for agentic coding, browser use, tool use
    
    Authentication: Uses Google Cloud service account credentials.
    
    Required:
    - GOOGLE_CLOUD_PROJECT: Your GCP project ID
    - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON
    
    Note: Qwen3 Coder is available in us-south1 (Dallas) region.
    """
    
    project_id: Optional[str] = None
    region: str = "us-south1"  # Qwen3 Coder is in us-south1
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
            console.print(f"[green]✓[/] Qwen client initialized (project: {self.project_id}, region: {self.region})")
        except Exception as e:
            console.print(f"[red]ERROR[/]: Failed to initialize Qwen client: {e}")
            raise
    
    def _get_access_token(self) -> str:
        """Get a fresh access token."""
        import google.auth.transport.requests
        
        auth_req = google.auth.transport.requests.Request()
        self._credentials.refresh(auth_req)
        return self._credentials.token
    
    def _get_endpoint_url(self) -> str:
        """Get the Vertex AI endpoint URL for Qwen."""
        return f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/endpoints/openapi/chat/completions"
    
    def generate_code(
        self,
        model: str = "qwen/qwen3-coder-480b-a35b-instruct-maas",
        temperature: float = 0.0,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 2000,
        max_retries: int = 5,
        initial_backoff_s: float = 2.0,
        json_mode: bool = False,
    ) -> str:
        """
        Generate code using Qwen3 Coder model via Vertex AI.
        
        Args:
            model: Qwen model name:
                - "qwen/qwen3-coder-480b-a35b-instruct-maas" (480B MoE, 35B active)
            temperature: Sampling temperature (0.0 = deterministic)
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate
            max_retries: Number of retry attempts
            initial_backoff_s: Initial backoff time in seconds
            json_mode: Whether to request JSON output
            
        Returns:
            Generated text content
        """
        messages = messages or []
        
        # Normalize model name - add qwen/ prefix if not present
        if not model.startswith("qwen/"):
            model = f"qwen/{model}"
        
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
                
                response = requests.post(url, headers=headers, json=payload, timeout=180)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("choices") and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"].strip()
                    else:
                        raise ValueError("Empty response from Qwen")
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
                    console.print(f"[red]Qwen request failed[/]: {e}")
                    raise
                
                # Jittered exponential backoff
                sleep_s = backoff * (1.0 + 0.25 * random())
                console.print(f"[yellow]Retry {attempt}/{max_retries}[/] after {sleep_s:.1f}s: {e}")
                time.sleep(sleep_s)
                backoff *= 2
    
    def list_models(self) -> List[str]:
        """List available Qwen models on Vertex AI."""
        return [
            "qwen/qwen3-coder-480b-a35b-instruct-maas",  # 480B MoE, 35B active
        ]


# Convenience function
def create_qwen_client(
    project_id: Optional[str] = None,
    region: str = "us-south1",
    credentials_path: Optional[str] = None,
) -> QwenClient:
    """
    Create a Qwen client with the specified configuration.
    
    Args:
        project_id: GCP project ID (or set GOOGLE_CLOUD_PROJECT env var)
        region: Vertex AI region (default: us-south1 for Qwen3 Coder)
        credentials_path: Path to service account JSON file
        
    Returns:
        Configured QwenClient instance
    """
    return QwenClient(
        project_id=project_id,
        region=region,
        credentials_path=credentials_path,
    )

