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
class MistralClient:
    """
    Mistral client using Vertex AI's rawPredict API.
    
    Authentication: Uses Google Cloud service account credentials.
    
    Required:
    - GOOGLE_CLOUD_PROJECT: Your GCP project ID
    - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON
    
    Note: Mistral models are available in europe-west4 and us-central1.
    """
    
    project_id: Optional[str] = None
    region: str = "us-central1"  # or "europe-west4"
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
            console.print(f"[green]✓[/] Mistral client initialized (project: {self.project_id}, region: {self.region})")
        except Exception as e:
            console.print(f"[red]ERROR[/]: Failed to initialize Mistral client: {e}")
            raise
    
    def _get_access_token(self) -> str:
        """Get a fresh access token."""
        import google.auth.transport.requests
        
        auth_req = google.auth.transport.requests.Request()
        self._credentials.refresh(auth_req)
        return self._credentials.token
    
    def _get_endpoint_url(self, model: str) -> str:
        """Get the Vertex AI endpoint URL for Mistral rawPredict."""
        # Extract just the model name without publisher prefix
        model_name = model.split("/")[-1] if "/" in model else model
        return f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/mistralai/models/{model_name}:rawPredict"
    
    def generate_code(
        self,
        model: str = "codestral-2",
        temperature: float = 0.0,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 2000,
        max_retries: int = 5,
        initial_backoff_s: float = 2.0,
        json_mode: bool = False,
        # FIM-specific parameters
        prompt: Optional[str] = None,
        suffix: Optional[str] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Generate code using Mistral model via Vertex AI.
        
        Args:
            model: Mistral model name:
                - "codestral-2" (latest, best for code)
                - "codestral-2501" (previous version)
            temperature: Sampling temperature (0.0 = deterministic)
            messages: List of message dicts for chat mode (role, content)
            max_tokens: Maximum tokens to generate
            max_retries: Number of retry attempts
            initial_backoff_s: Initial backoff time in seconds
            json_mode: Whether to request JSON output
            prompt: For FIM mode - the code prefix
            suffix: For FIM mode - the code suffix (fills between prompt and suffix)
            stop: Optional stop tokens
            
        Returns:
            Generated text content
        """
        # Normalize model name
        if model.startswith("mistral/") or model.startswith("mistralai/"):
            model = model.split("/")[-1]
        
        # Determine if FIM mode or chat mode
        if prompt is not None:
            # Fill-in-the-middle mode
            payload = {
                "model": model,
                "prompt": prompt,
            }
            if suffix:
                payload["suffix"] = suffix
            if stop:
                payload["stop"] = stop
            if max_tokens:
                payload["max_tokens"] = max_tokens
            if temperature > 0:
                payload["temperature"] = temperature
        else:
            # Chat mode
            messages = messages or []
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if temperature > 0:
                payload["temperature"] = temperature
            if stop:
                payload["stop"] = stop
        
        url = self._get_endpoint_url(model)
        
        attempt = 0
        backoff = initial_backoff_s
        
        while True:
            try:
                token = self._get_access_token()
                headers = {
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                }
                
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                
                if response.status_code == 200:
                    result = response.json()
                    # Extract content from response
                    if result.get("choices") and len(result["choices"]) > 0:
                        choice = result["choices"][0]
                        # Chat mode returns message.content, FIM might return text directly
                        if "message" in choice:
                            return choice["message"]["content"].strip()
                        elif "text" in choice:
                            return choice["text"].strip()
                        else:
                            return str(choice).strip()
                    else:
                        raise ValueError("Empty response from Mistral")
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
                    console.print(f"[red]Mistral request failed[/]: {e}")
                    raise
                
                # Jittered exponential backoff
                sleep_s = backoff * (1.0 + 0.25 * random())
                console.print(f"[yellow]Retry {attempt}/{max_retries}[/] after {sleep_s:.1f}s: {e}")
                time.sleep(sleep_s)
                backoff *= 2
    
    def fill_in_middle(
        self,
        prompt: str,
        suffix: str = "",
        model: str = "codestral-2",
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Fill-in-the-middle code completion.
        
        Args:
            prompt: The code prefix (what comes before the cursor)
            suffix: The code suffix (what comes after the cursor)
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            stop: Stop tokens
            
        Returns:
            Generated code to fill the gap
        """
        return self.generate_code(
            model=model,
            prompt=prompt,
            suffix=suffix,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
    
    def list_models(self) -> List[str]:
        """List available Mistral models on Vertex AI."""
        return [
            "codestral-2",           # Latest Codestral (25.08)
            "codestral-2501",        # Previous version (25.01)
            "mistral-large-2411",    # Mistral Large
            "mistral-small-2503",    # Mistral Small
        ]


# Convenience function
def create_mistral_client(
    project_id: Optional[str] = None,
    region: str = "us-central1",
    credentials_path: Optional[str] = None,
) -> MistralClient:
    """
    Create a Mistral client with the specified configuration.
    
    Args:
        project_id: GCP project ID (or set GOOGLE_CLOUD_PROJECT env var)
        region: Vertex AI region (default: us-central1, also supports europe-west4)
        credentials_path: Path to service account JSON file
        
    Returns:
        Configured MistralClient instance
    """
    return MistralClient(
        project_id=project_id,
        region=region,
        credentials_path=credentials_path,
    )

