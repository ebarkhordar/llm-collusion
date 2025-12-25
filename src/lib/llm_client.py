from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Union

from rich.console import Console

from .openrouter import OpenRouterClient
from .gemini import GeminiClient

console = Console()


# Models that should be routed to Gemini (Vertex AI) instead of OpenRouter
GEMINI_MODELS = {
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
}


@dataclass
class LLMClient:
    """
    Unified LLM client that routes requests to the appropriate backend.
    
    - Gemini models (gemini-*) -> Google Vertex AI
    - All other models -> OpenRouter
    
    Configuration:
        OpenRouter: Set OPENROUTER_API_KEY env var
        Gemini: Set GOOGLE_CLOUD_PROJECT and GOOGLE_APPLICATION_CREDENTIALS env vars
    """
    
    # OpenRouter config
    openrouter_api_key: Optional[str] = None
    
    # Gemini config
    google_project_id: Optional[str] = None
    google_location: str = "us-central1"
    google_credentials_path: Optional[str] = None
    
    # Internal clients (lazy initialized)
    _openrouter_client: Optional[OpenRouterClient] = None
    _gemini_client: Optional[GeminiClient] = None
    
    def _get_openrouter_client(self) -> OpenRouterClient:
        if self._openrouter_client is None:
            self._openrouter_client = OpenRouterClient(api_key=self.openrouter_api_key)
        return self._openrouter_client
    
    def _get_gemini_client(self) -> GeminiClient:
        if self._gemini_client is None:
            self._gemini_client = GeminiClient(
                project_id=self.google_project_id,
                location=self.google_location,
                credentials_path=self.google_credentials_path,
            )
        return self._gemini_client
    
    def _is_gemini_model(self, model: str) -> bool:
        """Check if the model should be routed to Gemini."""
        # Check exact match
        if model in GEMINI_MODELS:
            return True
        # Check if model name starts with gemini (handles versioned names)
        model_lower = model.lower()
        if model_lower.startswith("gemini"):
            return True
        # Check for google/ prefix (from config)
        if model_lower.startswith("google/gemini"):
            return True
        return False
    
    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name for the appropriate backend."""
        # Remove google/ prefix for Gemini
        if model.lower().startswith("google/"):
            return model[7:]  # Remove "google/" prefix
        return model
    
    def generate_code(
        self,
        model: str,
        temperature: float = 0.0,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = 2000,
        max_retries: int = 5,
        initial_backoff_s: float = 1.0,
        json_mode: bool = False,
    ) -> str:
        """
        Generate code using the appropriate backend based on the model.
        
        Args:
            model: Model identifier (e.g., "google/gemini-2.5-flash", "anthropic/claude-3-opus")
            temperature: Sampling temperature
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate
            max_retries: Number of retry attempts
            initial_backoff_s: Initial backoff time in seconds
            json_mode: Whether to request JSON output
            
        Returns:
            Generated text content
        """
        if self._is_gemini_model(model):
            client = self._get_gemini_client()
            normalized_model = self._normalize_model_name(model)
            console.print(f"[dim]Using Gemini backend for {normalized_model}[/]")
        else:
            client = self._get_openrouter_client()
            normalized_model = model
            console.print(f"[dim]Using OpenRouter backend for {normalized_model}[/]")
        
        return client.generate_code(
            model=normalized_model,
            temperature=temperature,
            messages=messages,
            max_tokens=max_tokens,
            max_retries=max_retries,
            initial_backoff_s=initial_backoff_s,
            json_mode=json_mode,
        )


def create_llm_client(
    openrouter_api_key: Optional[str] = None,
    google_project_id: Optional[str] = None,
    google_location: str = "us-central1",
    google_credentials_path: Optional[str] = None,
) -> LLMClient:
    """
    Create a unified LLM client.
    
    Args:
        openrouter_api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
        google_project_id: GCP project ID (or set GOOGLE_CLOUD_PROJECT env var)
        google_location: GCP region (default: us-central1)
        google_credentials_path: Path to service account JSON file
        
    Returns:
        Configured LLMClient instance
    """
    return LLMClient(
        openrouter_api_key=openrouter_api_key,
        google_project_id=google_project_id,
        google_location=google_location,
        google_credentials_path=google_credentials_path,
    )

