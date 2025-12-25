from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Union

from rich.console import Console

from .openrouter import OpenRouterClient
from .gemini import GeminiClient
from .claude import ClaudeClient
from .llama import LlamaClient

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

# Claude models on Vertex AI (use Anthropic Vertex SDK)
CLAUDE_VERTEX_MODELS = {
    "claude-sonnet-4-5",
    "claude-sonnet-4-5@20250929",
    "claude-3-5-sonnet-v2",
    "claude-3-5-sonnet-v2@20241022",
    "claude-3-5-haiku",
    "claude-3-5-haiku@20241022",
    "claude-3-opus",
    "claude-3-opus@20240229",
}

# Llama 4 models on Vertex AI
LLAMA_VERTEX_MODELS = {
    "llama-4-scout",
    "llama-4-maverick",
    "llama-4-scout-17b-16e-instruct-maas",
    "llama-4-maverick-17b-128e-instruct-maas",
    "meta/llama-4-scout-17b-16e-instruct-maas",
    "meta/llama-4-maverick-17b-128e-instruct-maas",
}


@dataclass
class LLMClient:
    """
    Unified LLM client that routes requests to the appropriate backend.
    
    - Claude models with vertex/ prefix -> Anthropic Vertex AI SDK
    - Gemini models (gemini-*) -> Google Vertex AI
    - All other models -> OpenRouter
    
    Configuration:
        OpenRouter: Set OPENROUTER_API_KEY env var
        Gemini/Claude Vertex: Set GOOGLE_CLOUD_PROJECT and GOOGLE_APPLICATION_CREDENTIALS env vars
    """
    
    # OpenRouter config
    openrouter_api_key: Optional[str] = None
    
    # Google Cloud config (for Gemini and Claude Vertex)
    google_project_id: Optional[str] = None
    google_location: str = "us-central1"
    google_credentials_path: Optional[str] = None
    
    # Internal clients (lazy initialized)
    _openrouter_client: Optional[OpenRouterClient] = None
    _gemini_client: Optional[GeminiClient] = None
    _claude_client: Optional[ClaudeClient] = None
    _llama_client: Optional[LlamaClient] = None
    
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
    
    def _get_claude_client(self) -> ClaudeClient:
        if self._claude_client is None:
            self._claude_client = ClaudeClient(
                project_id=self.google_project_id,
                region="global",  # Claude on Vertex uses global region
            )
        return self._claude_client
    
    def _get_llama_client(self) -> LlamaClient:
        if self._llama_client is None:
            self._llama_client = LlamaClient(
                project_id=self.google_project_id,
                region="us-east5",  # Llama 4 only in us-east5
                credentials_path=self.google_credentials_path,
            )
        return self._llama_client
    
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
    
    def _is_claude_vertex_model(self, model: str) -> bool:
        """Check if the model should be routed to Claude on Vertex AI."""
        model_lower = model.lower()
        # Check for vertex/ prefix with claude - explicit routing to Vertex AI
        if model_lower.startswith("vertex/claude"):
            return True
        # Check for vertex-claude or claude-vertex prefix
        if "vertex" in model_lower and "claude" in model_lower:
            return True
        return False
    
    def _is_llama_vertex_model(self, model: str) -> bool:
        """Check if the model should be routed to Llama on Vertex AI."""
        model_lower = model.lower()
        # Check for vertex/llama or vertex/meta prefix
        if model_lower.startswith("vertex/llama") or model_lower.startswith("vertex/meta"):
            return True
        # Check exact match in known models
        if model in LLAMA_VERTEX_MODELS or model_lower in LLAMA_VERTEX_MODELS:
            return True
        # Check for llama-4 pattern
        if "llama-4" in model_lower or "llama4" in model_lower:
            return True
        return False
    
    def _normalize_model_name(self, model: str, backend: str) -> str:
        """Normalize model name for the appropriate backend."""
        # Remove prefixes
        if model.lower().startswith("google/"):
            return model[7:]  # Remove "google/" prefix
        if model.lower().startswith("vertex/"):
            model = model[7:]  # Remove "vertex/" prefix
        
        # Backend-specific normalization
        if backend == "claude":
            # Add version suffix if not present for Claude
            if "@" not in model:
                if model == "claude-sonnet-4-5":
                    return "claude-sonnet-4-5@20250929"
                elif model == "claude-3-5-sonnet-v2":
                    return "claude-3-5-sonnet-v2@20241022"
                elif model == "claude-3-5-haiku":
                    return "claude-3-5-haiku@20241022"
                elif model == "claude-3-opus":
                    return "claude-3-opus@20240229"
        elif backend == "llama":
            # Normalize Llama model names to full format
            model_lower = model.lower()
            if model_lower in ("llama-4-scout", "llama4-scout"):
                return "meta/llama-4-scout-17b-16e-instruct-maas"
            elif model_lower in ("llama-4-maverick", "llama4-maverick"):
                return "meta/llama-4-maverick-17b-128e-instruct-maas"
            # Add meta/ prefix if not present
            if not model.startswith("meta/"):
                return f"meta/{model}"
        
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
            model: Model identifier. Use prefixes to control routing:
                - "vertex/claude-sonnet-4-5" -> Claude via Vertex AI
                - "vertex/llama-4-scout" -> Llama 4 via Vertex AI
                - "google/gemini-2.5-flash" or "gemini-*" -> Gemini via Vertex AI
                - "anthropic/claude-*" -> Claude via OpenRouter
                - Other models -> OpenRouter
            temperature: Sampling temperature
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate
            max_retries: Number of retry attempts
            initial_backoff_s: Initial backoff time in seconds
            json_mode: Whether to request JSON output
            
        Returns:
            Generated text content
        """
        # Determine routing
        if self._is_claude_vertex_model(model):
            client = self._get_claude_client()
            normalized_model = self._normalize_model_name(model, "claude")
            console.print(f"[dim]Using Claude Vertex backend for {normalized_model}[/]")
        elif self._is_llama_vertex_model(model):
            client = self._get_llama_client()
            normalized_model = self._normalize_model_name(model, "llama")
            console.print(f"[dim]Using Llama Vertex backend for {normalized_model}[/]")
        elif self._is_gemini_model(model):
            client = self._get_gemini_client()
            normalized_model = self._normalize_model_name(model, "gemini")
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

