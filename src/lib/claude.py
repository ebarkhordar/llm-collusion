from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from random import random
from typing import Optional, List, Dict, Any

from rich.console import Console

# optional: load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

console = Console()


@dataclass
class ClaudeClient:
    """
    Claude client using Anthropic's Vertex AI SDK.
    
    Authentication: Uses Google Cloud credentials (ADC or service account).
    
    Required:
    - GOOGLE_CLOUD_PROJECT: Your GCP project ID
    - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON (optional if ADC)
    """
    
    project_id: Optional[str] = None
    region: str = "global"  # Claude on Vertex uses "global" region
    _client: Any = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        # Get project ID from environment if not provided
        if not self.project_id:
            self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        
        if not self.project_id:
            console.print("[yellow]WARNING[/]: GOOGLE_CLOUD_PROJECT not set; requests may fail.")
        
        self._init_client()
    
    def _init_client(self) -> None:
        """Initialize the Anthropic Vertex client."""
        try:
            from anthropic import AnthropicVertex
            
            self._client = AnthropicVertex(
                region=self.region,
                project_id=self.project_id,
            )
            console.print(f"[green]✓[/] Claude client initialized (project: {self.project_id}, region: {self.region})")
        except ImportError:
            console.print("[red]ERROR[/]: anthropic[vertex] not installed. Run: pip install 'anthropic[vertex]'")
            raise
        except Exception as e:
            console.print(f"[red]ERROR[/]: Failed to initialize Claude client: {e}")
            raise
    
    def generate_code(
        self,
        model: str = "claude-sonnet-4-5@20250929",
        temperature: float = 0.0,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 2000,
        max_retries: int = 5,
        initial_backoff_s: float = 2.0,
        json_mode: bool = False,
    ) -> str:
        """
        Generate code using Claude model via Vertex AI.
        
        Args:
            model: Claude model name (e.g., "claude-sonnet-4-5@20250929")
            temperature: Sampling temperature (0.0 = deterministic)
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate
            max_retries: Number of retry attempts
            initial_backoff_s: Initial backoff time in seconds
            json_mode: Whether to request JSON output (not directly supported, uses prompt)
            
        Returns:
            Generated text content
        """
        messages = messages or []
        
        # Extract system message if present
        system_content = None
        api_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_content = content
            else:
                api_messages.append({"role": role, "content": content})
        
        # Ensure we have at least one message
        if not api_messages:
            api_messages = [{"role": "user", "content": "Hello"}]
        
        attempt = 0
        backoff = initial_backoff_s
        
        while True:
            try:
                kwargs = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": api_messages,
                }
                
                if temperature > 0:
                    kwargs["temperature"] = temperature
                
                if system_content:
                    kwargs["system"] = system_content
                
                response = self._client.messages.create(**kwargs)
                
                # Extract text from response
                if response.content and len(response.content) > 0:
                    return response.content[0].text.strip()
                else:
                    raise ValueError("Empty response from Claude")
                    
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
                    console.print(f"[red]Claude request failed[/]: {e}")
                    raise
                
                # Jittered exponential backoff
                sleep_s = backoff * (1.0 + 0.25 * random())
                console.print(f"[yellow]Retry {attempt}/{max_retries}[/] after {sleep_s:.1f}s: {e}")
                time.sleep(sleep_s)
                backoff *= 2
    
    def list_models(self) -> List[str]:
        """List available Claude models on Vertex AI."""
        return [
            "claude-sonnet-4-5@20250929",
            "claude-3-5-sonnet-v2@20241022",
            "claude-3-5-haiku@20241022",
            "claude-3-opus@20240229",
        ]


# Convenience function
def create_claude_client(
    project_id: Optional[str] = None,
    region: str = "global",
) -> ClaudeClient:
    """
    Create a Claude client with the specified configuration.
    
    Args:
        project_id: GCP project ID (or set GOOGLE_CLOUD_PROJECT env var)
        region: Vertex AI region (default: global for Claude)
        
    Returns:
        Configured ClaudeClient instance
    """
    return ClaudeClient(project_id=project_id, region=region)

