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
class GeminiClient:
    """
    Google Gemini client using Vertex AI with service account authentication.
    
    Authentication options (in order of priority):
    1. Pass credentials_path directly
    2. Set GOOGLE_APPLICATION_CREDENTIALS environment variable
    3. Use application default credentials (ADC)
    
    Required environment variables:
    - GOOGLE_CLOUD_PROJECT: Your GCP project ID
    - GOOGLE_CLOUD_LOCATION: Region (default: us-central1)
    """
    
    project_id: Optional[str] = None
    location: str = "us-central1"
    credentials_path: Optional[str] = None
    _client: Any = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        # Get project ID from environment if not provided
        if not self.project_id:
            self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        
        # Get location from environment if available
        env_location = os.getenv("GOOGLE_CLOUD_LOCATION")
        if env_location:
            self.location = env_location
        
        # Set credentials path if provided
        if self.credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
        elif not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            # Check for common credential file locations
            default_paths = [
                "service-account.json",
                "credentials.json",
                os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
            ]
            for path in default_paths:
                if os.path.exists(path):
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
                    console.print(f"[dim]Using credentials from: {path}[/]")
                    break
        
        if not self.project_id:
            console.print("[yellow]WARNING[/]: GOOGLE_CLOUD_PROJECT not set; requests may fail.")
        
        self._init_client()
    
    def _init_client(self) -> None:
        """Initialize the Vertex AI client."""
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            
            vertexai.init(project=self.project_id, location=self.location)
            self._client = GenerativeModel
            console.print(f"[green]✓[/] Gemini client initialized (project: {self.project_id}, location: {self.location})")
        except ImportError:
            console.print("[red]ERROR[/]: google-cloud-aiplatform not installed. Run: pip install google-cloud-aiplatform")
            raise
        except Exception as e:
            console.print(f"[red]ERROR[/]: Failed to initialize Vertex AI: {e}")
            raise
    
    def generate_code(
        self,
        model: str = "gemini-2.5-flash-preview-05-20",
        temperature: float = 0.0,
        messages: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = 2000,
        max_retries: int = 5,
        initial_backoff_s: float = 1.0,
        json_mode: bool = False,
    ) -> str:
        """
        Generate code using Gemini model.
        
        Args:
            model: Gemini model name (e.g., "gemini-2.5-flash-preview-05-20", "gemini-2.5-pro-preview-05-06")
            temperature: Sampling temperature (0.0 = deterministic)
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate
            max_retries: Number of retry attempts
            initial_backoff_s: Initial backoff time in seconds
            json_mode: Whether to request JSON output
            
        Returns:
            Generated text content
        """
        from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part
        
        # Convert OpenAI-style messages to Gemini format
        gemini_contents = self._convert_messages(messages or [])
        
        # Build generation config
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        if json_mode:
            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            )
        
        # Initialize model
        gemini_model = self._client(model)
        
        attempt = 0
        backoff = initial_backoff_s
        
        while True:
            try:
                response = gemini_model.generate_content(
                    gemini_contents,
                    generation_config=generation_config,
                )
                
                # Extract text from response
                if response.candidates and response.candidates[0].content.parts:
                    return response.candidates[0].content.parts[0].text.strip()
                else:
                    raise ValueError("Empty response from Gemini")
                    
            except Exception as e:
                attempt += 1
                error_str = str(e).lower()
                
                # Check if it's a retryable error
                is_retryable = any(x in error_str for x in [
                    "429", "quota", "rate", "resource exhausted",
                    "500", "502", "503", "504", "unavailable", "timeout"
                ])
                
                if attempt > max_retries or not is_retryable:
                    console.print(f"[red]Gemini request failed[/]: {e}")
                    raise
                
                # Jittered exponential backoff
                sleep_s = backoff * (1.0 + 0.25 * random())
                console.print(f"[yellow]Retry {attempt}/{max_retries}[/] after {sleep_s:.1f}s: {e}")
                time.sleep(sleep_s)
                backoff *= 2
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> List[Any]:
        """Convert OpenAI-style messages to Gemini Content format."""
        from vertexai.generative_models import Content, Part
        
        contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                # Gemini handles system prompts differently - prepend to first user message
                system_instruction = content
            elif role == "user":
                if system_instruction:
                    content = f"{system_instruction}\n\n{content}"
                    system_instruction = None
                contents.append(Content(role="user", parts=[Part.from_text(content)]))
            elif role == "assistant":
                contents.append(Content(role="model", parts=[Part.from_text(content)]))
        
        return contents
    
    def list_models(self) -> List[str]:
        """List available Gemini models."""
        # Common Gemini models available on Vertex AI
        return [
            "gemini-2.0-flash-001",
            "gemini-2.0-flash-lite-001",
            "gemini-1.5-flash-002",
            "gemini-1.5-pro-002",
            "gemini-2.5-flash-preview-05-20",
            "gemini-2.5-pro-preview-05-06",
        ]


# Convenience function to create a client
def create_gemini_client(
    project_id: Optional[str] = None,
    location: str = "us-central1",
    credentials_path: Optional[str] = None,
) -> GeminiClient:
    """
    Create a Gemini client with the specified configuration.
    
    Args:
        project_id: GCP project ID (or set GOOGLE_CLOUD_PROJECT env var)
        location: GCP region (default: us-central1)
        credentials_path: Path to service account JSON file
        
    Returns:
        Configured GeminiClient instance
    """
    return GeminiClient(
        project_id=project_id,
        location=location,
        credentials_path=credentials_path,
    )

