#!/usr/bin/env python3
"""
Example script to test Gemini and Claude Vertex AI integration.

Setup:
1. Create a service account in GCP Console
2. Download the JSON key file
3. Set environment variables:
   
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"

Or pass them directly to the client.

Usage:
    python examples/test_gemini.py
    python examples/test_gemini.py --claude  # Test Claude only
    python examples/test_gemini.py --gemini  # Test Gemini only
"""

from src.lib import GeminiClient, LLMClient
from src.lib.claude import ClaudeClient

def test_gemini_direct():
    """Test Gemini client directly."""
    print("=" * 60)
    print("Testing GeminiClient directly")
    print("=" * 60)
    
    client = GeminiClient()
    
    messages = [
        {"role": "user", "content": "Write a Python function to calculate fibonacci numbers. Include a docstring."}
    ]
    
    response = client.generate_code(
        model="gemini-2.0-flash-001",
        messages=messages,
        temperature=0.0,
    )
    
    print("\nResponse:")
    print(response)


def test_claude_direct():
    """Test Claude Vertex client directly."""
    print("=" * 60)
    print("Testing ClaudeClient directly (Vertex AI)")
    print("=" * 60)
    
    client = ClaudeClient()
    
    messages = [
        {"role": "user", "content": "Write a Python function to calculate factorial. Include a docstring."}
    ]
    
    response = client.generate_code(
        model="claude-sonnet-4-5@20250929",
        messages=messages,
        temperature=0.0,
    )
    
    print("\nResponse:")
    print(response)


def test_unified_client():
    """Test unified LLM client with automatic routing."""
    print("\n" + "=" * 60)
    print("Testing LLMClient (unified routing)")
    print("=" * 60)
    
    client = LLMClient()
    
    messages = [
        {"role": "user", "content": "Write a Python function to check if a number is prime. Include a docstring."}
    ]
    
    # This will automatically route to Gemini
    print("\n1. Testing with google/gemini-2.0-flash-001 (routes to Gemini):")
    response = client.generate_code(
        model="google/gemini-2.0-flash-001",
        messages=messages,
        temperature=0.0,
    )
    print(f"Response length: {len(response)} chars")
    print(response[:500] + "..." if len(response) > 500 else response)


def test_unified_client_claude():
    """Test unified LLM client with Claude Vertex routing."""
    print("\n" + "=" * 60)
    print("Testing LLMClient with Claude Vertex")
    print("=" * 60)
    
    client = LLMClient()
    
    messages = [
        {"role": "user", "content": "Write a Python function to reverse a string. Include a docstring."}
    ]
    
    # This will route to Claude on Vertex AI
    print("\n1. Testing with vertex/claude-sonnet-4-5 (routes to Claude Vertex):")
    response = client.generate_code(
        model="vertex/claude-sonnet-4-5",
        messages=messages,
        temperature=0.0,
    )
    print(f"Response length: {len(response)} chars")
    print(response[:500] + "..." if len(response) > 500 else response)


def test_list_models():
    """List available models."""
    print("\n" + "=" * 60)
    print("Available Gemini models")
    print("=" * 60)
    
    client = GeminiClient()
    for model in client.list_models():
        print(f"  - {model}")
    
    print("\n" + "=" * 60)
    print("Available Claude models (Vertex AI)")
    print("=" * 60)
    
    claude = ClaudeClient()
    for model in claude.list_models():
        print(f"  - {model}")


if __name__ == "__main__":
    import sys
    
    print("Vertex AI Integration Test (Gemini + Claude)")
    print("Make sure you have set:")
    print("  - GOOGLE_CLOUD_PROJECT")
    print("  - GOOGLE_APPLICATION_CREDENTIALS")
    print()
    
    # Parse args
    test_claude = "--claude" in sys.argv
    test_gemini = "--gemini" in sys.argv
    test_all = not test_claude and not test_gemini
    
    try:
        test_list_models()
        
        if test_all or test_gemini:
            test_gemini_direct()
            test_unified_client()
        
        if test_all or test_claude:
            test_claude_direct()
            test_unified_client_claude()
        
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
