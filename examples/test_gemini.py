#!/usr/bin/env python3
"""
Example script to test Vertex AI integrations: Gemini, Claude, Llama 4, and Mistral.

Setup:
1. Create a service account in GCP Console
2. Download the JSON key file
3. Set environment variables:
   
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"

Usage:
    python examples/test_gemini.py            # Test all
    python examples/test_gemini.py --gemini   # Test Gemini only
    python examples/test_gemini.py --claude   # Test Claude only
    python examples/test_gemini.py --llama    # Test Llama 4 only
    python examples/test_gemini.py --mistral  # Test Mistral Codestral only
"""

from src.lib import GeminiClient, LLMClient
from src.lib.claude import ClaudeClient
from src.lib.llama import LlamaClient
from src.lib.mistral import MistralClient


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
    print(response[:500] + "..." if len(response) > 500 else response)


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
    print(response[:500] + "..." if len(response) > 500 else response)


def test_llama_direct():
    """Test Llama 4 Vertex client directly."""
    print("=" * 60)
    print("Testing LlamaClient directly (Vertex AI)")
    print("=" * 60)
    
    client = LlamaClient()
    
    messages = [
        {"role": "user", "content": "Write a Python function to check if a string is a palindrome. Include a docstring."}
    ]
    
    # Test Scout (smaller, faster)
    print("\n--- Llama 4 Scout ---")
    response = client.generate_code(
        model="meta/llama-4-scout-17b-16e-instruct-maas",
        messages=messages,
        temperature=0.0,
    )
    print(response[:500] + "..." if len(response) > 500 else response)


def test_unified_client():
    """Test unified LLM client with automatic routing."""
    print("\n" + "=" * 60)
    print("Testing LLMClient (unified routing)")
    print("=" * 60)
    
    client = LLMClient()
    
    messages = [
        {"role": "user", "content": "Write a Python function to check if a number is prime. Include a docstring."}
    ]
    
    # Test Gemini
    print("\n1. Testing with google/gemini-2.0-flash-001 (routes to Gemini):")
    response = client.generate_code(
        model="google/gemini-2.0-flash-001",
        messages=messages,
        temperature=0.0,
    )
    print(f"Response length: {len(response)} chars")
    print(response[:300] + "..." if len(response) > 300 else response)


def test_unified_client_claude():
    """Test unified LLM client with Claude Vertex routing."""
    print("\n" + "=" * 60)
    print("Testing LLMClient with Claude Vertex")
    print("=" * 60)
    
    client = LLMClient()
    
    messages = [
        {"role": "user", "content": "Write a Python function to reverse a string. Include a docstring."}
    ]
    
    print("\nTesting with vertex/claude-sonnet-4-5 (routes to Claude Vertex):")
    response = client.generate_code(
        model="vertex/claude-sonnet-4-5",
        messages=messages,
        temperature=0.0,
    )
    print(f"Response length: {len(response)} chars")
    print(response[:300] + "..." if len(response) > 300 else response)


def test_unified_client_llama():
    """Test unified LLM client with Llama Vertex routing."""
    print("\n" + "=" * 60)
    print("Testing LLMClient with Llama 4 Vertex")
    print("=" * 60)
    
    client = LLMClient()
    
    messages = [
        {"role": "user", "content": "Write a Python function to merge two sorted lists. Include a docstring."}
    ]
    
    print("\nTesting with vertex/llama-4-scout (routes to Llama 4 Vertex):")
    response = client.generate_code(
        model="vertex/llama-4-scout",
        messages=messages,
        temperature=0.0,
    )
    print(f"Response length: {len(response)} chars")
    print(response[:300] + "..." if len(response) > 300 else response)


def test_mistral_direct():
    """Test Mistral Codestral client directly."""
    print("=" * 60)
    print("Testing MistralClient directly (Vertex AI)")
    print("=" * 60)
    
    client = MistralClient()
    
    # Test chat mode
    messages = [
        {"role": "user", "content": "Write a Python function to binary search. Include a docstring."}
    ]
    
    print("\n--- Codestral 2 Chat Mode ---")
    response = client.generate_code(
        model="codestral-2",
        messages=messages,
        temperature=0.0,
        max_tokens=400,
    )
    print(response[:400] + "..." if len(response) > 400 else response)
    
    # Test FIM mode
    print("\n--- Codestral 2 Fill-in-the-Middle ---")
    fim_response = client.fill_in_middle(
        prompt="def fibonacci(n: int) -> int:",
        suffix="return result",
        model="codestral-2",
        max_tokens=150,
    )
    print("PROMPT: def fibonacci(n: int) -> int:")
    print("SUFFIX: return result")
    print("GENERATED:", fim_response[:200])


def test_unified_client_mistral():
    """Test unified LLM client with Mistral Vertex routing."""
    print("\n" + "=" * 60)
    print("Testing LLMClient with Mistral Codestral Vertex")
    print("=" * 60)
    
    client = LLMClient()
    
    messages = [
        {"role": "user", "content": "Write a Python function to find duplicates in a list. Include a docstring."}
    ]
    
    print("\nTesting with vertex/codestral-2 (routes to Mistral Vertex):")
    response = client.generate_code(
        model="vertex/codestral-2",
        messages=messages,
        temperature=0.0,
    )
    print(f"Response length: {len(response)} chars")
    print(response[:300] + "..." if len(response) > 300 else response)


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
    
    print("\n" + "=" * 60)
    print("Available Llama 4 models (Vertex AI)")
    print("=" * 60)
    llama = LlamaClient()
    for model in llama.list_models():
        print(f"  - {model}")
    
    print("\n" + "=" * 60)
    print("Available Mistral models (Vertex AI)")
    print("=" * 60)
    mistral = MistralClient()
    for model in mistral.list_models():
        print(f"  - {model}")


if __name__ == "__main__":
    import sys
    
    print("Vertex AI Integration Test (Gemini + Claude + Llama 4 + Mistral)")
    print("Make sure you have set:")
    print("  - GOOGLE_CLOUD_PROJECT")
    print("  - GOOGLE_APPLICATION_CREDENTIALS")
    print()
    
    # Parse args
    test_claude = "--claude" in sys.argv
    test_gemini = "--gemini" in sys.argv
    test_llama = "--llama" in sys.argv
    test_mistral = "--mistral" in sys.argv
    test_all = not (test_claude or test_gemini or test_llama or test_mistral)
    
    try:
        test_list_models()
        
        if test_all or test_gemini:
            test_gemini_direct()
            test_unified_client()
        
        if test_all or test_claude:
            test_claude_direct()
            test_unified_client_claude()
        
        if test_all or test_llama:
            test_llama_direct()
            test_unified_client_llama()
        
        if test_all or test_mistral:
            test_mistral_direct()
            test_unified_client_mistral()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
