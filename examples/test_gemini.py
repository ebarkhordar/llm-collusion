#!/usr/bin/env python3
"""
Example script to test Gemini integration.

Setup:
1. Create a service account in GCP Console
2. Download the JSON key file
3. Set environment variables:
   
   export GOOGLE_CLOUD_PROJECT="your-project-id"
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"

Or pass them directly to the client.

Usage:
    python examples/test_gemini.py
"""

from src.lib import GeminiClient, LLMClient

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


def test_list_models():
    """List available Gemini models."""
    print("\n" + "=" * 60)
    print("Available Gemini models")
    print("=" * 60)
    
    client = GeminiClient()
    for model in client.list_models():
        print(f"  - {model}")


if __name__ == "__main__":
    import sys
    
    print("Gemini Integration Test")
    print("Make sure you have set:")
    print("  - GOOGLE_CLOUD_PROJECT")
    print("  - GOOGLE_APPLICATION_CREDENTIALS")
    print()
    
    try:
        test_list_models()
        test_gemini_direct()
        test_unified_client()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

