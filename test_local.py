#!/usr/bin/env python3
"""
Local test script for the Runpod handler
"""
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_handler():
    """Test the handler with sample input"""
    from handler import handler

    # Load test input
    with open('test_input.json') as f:
        event = json.load(f)

    print("Testing handler with input:")
    print(json.dumps(event, indent=2))
    print("\n" + "="*80 + "\n")

    # Call handler
    result = handler(event)

    print("Handler result:")
    print(json.dumps(result, indent=2))

    # Check for errors
    if "error" in result:
        print(f"\n❌ Error: {result['error']}")
        return False

    print(f"\n✅ Success!")
    print(f"   Computed: {result.get('total_computed', 0)} nonces")
    print(f"   Valid: {result.get('total_valid', 0)} nonces")

    return True

if __name__ == "__main__":
    try:
        success = test_handler()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
