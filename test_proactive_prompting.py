#!/usr/bin/env python3
# Author: Bradley R. Kinnard
"""
Quick test to verify proactive suggestion prompting works in orchestrate().
Tests the 'build' mode with a multi-step planning query.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_proactive_prompting():
    """Test that the orchestrate function generates proactive suggestions."""
    from backend.main import orchestrate

    print("=" * 80)
    print("🧪 Testing Proactive Suggestion Prompting")
    print("=" * 80)

    # Test case: Multi-agent setup planning (should trigger proactive suggestions)
    query = "Plan a multi-agent setup for distributed RAG with 3 worker nodes"
    mode = "build"

    print(f"\n📝 Query: {query}")
    print(f"🎯 Mode: {mode}")
    print(f"\n⏳ Running orchestrate()...\n")

    try:
        result = orchestrate(query, mode)

        print("✅ Orchestration complete!")
        print("\n" + "=" * 80)
        print("📊 RESULTS")
        print("=" * 80)

        # Display response (truncated for readability)
        response = result.get("response", "")
        print(f"\n📄 Response ({len(response)} chars):")
        print("-" * 80)
        preview = response[:500] + "..." if len(response) > 500 else response
        print(preview)

        # Display fusion weights
        weights = result.get("fusion_weights", {})
        print(f"\n⚖️  Fusion Weights:")
        print(f"   RAG:   {weights.get('rag', 0):.3f}")
        print(f"   CAG:   {weights.get('cag', 0):.3f}")
        print(f"   Graph: {weights.get('graph', 0):.3f}")

        # Display reward
        reward = result.get("reward", 0)
        print(f"\n🎯 Reward: {reward:.3f}")

        # Display proactive suggestions (key test!)
        suggestions = result.get("proactive_suggestions", [])
        print(f"\n💡 Proactive Suggestions ({len(suggestions)}):")
        if suggestions:
            for i, sugg in enumerate(suggestions, 1):
                print(f"   {i}. {sugg}")
        else:
            print("   ⚠️  No suggestions generated!")

        # Validation
        print("\n" + "=" * 80)
        print("✅ VALIDATION")
        print("=" * 80)

        checks = [
            ("Response generated", len(response) > 0),
            ("Fusion weights present", bool(weights)),
            ("Reward computed", reward > 0),
            ("Suggestions generated", len(suggestions) > 0)
        ]

        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"{status} {check_name}")

        all_passed = all(passed for _, passed in checks)

        if all_passed:
            print("\n🎉 All checks passed! Proactive prompting is working.")
        else:
            print("\n⚠️  Some checks failed. Review output above.")

        return all_passed

    except Exception as e:
        print(f"\n❌ Error during orchestration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_proactive_prompting()
    sys.exit(0 if success else 1)
