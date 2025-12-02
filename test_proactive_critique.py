#!/usr/bin/env python3
# Author: Bradley R. Kinnard
"""
Test script demonstrating the enhanced proactive critique system.
Shows how the reward system now incentivizes intelligent suggestions.
"""

from backend.core.critique import critique

def test_proactive_critique():
    """Test the enhanced critique system with proactive bonus."""

    print("=" * 80)
    print("🧪 Testing Enhanced Critique System with Proactivity Bonus")
    print("=" * 80)

    # Test case 1: Response without suggestions (baseline)
    print("\n📊 Test 1: Quality response WITHOUT proactive suggestions")
    print("-" * 80)

    query1 = "How do I optimize FAISS indexing for 10M vectors?"
    context1 = "FAISS documentation shows IVF indexing scales to billions..."
    response1 = """For 10M vectors, I recommend using IndexIVFPQ with these parameters:
- nlist=4096 (cluster count)
- M=16, nbits=8 (product quantization)
- GPU acceleration enabled

This gives ~256x compression with <3% recall loss."""

    try:
        result1 = critique(query1, context1, response1)
        print(f"Query: {query1}")
        print(f"Response length: {len(response1)} chars")
        print(f"\n✅ Critique Result:")
        print(f"  Base score: {result1['breakdown']['base_score']:.3f}")
        print(f"  Scaled base: {result1['breakdown']['scaled_base']:.3f}")
        print(f"  Proactivity bonus: +{result1['breakdown']['proactivity_bonus']:.3f}")
        print(f"  Final reward: {result1['breakdown']['final_reward']:.3f}")
        print(f"  Suggestions generated: {len(result1['proactive_suggestions'])}")
        if result1['proactive_suggestions']:
            print(f"\n  🎯 Proactive Suggestions:")
            for i, sugg in enumerate(result1['proactive_suggestions'], 1):
                print(f"     {i}. {sugg}")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test case 2: Same response but critique generates suggestions
    print("\n\n📊 Test 2: Same query - system should generate suggestions automatically")
    print("-" * 80)

    try:
        result2 = critique(query1, context1, response1)
        print(f"Query: {query1}")
        print(f"\n✅ Critique Result:")
        print(f"  Base score: {result2['breakdown']['base_score']:.3f}")
        print(f"  Scaled base: {result2['breakdown']['scaled_base']:.3f}")
        print(f"  Proactivity bonus: +{result2['breakdown']['proactivity_bonus']:.3f}")
        print(f"  Final reward: {result2['breakdown']['final_reward']:.3f}")
        print(f"\n  🎯 Proactive Suggestions ({len(result2['proactive_suggestions'])}):")
        for i, sugg in enumerate(result2['proactive_suggestions'], 1):
            print(f"     {i}. {sugg}")

        # Show the reward boost
        if result2['breakdown']['proactivity_bonus'] > 0:
            boost_pct = (result2['breakdown']['proactivity_bonus'] /
                        result2['breakdown']['scaled_base'] * 100)
            print(f"\n  📈 Proactivity Boost: +{boost_pct:.1f}% over base reward")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Test case 3: Different query to show variation
    print("\n\n📊 Test 3: Different query - security audit scenario")
    print("-" * 80)

    query3 = "Review my FastAPI endpoint for security issues"
    context3 = "Code shows async endpoint with DB writes, no input validation..."
    response3 = """Critical security issues found:

1. No input validation - add Pydantic models with validators
2. Missing rate limiting - attackers can spam endpoint
3. SQL injection risk - use parameterized queries
4. No authentication - add OAuth2 dependency

Fix these immediately before deploying to production."""

    try:
        result3 = critique(query3, context3, response3)
        print(f"Query: {query3}")
        print(f"\n✅ Critique Result:")
        print(f"  Base score: {result3['breakdown']['base_score']:.3f}")
        print(f"  Scaled base: {result3['breakdown']['scaled_base']:.3f}")
        print(f"  Proactivity bonus: +{result3['breakdown']['proactivity_bonus']:.3f}")
        print(f"  Final reward: {result3['breakdown']['final_reward']:.3f}")
        print(f"\n  🎯 Proactive Suggestions ({len(result3['proactive_suggestions'])}):")
        for i, sugg in enumerate(result3['proactive_suggestions'], 1):
            print(f"     {i}. {sugg}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print("\n" + "=" * 80)
    print("🎓 Key Takeaways:")
    print("=" * 80)
    print("""
1. Base Quality Score: LLM judge evaluates response quality (0.0-1.0)
2. Proactivity Generation: System automatically generates 2-3 smart suggestions
3. Proactivity Scoring: Frozen judge evaluates suggestion quality (prevents gaming)
4. Final Reward: base + (proactivity_bonus up to +0.15), capped at 1.0

This structure incentivizes the RL policy to:
- Maintain high response quality (85% of score)
- Generate intelligent proactive suggestions (up to 15% bonus)
- Think ahead about user needs (not just answer the immediate query)

The frozen judge prevents gaming because the policy can't optimize for both
the quality judge and proactivity judge simultaneously without genuinely
good responses AND suggestions.
""")

if __name__ == "__main__":
    test_proactive_critique()
