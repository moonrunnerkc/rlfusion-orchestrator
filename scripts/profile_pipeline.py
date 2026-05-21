# Author: Bradley R. Kinnard
# Profile each pipeline stage to identify latency bottlenecks.
# Run from project root: python scripts/profile_pipeline.py

import os
import sys
import time

os.environ["RLFUSION_FORCE_CPU"] = "true"
os.environ["RLFUSION_DEVICE"] = "cpu"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def timed(label, fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"  {label}: {elapsed * 1000:.1f} ms")
    return result, elapsed


def main():
    print("=" * 60)
    print("RLFUSION PIPELINE LATENCY PROFILER")
    print("=" * 60)

    query = "What GPU is recommended for running large language models?"

    # Stage 0: Module imports (cold start cost)
    t0 = time.perf_counter()
    from backend.config import cfg
    from backend.core.utils import embed_text
    from backend.core.decomposer import _heuristic_decompose
    from backend.core.retrievers import (
        retrieve, retrieve_cag, retrieve_graph, score_chunks, build_pack,
    )
    from backend.core.memory import expand_query_with_context
    from backend.core.critique import critique, check_safety
    from backend.agents.orchestrator import classify_complexity
    import_time = time.perf_counter() - t0
    print(f"\n[COLD START] Module imports: {import_time * 1000:.1f} ms")

    # Stage 0b: Embedding model warm-up
    t0 = time.perf_counter()
    _ = embed_text("warm-up")
    warmup_time = time.perf_counter() - t0
    print(f"[COLD START] Embedding model warm-up: {warmup_time * 1000:.1f} ms")

    print(f"\n{'=' * 60}")
    print("PER-QUERY STAGE PROFILING")
    print(f"Query: '{query}'")
    print(f"{'=' * 60}")

    total_t0 = time.perf_counter()

    _, t_classify = timed("1. Classify complexity", classify_complexity, query)
    (exp_q, meta), t_memory = timed(
        "2. Memory expansion", expand_query_with_context, "profiler", query,
    )
    _, t_decompose_h = timed("3. Heuristic decompose", _heuristic_decompose, query)
    _, t_embed = timed("4. Query embedding", embed_text, query)
    _, t_cag = timed("5. CAG retrieval", retrieve_cag, query)
    _, t_graph = timed("6. Graph retrieval", retrieve_graph, query)
    retrieval, t_retrieve = timed("7. Full retrieve() [CAG + Graph]", retrieve, query)

    chunks = retrieval.get("graph", [])
    if chunks:
        profile = _heuristic_decompose(query)
        profile["query_text"] = query
        for c in chunks:
            c.setdefault("local_stability", 0.0)
            c.setdefault("question_fit", 0.0)
            c.setdefault("drift_penalty", 0.0)
            c.setdefault("csw_score", 0.0)
        scored, t_cswr = timed(
            "8. CSWR re-rank", score_chunks, chunks, profile, cfg.get("cswr", {}),
        )
        _, t_pack = timed(
            "9. Build pack (top chunk)", build_pack, scored[0], scored, 1800,
        )
    else:
        t_cswr = 0.0
        t_pack = 0.0
        print("  8. CSWR re-rank: SKIPPED (no chunks)")

    try:
        _, t_safety = timed("10. Safety check", check_safety, query)
    except Exception as e:
        print(f"  10. Safety check: FAILED ({e})")
        t_safety = 0.0

    from backend.agents.fusion_agent import compute_rl_weights
    _, t_rl = timed("11. RL fusion weights", compute_rl_weights, query, None)

    fake_response = "The NVIDIA RTX 4090 is recommended for large language models."
    fake_context = "GPU benchmarks show the RTX 4090 leads in LLM inference."
    try:
        _, t_critique = timed("12. Critique (LLM)", critique, query, fake_context, fake_response)
    except Exception as e:
        print(f"  12. Critique: FAILED ({e})")
        t_critique = 0.0

    try:
        from backend.core.model_router import get_engine
        engine = get_engine()
        t0 = time.perf_counter()
        resp = engine.generate(
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Be concise."},
                {"role": "user", "content": query},
            ],
            temperature=0.1, num_predict=200, num_ctx=4096,
        )
        t_generate = time.perf_counter() - t0
        token_count = len(resp.split())
        print(f"  13. LLM generation: {t_generate * 1000:.1f} ms ({token_count} words)")
    except Exception as e:
        print(f"  13. LLM generation: FAILED ({e})")
        t_generate = 0.0

    total_elapsed = time.perf_counter() - total_t0

    print(f"\n{'=' * 60}")
    print("LATENCY BREAKDOWN SUMMARY")
    print(f"{'=' * 60}")
    llm_calls = []
    if t_safety > 0:
        llm_calls.append(("Safety check", t_safety))
    if t_generate > 0:
        llm_calls.append(("Generation", t_generate))
    if t_critique > 0:
        llm_calls.append(("Critique", t_critique))
    total_llm = sum(t for _, t in llm_calls)
    print(f"\nLLM calls in critical path: {len(llm_calls)}")
    for name, t in llm_calls:
        print(f"  {name}: {t * 1000:.1f} ms")
    pct = total_llm / total_elapsed * 100 if total_elapsed else 0.0
    print(f"  TOTAL LLM time: {total_llm * 1000:.1f} ms ({pct:.0f}% of total)")

    non_llm = total_elapsed - total_llm
    print(f"\nNon-LLM compute: {non_llm * 1000:.1f} ms")
    print(f"  Embedding + CSWR: ~{(t_embed + t_cswr) * 1000:.1f} ms")
    print(f"  Memory/classify/misc: {(t_classify + t_memory) * 1000:.1f} ms")
    print(f"\nTOTAL query time: {total_elapsed * 1000:.1f} ms")


if __name__ == "__main__":
    main()
