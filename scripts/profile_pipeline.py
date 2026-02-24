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
    print(f"  {label}: {elapsed*1000:.1f} ms")
    return result, elapsed


def main():
    print("=" * 60)
    print("RLFUSION PIPELINE LATENCY PROFILER")
    print("=" * 60)

    query = "What GPU is recommended for running large language models?"

    # Stage 0: Module imports (cold start cost)
    t0 = time.perf_counter()
    from backend.config import cfg
    from backend.core.utils import embed_text, embed_batch
    from backend.core.decomposer import decompose_query, _heuristic_decompose
    from backend.core.retrievers import (
        get_rag_index, retrieve, retrieve_rag_structured,
        retrieve_cag, retrieve_graph, score_chunks, check_answerable,
    )
    from backend.core.memory import expand_query_with_context
    from backend.core.critique import critique, check_safety, strip_critique_block
    from backend.core.fusion import fuse_context
    from backend.agents.orchestrator import classify_complexity
    import_time = time.perf_counter() - t0
    print(f"\n[COLD START] Module imports: {import_time*1000:.1f} ms")

    # Stage 0b: Embedding model warm-up (first call loads model)
    t0 = time.perf_counter()
    _ = embed_text("warm-up")
    warmup_time = time.perf_counter() - t0
    print(f"[COLD START] Embedding model warm-up: {warmup_time*1000:.1f} ms")

    # Stage 0c: FAISS index load
    idx, idx_time = timed("[COLD START] FAISS index load", get_rag_index)
    print(f"  Index size: {idx.ntotal} vectors")

    print(f"\n{'='*60}")
    print("PER-QUERY STAGE PROFILING")
    print(f"Query: '{query}'")
    print(f"{'='*60}")

    total_t0 = time.perf_counter()

    # 1. Complexity classification (no LLM)
    _, t_classify = timed("1. Classify complexity", classify_complexity, query)

    # 2. Memory expansion (no LLM, in-memory lookup)
    (exp_q, meta), t_memory = timed(
        "2. Memory expansion", expand_query_with_context, "profiler", query
    )

    # 3a. Decomposition via heuristic (no LLM)
    _, t_decompose_h = timed("3a. Heuristic decompose", _heuristic_decompose, query)

    # 3b. Decomposition via LLM (expensive!)
    try:
        _, t_decompose_llm = timed("3b. LLM decompose", decompose_query, query)
    except Exception as e:
        print(f"  3b. LLM decompose: FAILED ({e})")
        t_decompose_llm = 0

    # 4. Embedding (query)
    _, t_embed = timed("4. Query embedding", embed_text, query)

    # 5. FAISS search (the vector lookup itself)
    import numpy as np
    vec = embed_text(query).reshape(1, 384)
    t0 = time.perf_counter()
    dists, idxs = idx.search(vec, 20)
    t_faiss = time.perf_counter() - t0
    print(f"  5. FAISS search (top-20): {t_faiss*1000:.1f} ms")

    # 6. CSWR scoring (includes batch embedding)
    import json
    from pathlib import Path
    meta_path = Path("indexes/metadata.json")
    chunks = []
    if meta_path.exists():
        all_meta = json.loads(meta_path.read_text())
        for i in range(len(idxs[0])):
            idx_val = int(idxs[0][i])
            if 0 <= idx_val < len(all_meta):
                chunks.append({
                    "text": all_meta[idx_val]["text"],
                    "source": all_meta[idx_val]["source"],
                    "score": 1.0 / (1.0 + float(dists[0][i])),
                    "id": str(idx_val),
                    "local_stability": 0.0,
                    "question_fit": 0.0,
                    "drift_penalty": 0.0,
                    "csw_score": 0.0,
                })

    if chunks:
        profile = _heuristic_decompose(query)
        profile["query_text"] = query
        _, t_cswr = timed("6. CSWR scoring", score_chunks, chunks, profile, cfg.get("cswr", {}))
    else:
        t_cswr = 0
        print("  6. CSWR scoring: SKIPPED (no chunks)")

    # 7. Answerability check (LLM call!)
    if chunks:
        from backend.core.retrievers import build_pack
        pack = build_pack(chunks[0], chunks, 1800)
        try:
            _, t_answer = timed("7. Answerability check (LLM)", check_answerable, pack, profile)
        except Exception as e:
            print(f"  7. Answerability check: FAILED ({e})")
            t_answer = 0
    else:
        t_answer = 0

    # 8. CAG retrieval (DB + embedding)
    _, t_cag = timed("8. CAG retrieval", retrieve_cag, query)

    # 9. Graph retrieval
    _, t_graph = timed("9. Graph retrieval", retrieve_graph, query)

    # 10. Full retrieve() (all paths combined)
    _, t_retrieve_full = timed("10. Full retrieve() [all paths]", retrieve, query)

    # 11. Safety check (LLM call!)
    try:
        _, t_safety = timed("11. Safety check (LLM)", check_safety, query)
    except Exception as e:
        print(f"  11. Safety check: FAILED ({e})")
        t_safety = 0

    # 12. RL fusion weights (embedding + policy inference)
    from backend.agents.fusion_agent import compute_rl_weights
    _, t_rl = timed("12. RL fusion weights", compute_rl_weights, query, None)

    # 13. Critique (LLM call!)
    fake_response = "The NVIDIA RTX 4090 is recommended for large language models."
    fake_context = "GPU benchmarks show the RTX 4090 leads in LLM inference."
    try:
        _, t_critique = timed("13. Critique (LLM)", critique, query, fake_context, fake_response)
    except Exception as e:
        print(f"  13. Critique: FAILED ({e})")
        t_critique = 0

    # 14. LLM generation (the main response)
    try:
        from ollama import Client
        client = Client(host=cfg["llm"]["host"])
        t0 = time.perf_counter()
        resp = client.chat(
            model=cfg["llm"]["model"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Be concise."},
                {"role": "user", "content": query},
            ],
            options={"temperature": 0.1, "num_predict": 200, "num_ctx": 4096},
        )
        t_generate = time.perf_counter() - t0
        token_count = len(resp["message"]["content"].split())
        print(f"  14. LLM generation: {t_generate*1000:.1f} ms ({token_count} words)")
    except Exception as e:
        print(f"  14. LLM generation: FAILED ({e})")
        t_generate = 0

    total_elapsed = time.perf_counter() - total_t0

    # Summary
    print(f"\n{'='*60}")
    print("LATENCY BREAKDOWN SUMMARY")
    print(f"{'='*60}")

    # Count LLM calls in the critical path
    llm_calls = []
    if t_decompose_llm > 0:
        llm_calls.append(("Decomposition", t_decompose_llm))
    if t_answer > 0:
        llm_calls.append(("Answerability check", t_answer))
    if t_safety > 0:
        llm_calls.append(("Safety check", t_safety))
    if t_generate > 0:
        llm_calls.append(("Generation", t_generate))
    if t_critique > 0:
        llm_calls.append(("Critique", t_critique))

    total_llm = sum(t for _, t in llm_calls)
    print(f"\nLLM calls in critical path: {len(llm_calls)}")
    for name, t in llm_calls:
        print(f"  {name}: {t*1000:.1f} ms")
    print(f"  TOTAL LLM time: {total_llm*1000:.1f} ms ({total_llm/total_elapsed*100:.0f}% of total)")

    non_llm = total_elapsed - total_llm
    print(f"\nNon-LLM compute: {non_llm*1000:.1f} ms")
    print(f"  Embedding: ~{(t_embed + t_cswr)*1000:.1f} ms (embed + CSWR scoring)")
    print(f"  FAISS search: {t_faiss*1000:.1f} ms")
    print(f"  Memory/classify/misc: {(t_classify + t_memory)*1000:.1f} ms")
    print(f"\nTOTAL query time: {total_elapsed*1000:.1f} ms")

    print(f"\n{'='*60}")
    print("OPTIMIZATION TARGETS")
    print(f"{'='*60}")
    print(f"LLM calls are {total_llm/total_elapsed*100:.0f}% of latency.")
    print(f"The system makes {len(llm_calls)} serial LLM calls per query.")
    if t_decompose_llm > 0:
        print(f"  -> Decompose could use heuristic ({t_decompose_h*1000:.1f} ms) vs LLM ({t_decompose_llm*1000:.1f} ms)")
    print(f"  -> Safety, answerability, and critique each add ~{(t_safety)*1000:.0f} ms")


if __name__ == "__main__":
    main()
