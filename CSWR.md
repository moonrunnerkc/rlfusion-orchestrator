# Context Stability Weighted Retrieval (CSWR) -- v2.0

Author: Bradley R. Kinnard

CSWR is a multi-axis scoring filter that evaluates every chunk **before** it enters
the generation context. It fixes the two main reasons retrieval systems hallucinate:
fractured context at chunk boundaries, and high vector similarity not actually meaning
the chunk is useful for answering the question.

In the current 2-path architecture (CAG + GraphRAG), CSWR scoring is applied to
GraphRAG traversal results. FAISS vector search has been removed from the hot path.

## Scoring Formula

```
CSW = 0.35 x vector_score + 0.25 x local_stability + 0.20 x question_fit
    + 0.10 x drift_penalty + 0.10 x project_coherence
```

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| **Vector Score** | 0.35 | `1 / (1 + L2_distance)`, semantic similarity |
| **Local Stability** | 0.25 | Cosine similarity to adjacent chunks. Boundary penalty: 0.15 |
| **Question Fit** | 0.20 | Entity coverage (40%), fact coverage (30%), shape match (20+10%) |
| **Drift Penalty** | 0.10 | Penalizes chunks where neighbor similarity < 0.5. Isolated: 1.5x |
| **Project Coherence** | 0.10 | Cross-reference against known project centroids |

## Domain-Adaptive Thresholds

Stability requirements adjust per domain (detected via keywords):

| Domain | Stability Threshold |
|--------|-------------------|
| General | 0.70 |
| Tech (`gpu`, `model`, `embedding`, `transformer`) | 0.65 |
| Code (`function`, `class`, `def`, `import`) | 0.55 |

Recalibrate from logged episodes: `compute_domain_quantiles()` recalculates the
25th/50th/75th percentile stability scores per domain.

## Context Packing

After scoring and filtering, chunks are packed into coherent neighborhoods:
- Budget: 1,800 tokens per pack (configurable)
- Up to 4 packs per query
- Main content threshold: stability >= 0.65
- Supporting content threshold: CSW >= 0.40
- Answerability gate: packs below 0.55 are dropped (fallback to highest-scoring)

## Pipeline

1. Query decomposition (intent, entities, answer shape)
2. GraphRAG retrieval (entity resolution, multi-hop traversal)
3. CSWR multi-axis scoring on retrieved chunks
4. Stability context packing (coherent neighborhoods)
5. Answerability filtering (drop low-confidence packs)
6. Fused context assembly for generation

## Results

- Near-zero hallucinations on edge chunks
- 40-60% noise reduction vs standard top-k retrieval
- Fully local, private, runs on consumer GPUs
- All scoring weights configurable in `backend/config.yaml` under `cswr:`

## Implementation

- `backend/core/retrievers.py`: `score_chunks()`, `compute_stability()`,
  `compute_fit()`, `compute_drift()`, `build_pack()`
- `backend/config.yaml`: `cswr:` and `cswr_quantiles:` sections
- Tests: 168 unit tests in `tests/test_core_units.py`
- Full technical details: [WHITEPAPER.md](WHITEPAPER.md) Section 3
