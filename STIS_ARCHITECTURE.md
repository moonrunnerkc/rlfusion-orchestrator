# STIS Architecture: Sub-Token Intuition Swarms

Author: Bradley R. Kinnard
Version: 0.1.0
Status: Implemented (Phases 1-9)

## Overview

STIS (Sub-Token Intuition Swarms) is a secondary generation engine that activates
when the primary asymmetric pipeline encounters irreconcilable contradictions between
retrieval sources. Instead of forcing a language model to arbitrarily pick one
source over another, STIS achieves consensus through mathematical convergence
in continuous latent space, before any discrete token is sampled.

## Problem Statement

When CAG (cached answers) and GraphRAG (knowledge graph traversal) return conflicting
factual claims, the GPU executor (Llama 3.1 8B) has no principled way to resolve
the conflict. It typically picks whichever claim appears more prominent in the fused
context, which is an arbitrary and unreliable heuristic. STIS replaces this with
mathematically forced agreement.

## Architecture

```
                           +------------------+
                           |  User Query      |
                           +--------+---------+
                                    |
                           +--------v---------+
                           |  CAG Lookup      |
                           | (< 5ms on hit)   |
                           +--------+---------+
                              hit /    \ miss
                             /          \
                    [return]    +--------v---------+
                                |  Safety Gate     |
                                | (CPU triage)     |
                                +--------+---------+
                                         |
                         +---------------v---------------+
                         |       GraphRAG Retrieval       |
                         |  Entity resolution + 2-hop    |
                         +---------------+---------------+
                                         |
                                +--------v---------+
                                |  RL Fusion        |
                                | [cag, graph]      |
                                +--------+---------+
                                         |
                          +--------------v--------------+
                          |   Contradiction Detector     |
                          |   (detect_contradiction)     |
                          |                              |
                          |  Checks:                     |
                          |  1. CAG/Graph cosine sim     |
                          |  2. CSWR confidence score    |
                          +---------+----------+---------+
                                    |          |
                          sim >= 0.40    sim < 0.40
                          OR cswr >= 0.70   AND cswr < 0.70
                                    |          |
                         +----------v--+    +--v-----------+
                         | GPU Executor|    |  STIS Engine  |
                         | (Llama 8B) |    |  (fallback)   |
                         +----------+--+    +--+-----------+
                                    |          |
                                    +----+-----+
                                         |
                                +--------v---------+
                                |  Critique Agent   |
                                | (GPU Executor)    |
                                +--------+---------+
                                         |
                                +--------v---------+
                                |  Response         |
                                +------------------+
```

## Dual-Condition Gate

STIS routing requires BOTH conditions to be true simultaneously:

1. **Contradiction detected**: Cosine similarity between the top CAG result and
   top Graph result falls below `STIS_SIMILARITY_FLOOR` (0.40). This indicates
   the two sources are discussing fundamentally different or opposing topics.

2. **Low CSWR confidence**: The best CSWR (Contextual Stability-Weighted Ranking)
   score across all retrieval results falls below `STIS_CSWR_THRESHOLD` (0.70).
   This confirms the retrieval pipeline lacks confidence in its results.

If only one condition is met, the GPU executor proceeds normally:
- Contradiction without low CSWR: likely a topic mismatch, not a real conflict.
  The retrieval pipeline is confident enough to handle it.
- Low CSWR without contradiction: weak retrieval, but sources agree on what
  little they found. The GPU executor can work with that.

## STIS Swarm Engine

### Core Algorithm

The swarm engine operates on transformer hidden states (the continuous vectors
before the lm_head projection). At each token generation step:

1. **Extract**: Run N agents (default 2) through the model. Each agent maintains
   its own input sequence. Extract the final-layer hidden state at the last
   token position from each agent.

2. **Measure**: Compute pairwise cosine similarity across all N agent hidden
   state vectors. Calculate mean similarity and maximum deviation.

3. **Converge**: If mean similarity < threshold (default 0.92), blend each
   agent's state toward the centroid:
   `state_new = (1 - alpha) * state + alpha * centroid`
   Repeat until convergence or max iterations (default 30).

4. **Sample**: Once converged, project the unified centroid through lm_head
   to get logits. Apply temperature + nucleus (top-p) sampling to select
   the next token.

5. **Append**: Add the chosen token to all agent sequences. Repeat from step 1.

### Mathematical Invariants

Three invariants are enforced and tested:

- **Dimension Stability**: Hidden state vectors never change dimensionality
  through the convergence loop. Output is always (hidden_dim,).

- **Centroid Preservation**: The blended centroid remains within the convex
  hull of the original agent states. No dimension of the centroid exceeds
  the per-dimension min/max across agents.

- **Convergence Monotonicity**: Mean pairwise similarity is non-decreasing
  across iterations. The alpha blend can only pull agents closer together,
  never push them apart.

### Model

Qwen2.5-1.5B loaded in float16 for a 3GB VRAM footprint. The model runs
as a standalone FastAPI microservice on port 8100, independent of the main
asymmetric dual-model pipeline. Lazy-loaded on first `/generate` request
and auto-unloaded after 120s of inactivity to free VRAM.

## Components

### stis_engine/ (Microservice)

| File | Role |
|------|------|
| `config.py` | Frozen dataclass config with env var overrides |
| `swarm.py` | Core SwarmEngine: convergence loop, hidden state extraction, token sampling |
| `model_loader.py` | Qwen2.5-1.5B loading with GPU/CPU fallback and module-level caching |
| `schemas.py` | Pydantic request/response models for the FastAPI endpoints |
| `server.py` | FastAPI app with /generate and /health endpoints |
| `__main__.py` | CLI entrypoint for `python -m stis_engine` |

### backend/core/ (Integration)

| File | Symbol | Role |
|------|--------|------|
| `critique.py` | `detect_contradiction()` | Compares top CAG vs Graph results via cosine similarity |
| `critique.py` | `should_route_to_stis()` | Dual-condition gate: contradiction AND low CSWR |
| `critique.py` | `STIS_CSWR_THRESHOLD` | 0.70, hard threshold for CSWR confidence |
| `critique.py` | `STIS_SIMILARITY_FLOOR` | 0.40, cosine similarity floor for contradiction |
| `stis_client.py` | `format_axiom_prompt()` | Formats CAG/Graph claims as opposing axioms |
| `stis_client.py` | `request_stis_consensus()` | httpx POST to /generate with 45s timeout |
| `stis_client.py` | `log_stis_resolution()` | SQLite audit trail in stis_resolutions table |
| `stis_client.py` | `check_stis_health()` | Non-blocking health probe |

### backend/agents/ (Orchestration)

| File | Symbol | Role |
|------|--------|------|
| `orchestrator.py` | `step_stis_check()` | Evaluates STIS routing for step-by-step /ws pipeline |
| `orchestrator.py` | `run()` | STIS routing for /chat endpoint (non-streaming) |
| `base.py` | `PreparedContext` | Extended with `retrieval_results` for STIS routing |

### backend/main.py (Wiring)

The /ws WebSocket pipeline checks for contradictions after fusion (Step 3.5)
and before LLM generation (Step 4). If STIS resolves the contradiction, the
GPU executor streaming loop is skipped entirely. The STIS response is sent as a
single chunk. If STIS fails (timeout, unreachable, HTTP error), the pipeline
falls back to the GPU executor with a logged warning.

Prometheus counter `rlfusion_stis_routing_total` tracks events by outcome:
`resolved`, `failed`, or `skipped`.

## Configuration

In `backend/config.yaml`:

```yaml
stis:
  enabled: true
  host: http://localhost:8100
  timeout_secs: 45
  max_new_tokens: 128
  num_agents: null          # override engine default (2)
  similarity_threshold: null # override engine default (0.92)
  alpha: null               # override engine default (0.5)
```

STIS engine itself is configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `STIS_NUM_AGENTS` | 2 | Number of swarm agents |
| `STIS_SIM_THRESHOLD` | 0.92 | Convergence similarity threshold |
| `STIS_ALPHA` | 0.5 | Centroid blending rate |
| `STIS_PORT` | 8100 | Server port |
| `STIS_SEED` | 42 | Random seed for reproducibility |
| `STIS_MODEL` | Qwen/Qwen2.5-1.5B | HuggingFace model ID |
| `STIS_TIMEOUT` | 45 | Server request timeout (seconds) |
| `STIS_IDLE_TIMEOUT` | 120 | Seconds before auto-unloading model from GPU |

## SQLite Audit Trail

Every STIS routing event (successful or failed) is logged to
`db/rlfo_cache.db` in the `stis_resolutions` table:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-increment primary key |
| `timestamp` | DATETIME | Event time |
| `query` | TEXT | Original user query |
| `rag_claim` | TEXT | Top RAG chunk text (truncated to 500 chars) |
| `graph_claim` | TEXT | Top Graph chunk text (truncated to 500 chars) |
| `contradiction_similarity` | REAL | Cosine similarity between claims |
| `best_cswr` | REAL | Best CSWR score across all results |
| `resolved` | INTEGER | 1 if STIS succeeded, 0 if failed |
| `resolution_text` | TEXT | STIS consensus text (null if failed) |
| `total_tokens` | INTEGER | Tokens generated by STIS |
| `final_similarity` | REAL | Final agent convergence similarity |
| `total_iterations` | INTEGER | Total convergence iterations |
| `stis_wall_time` | REAL | STIS generation wall time (seconds) |
| `error` | TEXT | Error message (null if resolved) |
| `latency_secs` | REAL | Total client-side latency |

## Test Coverage

| Test File | Tests | Scope |
|-----------|-------|-------|
| `tests/test_stis_engine.py` | 29 | Convergence invariants, sampling, schemas, config |
| `tests/test_stis_contradiction.py` | 26 | Contradiction detection, routing logic, BGE embeddings |
| `tests/test_stis_integration.py` | 29 | Client fallbacks, SQLite logging, orchestrator wiring |

## Running

Start the STIS engine (requires GPU with 3GB+ VRAM):

```bash
python -m stis_engine
# or
uvicorn stis_engine.server:app --host 0.0.0.0 --port 8100
```

The main RLFO backend will automatically route contradictions to the STIS
engine when `stis.enabled: true` in config.yaml. If the STIS engine is
unreachable, the backend falls back to the GPU executor silently.

## Backward Compatibility

- No existing function signatures changed
- No existing response fields removed or renamed
- All 623 tests pass (168 core + 49 asymmetric + 66 agent + 340 others)
- `PreparedContext` gained one new field (`retrieval_results`) with a safe default
- New config key `stis` added with safe defaults (no manual config required)
- STIS is opt-in: enabled by default but degrades gracefully when the engine is unreachable
