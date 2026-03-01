<p align="center">
  <h1 align="center">RLFusion Orchestrator</h1>
  <p align="center">
    <strong>Local-first cognitive engine with asymmetric dual-model inference, offline RL routing, chunk stability filtering, and sub-token consensus generation.</strong>
  </p>
  <p align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"/></a>
    <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"/>
    <img src="https://img.shields.io/badge/tests-623%20passing-brightgreen" alt="623 tests"/>
    <img src="https://img.shields.io/badge/cuda-12.1%2B-76B900" alt="CUDA 12.1+"/>
    <img src="https://img.shields.io/badge/inference-llama--cpp--python-orange" alt="llama-cpp-python"/>
    <img src="https://img.shields.io/badge/platform-linux%20|%20docker-lightgrey" alt="Platforms"/>
    <img src="https://img.shields.io/badge/code%20style-black-000000" alt="Code Style: Black"/>
  </p>
</p>

**Author:** Bradley R. Kinnard
**LinkedIn:** [linkedin.com/in/brad-kinnard](https://www.linkedin.com/in/brad-kinnard/)
**Whitepaper:** [WHITEPAPER.md](WHITEPAPER.md) -- full technical write-up covering CSWR, CQL routing, self-critique rewards, and benchmark results.

If you use this work in your own project, research, or write-up, please cite this repo and the LinkedIn above.

<img width="1921" height="917" alt="ui-screenshot-rlfusion" src="https://github.com/user-attachments/assets/51043408-e3b9-4729-bef8-184e96073169" />

---

<details>
<summary><strong>Table of Contents</strong> (click to expand)</summary>

- [What This Is](#what-this-is)
- [Headline Features](#headline-features)
  - [CSWR -- Chunk Stability Weighted Retrieval](#-cswr--chunk-stability-weighted-retrieval)
  - [STIS -- Sub-Token Intuition Swarms](#-stis--sub-token-intuition-swarms)
- [Architecture](#architecture)
- [Asymmetric Dual-Model Pipeline](#asymmetric-dual-model-pipeline)
- [Two-Path Retrieval](#two-path-retrieval)
- [RL-Based Fusion Routing](#rl-based-fusion-routing)
- [Safety and Quality Layers](#safety-and-quality-layers)
- [Dynamic Tool System](#dynamic-tool-system)
- [Conversation Memory](#conversation-memory)
- [Quick Start](#quick-start)
- [Docker](#docker)
- [Frontend](#frontend)
- [RL Training](#rl-training)
- [Configuration](#configuration)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Test Suite](#test-suite)
- [Benchmarks](#benchmarks)
- [Hardware Compatibility](#hardware-compatibility)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)

</details>

---

## What This Is

RLFusion is a local cognitive engine that controls **how** it retrieves, **what** it trusts, and **when** to override its own sources, all running on consumer hardware with zero cloud dependencies.

The system uses an **asymmetric dual-model pipeline**: a fast 1.5B parameter model (Qwen 2.5, CPU-pinned) handles all triage, classification, and orchestration tasks, while a full 8B parameter model (Llama 3.1, GPU-pinned) handles generation, critique, and deep reasoning. Two retrieval paths (CAG semantic cache and GraphRAG entity traversal) are fused via an **offline RL policy** that learns the optimal weight mix per query type. A **stability filter** (CSWR) scores every chunk before it touches the LLM and kills noisy context at the gate. When retrieval sources contradict each other and confidence is low, a **secondary swarm engine** forces mathematical consensus in continuous latent space before any token is sampled.

Everything is transparent. You can watch fusion weights shift in real time from the UI. Nothing is hidden and nothing leaves your machine.

**Core pillars:**

| | |
|---|---|
| **Asymmetric Inference** | Qwen 2.5 1.5B (Q4_K_M, CPU) for triage + Llama 3.1 8B (Q8_0, GPU) for generation. Single process, zero inter-process latency. |
| **CSWR** | Chunk Stability Weighted Retrieval -- scores every chunk on local stability, question fit, and drift before it touches the LLM |
| **STIS** | Sub-Token Intuition Swarms -- resolves source contradictions via multi-agent convergence in hidden-state space |
| **CQL/PPO/DPO** | Three-stage adaptive RL policy that learns which retrieval path to trust for each query type |
| **CAG + GraphRAG** | Two-path retrieval: instant cache hits via SHA-256/semantic lookup, with entity graph traversal fallback |
| **Multi-Agent Pipeline** | LangGraph-orchestrated agents (safety -> retrieval -> fusion -> generation -> critique) with complexity-based routing |
| **623 Tests** | Full coverage across 12 test files, including asymmetric pipeline, core modules, agents, tools, RL, STIS, and benchmarks |

---

## Headline Features

### CSWR -- Chunk Stability Weighted Retrieval

> *Standard RAG pulls garbage. CSWR stops it at the gate.*

CSWR replaces naive top-k retrieval with a multi-axis scoring function that evaluates every chunk **before** it enters the generation context. This is the single biggest lever against hallucination in the system. In the current 2-path architecture, CSWR scoring is applied to GraphRAG traversal results.

**How it scores each chunk:**

```
CSW = 0.35 x vector_score + 0.25 x local_stability + 0.20 x question_fit
    + 0.10 x drift_penalty + 0.10 x project_coherence
```

| Axis | What It Measures | How |
|------|-----------------|-----|
| **Vector Score** | Semantic similarity | `1 / (1 + L2_distance)`, standard but insufficient alone |
| **Local Stability** | Is this chunk coherent with its document neighbors? | Cosine similarity to adjacent chunks. Boundary chunks receive a 0.15 penalty |
| **Question Fit** | Does this chunk actually address the query? | Entity coverage (40%), required fact coverage (30%), intent keyword matching (20%), answer shape match (10%) |
| **Drift Penalty** | Does this chunk sit in a topic-shifting neighborhood? | Penalizes chunks where average neighbor similarity drops below 0.5. Fully isolated chunks get 1.5x severity |
| **Project Coherence** | Does this chunk belong to the active project context? | Cross-references against known project centroids |

**Domain-adaptive thresholds** -- stability requirements adjust based on detected query domain:

| Domain | Detection | Stability Threshold |
|--------|-----------|-------------------|
| General | Default | 0.70 |
| Tech | Keywords: `gpu`, `model`, `embedding`, `transformer`, etc. | 0.65 |
| Code | Keywords: `function`, `class`, `def`, `import`, `return`, etc. | 0.55 |

**Context Packing** -- after scoring, chunks are packed into coherent neighborhoods (up to 1,800 tokens per pack) with an answerability gate. Packs scoring below 0.55 answerability are dropped.

These thresholds are recalibrated from logged episodes via `compute_domain_quantiles()`, which recalculates the 25th/50th/75th percentile stability scores per domain.

> **Implementation:** `backend/core/retrievers.py::score_chunks`, `build_pack`
> **Config:** `backend/config.yaml` under `cswr:` and `cswr_quantiles:`
> **Tests:** 168 unit tests in `tests/test_core_units.py` covering `TestScoreChunks`, `TestComputeStability`, `TestComputeFit`, `TestComputeDrift`, `TestBuildPack`

---

### STIS -- Sub-Token Intuition Swarms

> *When your sources disagree and confidence is low, don't guess, converge.*

STIS is a secondary generation engine that activates when CAG and Graph retrieval produce contradictory facts **and** CSWR confidence is simultaneously low. Instead of letting the LLM arbitrarily pick one source, STIS achieves consensus through mathematical convergence in continuous latent space before any discrete token is sampled.

**Dual-condition gate, both must be true:**

| Condition | Threshold | Why |
|-----------|-----------|-----|
| **Contradiction detected** | Cosine similarity between top CAG result and top Graph result < **0.40** | Sources are discussing fundamentally different or opposing content |
| **Low CSWR confidence** | Best CSWR score across all results < **0.70** | The retrieval pipeline lacks confidence in what it found |

If only one condition fires, the LLM handles it normally. A contradiction without low CSWR means the pipeline is confident enough. Low CSWR without contradiction means sources agree on limited content.

**How the swarm works (per token):**

```
1. EXTRACT  -> Run N agents through the model, extract final-layer hidden states
2. MEASURE  -> Compute pairwise cosine similarity across all agent hidden states
3. CONVERGE -> If mean similarity < 0.92, blend toward centroid:
               state_new = (1 - a) x state + a x centroid
               Repeat until convergence or max 30 iterations
4. SAMPLE   -> Project unified centroid through lm_head -> logits -> top-p sampling
5. APPEND   -> Add chosen token to all agent sequences, repeat from step 1
```

**Mathematical invariants enforced at every step:**

- **Dimension Stability** -- hidden state vectors never change dimensionality through convergence
- **Centroid Preservation** -- the blended centroid remains within the convex hull of agent states
- **Convergence Monotonicity** -- mean pairwise similarity is non-decreasing across iterations

**Operational design:**

| Property | Value |
|----------|-------|
| Model | Qwen2.5-1.5B (float16, ~3 GB VRAM) |
| Agents | 2 default (configurable up to 16) |
| Convergence threshold | 0.92 |
| Blending rate | 0.5 |
| Max iterations per token | 20 (runtime default via `load_config()`) |
| Server | FastAPI microservice on port 8100 |
| Loading | **Lazy** -- model stays off-GPU until first `/generate` request |
| Idle unload | Auto-unloads after 120s of inactivity to free VRAM for the main pipeline |
| Audit trail | Every routing event logged to SQLite (`stis_resolutions` table) |

When STIS handles a query, the LLM streaming loop is skipped entirely. The STIS response is sent as a single chunk. If STIS fails (timeout, unreachable, HTTP error), the pipeline falls back to normal LLM generation silently with a logged warning.

> **Full architecture doc:** [STIS_ARCHITECTURE.md](STIS_ARCHITECTURE.md)
> **Engine:** `stis_engine/swarm.py` -- core convergence loop and token sampling
> **Client:** `backend/core/stis_client.py` -- axiom formatting, httpx POST, SQLite audit
> **Gate:** `backend/core/critique.py::should_route_to_stis`, `detect_contradiction`
> **Config:** `backend/config.yaml` under `stis:`
> **Tests:** 84 tests across `test_stis_engine.py` (29), `test_stis_contradiction.py` (26), `test_stis_integration.py` (29)

---

## Architecture

The system is organized as a **multi-agent pipeline** orchestrated by LangGraph. Each agent follows a `plan() -> act() -> reflect()` protocol. The orchestrator classifies query complexity and routes through the appropriate agent chain. All LLM calls route through the asymmetric orchestrator, which dispatches triage tasks to the CPU worker and generation tasks to the GPU executor.

```
+-------------------------------------------------------------------------+
|                           USER QUERY                                     |
+-------------------------------+-----------------------------------------+
                                |
                    +-----------v-----------+
                    |     ORCHESTRATOR      |
                    |  classify_complexity  |
                    |  simple | complex |   |
                    |     adversarial       |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |    SAFETY AGENT       |
                    |  (CPU triage worker)  |
                    |  * Regex attack scan  |
                    |  * OOD detection      |
                    |  * LLM safety check   |
                    +-----+---------+-------+
                    blocked         safe
                      |               |
                      v               v
                  [BLOCKED]   +---------------+
                              |RETRIEVAL AGENT|
                              +--+--------+---+
                                 |        |
                    +------------+        +------------+
                    v                                  v
                +-------+                        +---------+
                |  CAG  |                        | GraphRAG|
                | Cache |                        | Entity  |
                | (< 5ms|                        | Traverse|
                | on hit)|                       +---------+
                +---+---+                             |
                    |                                 |
                    +------+--------+-----------------+
                           |
                +-----------v-----------+
                |    FUSION AGENT       |
                |  CQL/PPO/DPO policy   |
                |  2-path weighted merge|
                +-----------+-----------+
                            |
                +-----------v-----------+
                | CONTRADICTION CHECK   |------ yes --> STIS ENGINE
                | sim < 0.40 AND        |              (swarm consensus)
                | cswr < 0.70?          |                    |
                +-----------+-----------+                    |
                      no    |   <----------------------------+
                            |
                +-----------v-----------+
                |   LLM GENERATION      |
                |   (GPU executor,      |
                |    Llama 3.1 8B)      |
                +-----------+-----------+
                            |
                +-----------v-----------+
                |   CRITIQUE AGENT      |
                |   (GPU executor)      |
                |  * Factual accuracy   |
                |  * Helpfulness        |
                |  * Proactivity        |
                |  * Citation coverage  |
                +-----------+-----------+
                            |
                +-----------v-----------+
                |   RESPONSE + UI       |
                |  weights, reward,     |
                |  suggestions shown    |
                +-----------+-----------+
                            |
                            v
                +-----------------------+
                |  RL POLICY UPDATE     |
                |  (offline training)   |
                +-----------------------+
```

**Pipeline step-by-step:**

1. **Query intake** -- sanitize, length check (4,000 char max), assign correlation ID
2. **CAG lookup** -- SHA-256 exact match or semantic similarity check. On strong hit (score >= 0.85), return immediately, bypassing steps 3-10 entirely. This is the primary latency win: repeated queries cost < 5 ms.
3. **Safety agent** -- three-phase screening via CPU triage: regex attack patterns, Mahalanobis OOD detection, LLM safety classification
4. **Intent decomposition** -- CPU triage parses query into structured JSON (intent, entities, expected shape)
5. **GraphRAG traversal** -- entity resolution, Leiden community detection, 2-hop traversal for structured context
6. **RL observation build** -- CPU triage assembles the observation vector (394 dimensions)
7. **RL policy** -- CQL/PPO/DPO predicts 2D fusion weights [cag, graph]
8. **Fusion agent** -- merges CAG and Graph results weighted by RL output
9. **Contradiction check** -- if CAG vs Graph similarity < 0.40 *and* best CSWR < 0.70, routes to STIS
10. **LLM generates** -- GPU executor (Llama 3.1 8B) streams response grounded in fused context
11. **Critique agent** -- GPU executor scores factual accuracy, helpfulness, proactivity
12. **CAG store** -- high-reward responses cached for future instant retrieval
13. **Episode logged** -- query, weights, reward stored in SQLite replay buffer for RL training

---

## Asymmetric Dual-Model Pipeline

The system runs two GGUF models in a single Python process with zero inter-process communication:

| Model | File | Quant | Location | Role |
|---|---|---|---|---|
| Qwen 2.5 1.5B Instruct | `qwen2.5-1.5b-instruct-q4_k_m.gguf` | Q4_K_M | CPU (system RAM, ~2 GB) | Triage worker |
| Llama 3.1 8B Instruct | `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf` | Q8_0 | GPU (VRAM, ~9 GB) | Generation executor |

**Task routing rules:**

| Task Type | Worker | Rationale |
|---|---|---|
| `intent_parse` | CPU triage | Fast structured JSON, no deep reasoning needed |
| `safety_check` | CPU triage | Pattern + classification, low complexity |
| `cag_lookup` | CPU triage | Cache orchestration, not generation |
| `graph_trigger` | CPU triage | Entity extraction, structured output |
| `obs_build` | CPU triage | Observation vector assembly |
| `generation` | GPU executor | Full reasoning, long-form output |
| `critique` | GPU executor | Quality evaluation requires deep model |
| `stis_deep` | GPU executor | Contradiction resolution needs capacity |
| `faithfulness` | GPU executor | Claim verification against sources |

All triage calls that expect JSON use `json_repair` as a safety net. If the CPU worker produces malformed JSON, it is repaired automatically. If repair fails, one retry with a corrective prompt is attempted.

Both models loaded once at startup. Context windows set to 8,192 tokens each. GPU model pinned with `n_gpu_layers=-1` (all layers on GPU). CPU model pinned with `n_gpu_layers=0`. Thread-safe access via per-model locks.

**OOM fallback:** If the GPU executor hits an out-of-memory error, the system temporarily offloads it to CPU, logs the event, and recovers on the next successful load.

**VRAM/RAM monitoring:** A background monitor (via `pynvml`) logs memory snapshots every 10 seconds as structured JSON. Current readings available via the `health_check()` method on the orchestrator.

> **Implementation:** `backend/core/asymmetric_llm.py`, `backend/core/metrics.py`

---

## Two-Path Retrieval

The system uses two retrieval paths: CAG (Cached Answer Graph) for instant cache hits, and GraphRAG for entity-based knowledge traversal. FAISS vector search and Tavily web search have been removed from the hot path.

### CAG (Cached Answer Graph)

SQLite-backed cache with SHA-256 exact-match and semantic-similarity lookup. Queries are matched by:
1. SHA-256 hash of normalized query (strip, lowercase, hash)
2. Embedding cosine similarity (threshold >= 0.85)

High-reward responses (reward >= 0.70) are automatically cached after each interaction. On a strong cache hit, the response is returned in < 5 ms, bypassing the entire retrieval-fusion-generation pipeline.

### GraphRAG

NetworkX directed knowledge graph with:
- **Entity resolution** -- deduplication via embedding cosine similarity (threshold >= 0.92)
- **Leiden community detection** -- via igraph/leidenalg when available, connected-components fallback otherwise
- **Multi-hop traversal** -- up to 2 hops from matched entities, scores decay by 0.8 per hop
- **Qdrant in-memory** -- entity vector + payload search when qdrant-client is installed, numpy cosine fallback otherwise

The graph engine provides co-occurrence bonuses, coherence penalties, and path distance weights that feed into CSWR's question-fit scoring component.

> **Implementation:** `backend/core/retrievers.py`, `backend/core/graph_engine.py`

---

## RL-Based Fusion Routing

The system uses reinforcement learning to decide how much to trust each retrieval path for a given query. No static weights.

### Three-Stage Adaptive Policy

The `AdaptivePolicy` automatically transitions through RL stages as interaction data accumulates:

| Stage | Interactions | Method | Behavior |
|-------|-------------|--------|----------|
| **CQL** | 0-50 | Conservative Q-Learning (offline) | Stable defaults from pre-trained policy. No exploration. Ships with the repo. |
| **PPO** | 50-500 | Proximal Policy Optimization (online) | Weights start reflecting your usage patterns. Cautious exploration. |
| **DPO** | 500+ | Direct Preference Optimization | Preference-based refinement from high-reward vs low-reward episode pairs. |

### Observation Space (394 dimensions)

| Dims | Signal |
|------|--------|
| 384 | Query embedding (BGE-small-en-v1.5) |
| 1 | Binary cache hit indicator |
| 1 | CAG top result score |
| 1 | Graph connectivity signal (results / 10) |
| 3 | Graph top-3 result scores |
| 1 | Normalized query length (words / 50) |
| 3 | Query type vector (factual, relational, general) |

### Action Space (2 continuous values)

Raw logits for [cag, graph] passed through softmax and clamped to minimum 0.05 per source. Neither retrieval path is ever fully zeroed out.

### Heuristic Fallback

When the policy outputs near-uniform weights (max - min < 0.05), keyword heuristics take over:

| Query Pattern | CAG | Graph |
|---------------|-----|-------|
| "how does", "architecture", "design", "workflow", "system", "relationship" | 0.30 | 0.70 |
| "what is", "explain", "describe", "define" | 0.60 | 0.40 |
| Default | 0.50 | 0.50 |

### Reward Signal

Rewards come from the critique agent (running on the GPU executor), a dedicated LLM call that scores each response on factual accuracy, proactivity, and helpfulness (each 0.0-1.0). The mean becomes the scalar reward logged to the replay buffer. No human annotation required.

> **Implementation:** `backend/agents/fusion_agent.py`, `backend/rl/fusion_env.py`, `backend/rl/train_rl.py`, `backend/rl/train_ppo.py`, `backend/rl/train_dpo.py`

---

## Safety and Quality Layers

### Three-Phase Safety Gate

Every query passes through the safety agent (routed to the CPU triage worker) before retrieval begins:

1. **Regex pre-filter** -- fast pattern matching for prompt injection, SQL injection, template injection, XSS, jailbreak attempts
2. **OOD detection** -- Mahalanobis distance with Ledoit-Wolf covariance shrinkage, fitted on the embedding distribution of indexed documents. Queries exceeding the threshold (default: 50.0) are flagged
3. **LLM safety classification** -- binary SAFE/UNSAFE classification via the CPU triage model for patterns that escape regex

### Self-Critique

The critique agent runs a **dedicated, separate LLM call** on the GPU executor after generation (not inline). It returns structured JSON scores:

- **Factual accuracy** (0.0-1.0)
- **Proactivity** (0.0-1.0)
- **Helpfulness** (0.0-1.0)
- **Follow-up questions** (3 specific, non-generic suggestions)

Citation coverage is computed independently by counting `[1]`, `[2]`, etc. markers against substantive sentences.

### Tree-Structured Reasoning (ORPS)

For high-sensitivity queries (sensitivity > 0.7, as determined by query decomposition), the system runs **Outcome-Refining Process Supervision**:

1. Generate N candidates (default beam width: 3) with progressively higher temperature (0.2 -> 0.8)
2. Score each via critique
3. Prune candidates below the reward threshold (0.3)
4. Optionally refine top candidates
5. Select the best by composite reward

### Faithfulness Checking

Individual claims can be verified against source chunks. A TTL cache (default 300s) prevents redundant LLM calls for the same claim within the same context window. Used on the hot path only when sensitivity exceeds the gate threshold.

> **Implementation:** `backend/core/reasoning.py`, `backend/core/critique.py`, `backend/agents/safety_agent.py`

---

## Dynamic Tool System

The tool registry dispatches to specialized tools based on query content. Tools follow the `BaseTool` protocol and are rate-limited (default: 10 calls per tool per 60 seconds).

| Tool | What It Does | Safety |
|------|-------------|--------|
| **Calculator** | AST-based math evaluation (no `eval`), unit conversions (length, mass, data, time, temperature) | Safe operators whitelist, no code execution |
| **Code Executor** | Sandboxed Python via subprocess, stdout/stderr capture, 10s timeout | Banned modules list (os, sys, subprocess, socket, etc.), import blocker injected at runtime |
| **API Bridge** | Generic REST API calls with URL validation | Configurable timeout (15s default) |

> **Implementation:** `backend/tools/registry.py`, `backend/tools/calculator.py`, `backend/tools/code_executor.py`

---

## Conversation Memory

Per-session state tracked across turns with entity extraction and anaphora resolution:

- **Entity extraction** -- regex patterns identify business names, people, locations, and products from messages
- **Query expansion** -- follow-up queries with pronouns ("what are their hours?") are automatically expanded using tracked entities
- **Demonstrative filtering** -- "this project", "that algorithm" are recognized as self-contained and not expanded
- **Persistent user profile** -- facts stored via explicit commands or implicit statements. Stored in SQLite, injected into prompt context for personal queries

> **Implementation:** `backend/core/memory.py`, `backend/core/profile.py`

---

## Quick Start

### Requirements

- Python 3.10+
- NVIDIA GPU with 12+ GB VRAM (RTX 3080 or newer recommended, Blackwell/RTX 50-series supported)
- CUDA 12.1+ toolkit
- ~32 GB system RAM
- Node.js 18+ (frontend, optional)

### 1. Clone and install

```bash
git clone https://github.com/moonrunnerkc/rlfusion-orchestrator.git
cd rlfusion-orchestrator

python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

llama-cpp-python must be compiled with CUDA support. For Blackwell GPUs (RTX 5070/5080/5090):

```bash
CUDACXX=/usr/local/cuda/bin/nvcc \
  CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=100;120" \
  pip install llama-cpp-python --no-cache-dir
```

For Ampere/Ada GPUs (RTX 3090, 4090, etc.), adjust the architecture flags accordingly (e.g., `86;89`).

### 2. Download model artifacts

Both GGUF models are downloaded once and run fully offline after that:

```bash
# CPU triage model (~1.1 GB)
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models/

# GPU generation model (~8 GB)
huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
  Meta-Llama-3.1-8B-Instruct-Q8_0.gguf --local-dir models/
```

### 3. Initialize environment

```bash
cp .env.example .env
./scripts/init_db.sh
```

### 4. Add your documents

Drop `.txt`, `.md`, or `.pdf` files into `data/docs/`. Subdirectories are scanned recursively.

```bash
cp ~/my-notes/*.md  data/docs/
cp ~/papers/*.pdf   data/docs/
```

Documents feed the GraphRAG knowledge graph. The entity graph builds automatically from document content at indexing time.

### 5. Start the backend

```bash
uvicorn backend.main:app --port 8000
```

Both models load at startup (CPU worker in ~2s, GPU executor in ~5s). Interactive API docs at:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 6. Start the STIS engine (optional)

Only needed if you want contradiction resolution. Requires a GPU with 3+ GB VRAM.

```bash
python -m stis_engine
```

The engine runs on port 8100 and lazy-loads Qwen2.5-1.5B on first request. It auto-unloads after 120s of inactivity to free VRAM.

---

## Docker

Two profiles for different hardware configurations:

```bash
# CPU-only (both models run on CPU, slower but works without GPU)
docker compose --profile cpu up

# GPU (requires nvidia-container-toolkit, recommended)
docker compose --profile gpu up
```

The GPU profile uses NVIDIA CUDA 12.1 as base image and compiles llama-cpp-python with Blackwell architecture flags (`-DCMAKE_CUDA_ARCHITECTURES=100;120`). Multi-stage build keeps the production image lean by excluding training scripts, test files, and removed dependencies.

Models must be pre-downloaded to `models/` before building. Volumes are mounted for `data/`, `db/`, and `models/` so your documents, state, and model artifacts persist.

The container includes a health check that pings `http://localhost:8000/ping` every 30 seconds.

---

## Frontend

React + Vite + Tailwind CSS with real-time pipeline visualization.

```bash
cd frontend
npm install
npm run dev
```

Available at [http://localhost:5173](http://localhost:5173).

**What the UI shows:**

| Component | What It Does |
|-----------|-------------|
| **Agent Pipeline** | Live status of each agent (safety, retrieval, fusion, generation, critique) as the query processes |
| **Fusion Meter** | Real-time visualization of CAG/Graph weight distribution (2-bar meter) |
| **Monitoring Panel** | Weight history over time, reward tracking, system health indicators |
| **Chat Interface** | Multi-session chat with localStorage persistence, drag-and-drop file upload |
| **Settings Panel** | Runtime configuration toggles |
| **Connection Status** | WebSocket health indicator |

Shared TypeScript types live in `frontend/src/types/contracts.ts`, which mirrors backend response shapes exactly.

---

## RL Training

RLFusion ships with a pre-trained CQL policy (`models/rl_policy_cql.d3`, ~3.2 MB) that provides reasonable defaults out of the box. The system improves with use.

### Training commands

```bash
# Offline CQL (ships pre-trained, 2-path architecture)
python backend/rl/train_rl.py

# Retrain for 2-path architecture (500 simulated episodes)
python scripts/retrain_fusion.py

# Online PPO (after 50+ interactions)
python backend/rl/train_ppo.py

# DPO (after 500+ preference pairs)
python backend/rl/train_dpo.py
```

### Accelerate warm-up

```bash
python backend/rl/add_batch_episodes.py
python backend/rl/train_rl.py
```

### Additional training capabilities

| Feature | Description |
|---------|-------------|
| **LoRA SFT** | Fine-tune the base LLM on high-reward episodes via LoRA adapters, export to GGUF for local serving |
| **GRPO** | Group Relative Policy Optimization for multi-agent coordination |
| **Federated Learning** | Privacy-preserving policy updates across instances, weight deltas are L2-clipped and Gaussian-noised before sharing |
| **MoE Routing** | Register specialized models per task type (code, critique, generation) for mixture-of-experts dispatch |

> **Implementation:** `backend/rl/fine_tune.py`, `backend/rl/train_ppo.py::train_grpo`, `backend/rl/federated.py`, `backend/core/model_router.py`

---

## Configuration

All runtime behavior is controlled by `backend/config.yaml`. Every key has safe defaults. The system starts with zero manual configuration.

```yaml
llm:
  model: Llama-3.1-8B + Qwen-2.5-1.5B  # display name for dual-model pipeline
  host: http://localhost:11434
  temperature: 0.72
  max_tokens: 8192

inference:
  engine: llama_cpp_dual             # asymmetric dual-model pipeline
  cpu_model_path: models/qwen2.5-1.5b-instruct-q4_k_m.gguf
  gpu_model_path: models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf
  cpu_ctx_size: 8192                 # context window for CPU triage model
  gpu_ctx_size: 8192                 # context window for GPU generation model
  seed: 42                           # deterministic output
  max_concurrent: 4
  timeout_secs: 60

embedding:
  model: BAAI/bge-small-en-v1.5     # 384-dim embeddings
  device: cuda                       # or cpu

retrieval:
  paths:
    - cag                            # cached answer graph (first)
    - graph                          # GraphRAG entity traversal (fallback)

fusion:
  default_weights:
    cag: 0.4
    graph: 0.6

cswr:
  enabled: true
  top_k: 20
  pack_token_budget: 1800
  min_csw_score: 0.25
  answerability_threshold: 0.55
  stability_threshold: 0.7
  vector_weight: 0.35
  local_stability_weight: 0.25
  question_fit_weight: 0.20
  drift_penalty_weight: 0.10
  project_coherence_weight: 0.10

graph:
  enabled: true
  entity_similarity_threshold: 0.92
  max_hops: 2
  co_occurrence_bonus: 0.15
  coherence_penalty: 0.10
  path_distance_decay: 0.8

rl:
  policy_path: models/rl_policy_cql.d3
  adaptive_warmup:
    cql_until: 50
    ppo_until: 500
    dpo_after: 500

stis:
  enabled: true
  host: http://localhost:8100
  timeout_secs: 45
  max_new_tokens: 128

reasoning:
  beam_width: 3
  prune_threshold: 0.3
  faithfulness_on_hot_path: true
  faithfulness_sensitivity_gate: 0.7

web:
  enabled: false                     # web search removed from hot path

tools:
  enabled: true
  max_calls_per_tool: 10

multimodal:
  enabled: true
  clip_model: openai/clip-vit-base-patch32
  vision_model: llava

monitoring:
  prometheus_enabled: true
  correlation_id_header: X-Correlation-ID
```

---

## Environment Variables

Documented in `.env.example`:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `INFERENCE_ENGINE` | No | `llama_cpp_dual` | Inference backend. Use `llama_cpp_dual` for asymmetric pipeline. |
| `RLFUSION_DEVICE` | No | `cuda` | Compute device: `cpu` or `cuda`. |
| `RLFUSION_FORCE_CPU` | No | `false` | Force CPU mode even if CUDA is available. |
| `RLFUSION_ADMIN_KEY` | No | _(empty)_ | Bearer token for `POST /api/fine-tune`. If unset, the endpoint rejects all requests. |

STIS engine environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `STIS_NUM_AGENTS` | `2` | Number of swarm agents |
| `STIS_SIM_THRESHOLD` | `0.92` | Convergence similarity threshold |
| `STIS_ALPHA` | `0.5` | Centroid blending rate |
| `STIS_PORT` | `8100` | Server port |
| `STIS_MODEL_ID` | `Qwen/Qwen2.5-1.5B` | HuggingFace model ID |
| `STIS_IDLE_TIMEOUT` | `120` | Seconds before auto-unloading model from GPU |
| `STIS_MAX_ITERATIONS` | `20` | Max convergence iterations per token |
| `STIS_SEED` | `42` | Random seed for reproducibility |

---

## API Reference

| Method | Path | Rate Limit | Description |
|--------|------|------------|-------------|
| `POST` | `/chat` | 10/min | Query with fused response, weights, and reward |
| `WS` | `/ws` | -- | Streaming chat with real-time pipeline status |
| `GET` | `/api/config` | 10/min | Current configuration |
| `PATCH` | `/api/config` | 10/min | Update config at runtime |
| `GET` | `/ping` | 10/min | Health check (GPU status, models loaded, VRAM) |
| `POST` | `/api/upload` | 10/min | Upload documents to `data/docs/` |
| `POST` | `/api/reindex` | 3/min | Rebuild knowledge graph index |
| `DELETE` | `/api/reset` | 5/min | Wipe transient state (cache, episodes, replay) |
| `POST` | `/api/fine-tune` | 1/hour | Trigger LoRA SFT (requires `RLFUSION_ADMIN_KEY`) |
| `GET` | `/api/images/{path}` | -- | Serve processed images |
| `GET` | `/metrics` | -- | Prometheus metrics |

Full interactive documentation is auto-generated at `/docs` when the server is running.

### POST /chat Response Shape

```json
{
  "response": "...",
  "fusion_weights": {"cag": 0.4, "graph": 0.6},
  "reward": 0.85,
  "proactive_suggestions": ["...", "..."]
}
```

### WebSocket Done Message

```json
{
  "type": "done",
  "response": "...",
  "fusion_weights": {"cag": 0.4, "graph": 0.6},
  "reward": 0.85,
  "proactive": "...",
  "proactive_suggestions": ["..."],
  "query_expanded": false,
  "expanded_query": null,
  "web_status": "disabled"
}
```

### Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `rlfusion_query_latency_seconds` | Histogram | End-to-end query processing latency |
| `rlfusion_retrieval_path_total` | Counter | Per-path retrieval usage |
| `rlfusion_fusion_weight` | Histogram | Weight distribution per path |
| `rlfusion_safety_gate_triggers_total` | Counter | Blocked queries |
| `rlfusion_critique_reward` | Histogram | Reward score distribution |
| `rlfusion_replay_buffer_size` | Gauge | Replay buffer episode count |
| `rlfusion_stis_routing_total` | Counter | STIS routing events (resolved/failed/skipped) |
| `rlfusion_http_requests_total` | Counter | HTTP requests by endpoint and method |
| `rlfusion_ws_connections_active` | Gauge | Active WebSocket connections |

A Grafana dashboard template is included at `scripts/grafana/dashboard.json`.

---

## Project Structure

```
backend/
  main.py                  # FastAPI entry, pipeline wiring, Prometheus metrics, WS streaming
  config.py                # YAML config loader (cfg, PROJECT_ROOT, path helpers)
  config.yaml              # All runtime configuration with safe defaults
  agents/
    base.py                # BaseAgent protocol, PipelineState, QueryComplexity types
    orchestrator.py        # LangGraph DAG, complexity classification, prompt assembly
    retrieval_agent.py     # CAG + GraphRAG dispatch (2-path)
    fusion_agent.py        # RL policy inference, heuristic fallback, context building
    critique_agent.py      # Dedicated LLM scoring, reward computation
    safety_agent.py        # Regex + OOD + LLM safety gate
  core/
    asymmetric_llm.py      # Dual-model orchestrator: CPU triage + GPU executor
    metrics.py             # VRAM/RAM monitoring via pynvml, background gauges
    retrievers.py          # CAG cache, GraphRAG traversal, CSWR scoring, context packing
    fusion.py              # Weight normalization, context merging (2-path)
    critique.py            # Critique parsing, STIS gate, safety checks, faithfulness
    stis_client.py         # STIS httpx client, axiom formatting, SQLite audit
    decomposer.py          # Heuristic query decomposition (LLM call removed for speed)
    memory.py              # Entity extraction, anaphora resolution, session state
    profile.py             # Persistent user profile (SQLite)
    utils.py               # BGE embeddings, chunking, OOD Mahalanobis detection
    graph_engine.py        # NetworkX graph, Qdrant entity search, Leiden communities
    reasoning.py           # ORPS tree reasoning, faithfulness cache
    model_router.py        # MoE model selection per task type
    multimodal.py          # CLIP embeddings, PDF image extraction, cross-modal search
    scheduler.py           # Hardware profiling, quantization recommendation, task lanes
  rl/
    fusion_env.py          # Gymnasium env (394-dim obs, 2D continuous action)
    train_rl.py            # CQL offline training (d3rlpy)
    train_ppo.py           # PPO online, GRPO, AdaptivePolicy
    train_dpo.py           # DPO preference learning
    fine_tune.py           # LoRA SFT + GGUF export
    federated.py           # Differential privacy, FedAvg delta aggregation
    add_batch_episodes.py  # Seed replay buffer with synthetic episodes
    generate_training_data.py  # CoT trace extraction
    train_real_episodes.py # Training from real interaction episodes
    train_from_db.py       # Training from SQLite replay buffer
    train_supervised.py    # Supervised fine-tuning pipeline
  tools/
    base.py                # BaseTool protocol, ToolInput/ToolOutput types
    registry.py            # Thread-safe registry with per-tool rate limiting
    calculator.py          # AST-based math eval, unit conversions
    code_executor.py       # Sandboxed subprocess Python execution
    api_bridge.py          # Generic REST API calls with URL validation
stis_engine/
    swarm.py               # Core convergence loop, hidden state extraction, sampling
    model_loader.py        # Qwen2.5-1.5B lazy loading with GPU/CPU fallback
    config.py              # Frozen dataclass config with env var overrides
    schemas.py             # Pydantic request/response models
    server.py              # FastAPI /generate and /health with idle auto-unload
    __main__.py            # CLI entrypoint (python -m stis_engine)
frontend/
  src/
    App.tsx                # Multi-session chat, file upload, WebSocket management
    types/
      contracts.ts         # Shared TS interfaces (Message, Weights, ChatResponse, etc.)
    components/
      AgentPipeline.tsx    # Live per-agent status visualization
      ChatInput.tsx        # Message input with file drag-and-drop
      ChatList.tsx         # Chat history sidebar
      ChatMessage.tsx      # Markdown-rendered message bubbles
      FusionMeter.tsx      # Real-time CAG/Graph weight distribution display (2-bar)
      MonitoringPanel.tsx  # Weight history, reward tracking, system health
      SettingsPanel.tsx    # Runtime config toggles
      Sidebar.tsx          # Navigation and document management
      ConnectionStatus.tsx # WebSocket health indicator
      Header.tsx           # App header
models/
  rl_policy_cql.d3         # Pre-trained CQL policy (~3.2 MB)
  qwen2.5-1.5b-instruct-q4_k_m.gguf  # CPU triage model (~1.1 GB)
  Meta-Llama-3.1-8B-Instruct-Q8_0.gguf  # GPU executor model (~8 GB)
scripts/
  retrain_fusion.py        # Offline retraining for 2-path CQL policy
  init_db.sh               # SQLite database initialization
  profile_pipeline.py      # End-to-end latency profiler
  grafana/dashboard.json   # Grafana dashboard template
  compatibility/
    fix_blackwell.sh       # NVIDIA Blackwell (RTX 50-series) CUDA fix
tests/                     # 623 tests across 12 files
  benchmarks/              # Ground-truth evaluation framework
    ragchecker.py          # Retrieval precision/recall/F1@k
    hotpotqa.py            # Multi-hop QA (exact-match + token-F1)
    truthfulqa.py          # Hallucination detection with trap questions
    runner.py              # Unified runner with 7-day regression detection
```

---

## Test Suite

**623 tests** across 12 test files. Run the full suite:

```bash
python -m pytest tests/ -v
```

| File | Tests | Coverage |
|------|-------|----------|
| `tests/test_core_units.py` | 168 | CSWR scoring, stability, fit, drift, packing, critique, fusion, memory, utils, decomposer, graph engine, reasoning, model router, fine-tuning, multimodal, FusionEnv |
| `tests/test_tools.py` | 84 | BaseTool protocol, calculator (math + units), code executor (sandbox + banned modules), API bridge, registry (rate limiting + dispatch) |
| `tests/test_agents.py` | 66 | Agent protocol conformance, safety/retrieval/fusion/critique agents, orchestrator routing, LangGraph DAG, prompt generation |
| `tests/test_phase4_rl.py` | 52 | ReplayFusionEnv, PPO training, DPO preference pairs, GRPO, AdaptivePolicy transitions |
| `tests/test_phase8_edge.py` | 50 | Hardware detection, quantization recommendation, lane scheduling, federated delta extraction/aggregation, DP noise |
| `tests/test_asymmetric.py` | 49 | Asymmetric pipeline routing, CAG exact/semantic match, GraphRAG traversal, 2-path RL weights, OOM fallback, CPU/GPU isolation, VRAM/RAM metrics, frontend contract alignment, backward compat |
| `tests/test_phase9_benchmarks.py` | 40 | RAGChecker, HotpotQA, TruthfulQA, benchmark runner, Prometheus metrics validation, Grafana dashboard schema |
| `tests/test_stis_engine.py` | 29 | Convergence invariants (dimension stability, centroid preservation, monotonicity), token sampling, schemas, config loading |
| `tests/test_stis_integration.py` | 29 | STIS client fallbacks (timeout, unreachable, HTTP errors), SQLite audit logging, orchestrator STIS wiring |
| `tests/test_stis_contradiction.py` | 26 | Contradiction detection thresholds, dual-condition routing logic, BGE embedding similarity |
| `tests/test_inference_engine.py` | 26 | Inference engine abstraction, model loading, response parsing |
| `tests/test_api.py` | 4 | API endpoint smoke tests |

---

## Benchmarks

Ran on an RTX 5070 with Llama 3.1 8B. Six stress-test suites, 500 iterations each. All passed.

| Suite | Iterations | Pass | Key Metrics | Avg Latency |
|-------|-----------|------|-------------|-------------|
| Hallucination | 500 | **Yes** | Stable filtering, no crashes | ~11.2s |
| Proactive | 500 | **Yes** | 1.0 anticipation rate, 0.936 coherence | ~10.2s |
| Adversarial | 500 | **Yes** | 1.0 robustness, 0.65 jailbreak resistance | ~9.7s |
| Evolution | 500 | **Yes** | 1.0 drift resistance, 0.965 stability | ~10.0s |
| Extensibility | 500 | **Yes** | 0.97 weight stability | ~10.1s |
| Ethics & Bias | 500 | **Yes** | 1.0 safety, fairness >= 0.983 | ~10.0s |

**Overall pass rate:** 100%

Note: These benchmarks were measured on the pre-upgrade 4-path architecture using Ollama. The 2-path asymmetric pipeline is expected to show different latency characteristics. Updated benchmarks will be published after a full re-run with the new architecture.

### Ground-Truth Evaluation

The `tests/benchmarks/` framework provides three evaluation suites that measure retrieval quality and response faithfulness directly:

| Suite | What It Measures | Metrics |
|-------|-----------------|---------|
| **RAGChecker** | Retrieval quality against known-relevant documents | Precision@k, Recall@k, F1@k |
| **HotpotQA** | Multi-hop question answering | Exact match, token-level F1 against gold answers |
| **TruthfulQA** | Hallucination detection using trap questions | Correct/incorrect answer classification accuracy |

The `BenchmarkRunner` orchestrates all three, persists results as JSON, and detects regressions against trailing 7-day averages.

```bash
python -m pytest tests/test_phase9_benchmarks.py -v
```

---

## Hardware Compatibility

| Platform | Support |
|----------|---------|
| Linux x86_64 + NVIDIA GPU (12+ GB VRAM) | Full support (asymmetric dual-model + STIS) |
| Linux x86_64 CPU-only | Functional (both models on CPU, slower inference) |
| NVIDIA Blackwell (RTX 50-series) | Supported natively with CUDA arch 100/120 flags |
| NVIDIA Ampere/Ada (RTX 30/40-series) | Supported with appropriate CUDA arch flags |

### Blackwell GPU Setup

```bash
source venv/bin/activate
./scripts/compatibility/fix_blackwell.sh
```

Installs PyTorch with proper Blackwell CUDA support. Only needed for RTX 5070/5080/5090.

### Memory Requirements

| Component | RAM | VRAM |
|-----------|-----|------|
| CPU triage model (Qwen 2.5 1.5B Q4_K_M) | ~2 GB | 0 |
| GPU executor model (Llama 3.1 8B Q8_0) | 0 | ~9 GB |
| Embedding model (BGE-small-en-v1.5) | ~0.5 GB | ~0.5 GB |
| STIS engine (optional, Qwen2.5-1.5B fp16) | 0 | ~3 GB |
| **Total (without STIS)** | **~2.5 GB** | **~9.5 GB** |
| **Total (with STIS)** | **~2.5 GB** | **~12.5 GB** |

---

## Known Limitations

- **GPU required for asymmetric pipeline** -- the full dual-model setup needs 12+ GB VRAM. CPU-only mode works but is significantly slower.
- **Multimodal dependencies are optional** -- CLIP, PyMuPDF, and Pillow must be installed separately for image processing. The system degrades gracefully without them.
- **Federated learning is local-only** -- delta extraction, DP noise, and aggregation work, but no network transport layer exists for cross-instance communication yet.
- **STIS requires additional GPU headroom** -- the Qwen2.5-1.5B float16 model needs ~3 GB VRAM on top of the main pipeline. CPU fallback exists but is impractically slow.
- **Benchmarks measured on pre-upgrade architecture** -- the stress-test latency numbers were measured with the 4-path Ollama setup. The 2-path asymmetric pipeline should be faster, but new benchmarks have not yet been run.
- **Single-user design** -- no multi-tenant auth or per-user session isolation. Built for personal/development use.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and pull request guidelines.

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting. **Do not open public issues for security vulnerabilities.**

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## License

[MIT](LICENSE) -- Copyright (c) 2025-2026 Bradley R. Kinnard
