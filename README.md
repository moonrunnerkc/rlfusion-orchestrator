<p align="center">
  <h1 align="center">RLFusion Orchestrator</h1>
  <p align="center">
    <strong>Local-first multi-agent retrieval engine with offline RL routing, chunk stability filtering, and sub-token consensus generation.</strong>
  </p>
  <p align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"/></a>
    <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+"/>
    <img src="https://img.shields.io/badge/tests-544%20passing-brightgreen" alt="544 tests"/>
    <img src="https://img.shields.io/badge/cuda-optional-green" alt="CUDA Optional"/>
    <img src="https://img.shields.io/badge/platform-linux%20|%20docker%20|%20arm-lightgrey" alt="Platforms"/>
  </p>
</p>

**Author:** Bradley R. Kinnard
**LinkedIn:** [linkedin.com/in/brad-kinnard](https://www.linkedin.com/in/brad-kinnard/)
**Whitepaper:** [WHITEPAPER.md](WHITEPAPER.md) — full technical write-up covering CSWR, CQL routing, self-critique rewards, and benchmark results.

If you use this work in your own project, research, or write-up, please cite this repo and the LinkedIn above.

<img width="1921" height="917" alt="ui-screenshot-rlfusion" src="https://github.com/user-attachments/assets/51043408-e3b9-4729-bef8-184e96073169" />

---

<details>
<summary><strong>Table of Contents</strong> (click to expand)</summary>

- [What This Is](#what-this-is)
- [Headline Features](#headline-features)
  - [CSWR — Chunk Stability Weighted Retrieval](#-cswr--chunk-stability-weighted-retrieval)
  - [STIS — Sub-Token Intuition Swarms](#-stis--sub-token-intuition-swarms)
- [Architecture](#architecture)
- [The Four Retrieval Paths](#the-four-retrieval-paths)
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

RLFusion is not another RAG wrapper. It is a local cognitive engine that controls **how** it retrieves, **what** it trusts, and **when** to override its own sources — all running on consumer hardware.

The system runs four retrieval paths in parallel (vector search, semantic cache, knowledge graph, optional web), applies a **stability filter** to kill noisy chunks before they reach the LLM, and uses an **offline RL policy** to learn the optimal weight mix per query type. When retrieval sources contradict each other and confidence is low, a **secondary swarm engine** forces mathematical consensus in continuous latent space before any token is ever sampled.

Everything is transparent. You can watch fusion weights shift in real time from the UI. Nothing is hidden and nothing leaves your machine.

**Core pillars:**

| | |
|---|---|
| **CSWR** | Chunk Stability Weighted Retrieval — scores every chunk on local stability, question fit, and drift before it touches the LLM |
| **STIS** | Sub-Token Intuition Swarms — resolves source contradictions via multi-agent convergence in hidden-state space |
| **CQL/PPO/DPO** | Three-stage adaptive RL policy that learns which retrieval path to trust for each query type |
| **Multi-Agent Pipeline** | LangGraph-orchestrated agents (safety → retrieval → fusion → generation → critique) with complexity-based routing |
| **544 Tests** | Full coverage across 9 test files — core modules, agents, tools, RL, edge cases, benchmarks, STIS engine, contradiction detection, and integration |

---

## Headline Features

### ⚡ CSWR — Chunk Stability Weighted Retrieval

> *Standard RAG pulls garbage. CSWR stops it at the gate.*

CSWR replaces naive top-k retrieval with a four-axis scoring function that evaluates every chunk **before** it enters the generation context. This is the single biggest lever against hallucination in the system.

**How it scores each chunk:**

```
CSW = 0.4 × vector_score + 0.3 × local_stability + 0.2 × question_fit + 0.1 × drift_penalty
```

| Axis | What It Measures | How |
|------|-----------------|-----|
| **Vector Score** | Raw FAISS similarity | `1 / (1 + L2_distance)` — standard but insufficient alone |
| **Local Stability** | Is this chunk coherent with its document neighbors? | Cosine similarity to adjacent chunks. Boundary chunks (first/last in document) receive a 0.15 penalty |
| **Question Fit** | Does this chunk actually address the query? | Entity coverage (40%), required fact coverage (30%), intent keyword matching (20%), answer shape match (10%) |
| **Drift Penalty** | Does this chunk sit in a topic-shifting neighborhood? | Penalizes chunks where average neighbor similarity drops below 0.5. Fully isolated chunks get 1.5× severity |

**Domain-adaptive thresholds** — stability requirements adjust based on detected query domain:

| Domain | Detection | Stability Threshold |
|--------|-----------|-------------------|
| General | Default | 0.70 |
| Tech | Keywords: `gpu`, `model`, `embedding`, `transformer`, `faiss`, etc. | 0.65 |
| Code | Keywords: `function`, `class`, `def`, `import`, `return`, etc. | 0.55 |

**Context Packing** — after scoring, chunks are packed into coherent neighborhoods (up to 1,800 tokens per pack) with an answerability gate: the LLM is asked if the pack can actually answer the question. Packs scoring below 0.55 answerability are dropped.

These thresholds are recalibrated from logged episodes via `compute_domain_quantiles()`, which recalculates the 25th/50th/75th percentile stability scores per domain.

> **Implementation:** `backend/core/retrievers.py::score_chunks`, `build_pack`, `check_answerability`
> **Config:** `backend/config.yaml` under `cswr:` and `cswr_quantiles:`
> **Tests:** 168 unit tests in `tests/test_core_units.py` covering `TestScoreChunks`, `TestComputeStability`, `TestComputeFit`, `TestComputeDrift`, `TestBuildPack`

---

### 🧠 STIS — Sub-Token Intuition Swarms

> *When your sources disagree and confidence is low, don't guess — converge.*

STIS is a secondary generation engine that activates when RAG and Graph retrieval produce contradictory facts **and** CSWR confidence is simultaneously low. Instead of letting the LLM arbitrarily pick one source, STIS achieves consensus through mathematical convergence in continuous latent space — before any discrete token is sampled.

**Dual-condition gate — both must be true:**

| Condition | Threshold | Why |
|-----------|-----------|-----|
| **Contradiction detected** | Cosine similarity between top RAG chunk and top Graph chunk < **0.40** | Sources are discussing fundamentally different or opposing content |
| **Low CSWR confidence** | Best CSWR score across all results < **0.70** | The retrieval pipeline lacks confidence in what it found |

If only one condition fires, Ollama handles it normally — a contradiction without low CSWR means the pipeline is confident enough, and low CSWR without contradiction means sources agree on limited content.

**How the swarm works (per token):**

```
1. EXTRACT  → Run N agents through the model, extract final-layer hidden states
2. MEASURE  → Compute pairwise cosine similarity across all agent hidden states
3. CONVERGE → If mean similarity < 0.92, blend toward centroid:
               state_new = (1 - α) × state + α × centroid
               Repeat until convergence or max 30 iterations
4. SAMPLE   → Project unified centroid through lm_head → logits → top-p sampling
5. APPEND   → Add chosen token to all agent sequences, repeat from step 1
```

**Mathematical invariants enforced at every step:**

- **Dimension Stability** — hidden state vectors never change dimensionality through convergence
- **Centroid Preservation** — the blended centroid remains within the convex hull of agent states
- **Convergence Monotonicity** — mean pairwise similarity is non-decreasing across iterations

**Operational design:**

| Property | Value |
|----------|-------|
| Model | Qwen2.5-1.5B (float16, ~3 GB VRAM) |
| Agents | 2 default (configurable up to 16) |
| Convergence threshold | 0.92 |
| Blending rate (α) | 0.5 |
| Max iterations per token | 30 |
| Server | FastAPI microservice on port 8100S |
| Loading | **Lazy** — model stays off-GPU until first `/generate` request |
| Idle unload | Auto-unloads after 120s of inactivity to free VRAM for Ollama |
| Audit trail | Every routing event logged to SQLite (`stis_resolutions` table) |

When STIS handles a query, the Ollama streaming loop is skipped entirely. The STIS response is sent as a single chunk. If STIS fails (timeout, unreachable, HTTP error), the pipeline falls back to Ollama silently with a logged warning.

> **Full architecture doc:** [STIS_ARCHITECTURE.md](STIS_ARCHITECTURE.md)
> **Engine:** `stis_engine/swarm.py` — core convergence loop and token sampling
> **Client:** `backend/core/stis_client.py` — axiom formatting, httpx POST, SQLite audit
> **Gate:** `backend/core/critique.py::should_route_to_stis`, `detect_contradiction`
> **Config:** `backend/config.yaml` under `stis:`
> **Tests:** 84 tests across `test_stis_engine.py` (29), `test_stis_contradiction.py` (26), `test_stis_integration.py` (29)

---

## Architecture

The system is organized as a **multi-agent pipeline** orchestrated by LangGraph. Each agent follows a `plan() → act() → reflect()` protocol. The orchestrator classifies query complexity and routes through the appropriate agent chain.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER QUERY                                    │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │     ORCHESTRATOR      │
                    │  classify_complexity  │
                    │  simple │ complex │   │
                    │     adversarial       │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │    SAFETY AGENT       │
                    │  • Regex attack scan  │
                    │  • OOD detection      │
                    │  • LLM safety check   │
                    └─────┬─────────┬───────┘
                    blocked         safe
                      │               │
                      ▼               ▼
                  [BLOCKED]   ┌───────────────┐
                              │RETRIEVAL AGENT│
                              └──┬──┬──┬──┬───┘
                                 │  │  │  │
                    ┌────────────┘  │  │  └────────────┐
                    ▼              ▼  ▼                ▼
                ┌──────┐    ┌─────┐  ┌───────┐    ┌──────┐
                │ RAG  │    │ CAG │  │ Graph │    │ Web  │
                │+CSWR │    │Cache│  │  RAG  │    │Search│
                └──┬───┘    └──┬──┘  └──┬────┘    └──┬───┘
                   │           │        │            │
                   └─────┬─────┘────────┘────────────┘
                         │
              ┌──────────▼──────────┐
              │    FUSION AGENT     │
              │  CQL/PPO/DPO policy │
              │  weighted merging   │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │ CONTRADICTION CHECK │────── yes ──▶ STIS ENGINE
              │ sim < 0.40 AND      │              (swarm consensus)
              │ cswr < 0.70?        │                    │
              └──────────┬──────────┘                    │
                    no   │   ◀───────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   LLM GENERATION    │
              │   (Ollama stream)   │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   CRITIQUE AGENT    │
              │  • Factual accuracy │
              │  • Helpfulness      │
              │  • Proactivity      │
              │  • Citation coverage│
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   RESPONSE + UI     │
              │  weights, reward,   │
              │  suggestions shown  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  RL POLICY UPDATE   │
              │  (offline training) │
              └─────────────────────┘
```

**Pipeline step-by-step:**

1. **Query enters** — orchestrator classifies complexity as `simple`, `complex`, or `adversarial`
2. **Safety agent** — three-phase screening: regex attack patterns, Mahalanobis OOD detection, LLM safety classification. Adversarial queries can be blocked before retrieval starts
3. **Retrieval agent** — dispatches to all four paths in parallel. Depth scales with query complexity
4. **RL policy** — CQL/PPO/DPO evaluates the query embedding (396-dim observation) and outputs 4D fusion weights
5. **CSWR** — filters RAG results by stability, fit, and drift. Domain-adaptive thresholds applied
6. **Graph engine** — entity resolution, Leiden community detection, 2-hop traversal for structured context
7. **Fusion agent** — merges results weighted by RL output. Per-path score thresholds enforce quality floors
8. **Contradiction check** — if RAG vs Graph similarity < 0.40 *and* best CSWR < 0.70, routes to STIS
9. **LLM generates** — Ollama streams response grounded in fused context (or STIS resolves it)
10. **Critique agent** — dedicated LLM call scores factual accuracy, helpfulness, proactivity; optional ORPS tree reasoning for high-sensitivity queries
11. **Response delivered** — fusion weights, reward, citations, and proactive suggestions visible in the UI
12. **Episode logged** — query, weights, reward stored in SQLite replay buffer for RL training

---

## The Four Retrieval Paths

### RAG + CSWR

FAISS-based vector search using **BGE-small-en-v1.5** embeddings (384 dimensions). Documents are chunked at 400 tokens and stored in a flat L2 index. Raw FAISS results pass through the CSWR filter (see [Headline Features](#-cswr--chunk-stability-weighted-retrieval)) before entering the fusion stage.

Supports `.txt`, `.md`, and `.pdf` documents. PDFs are text-extracted via PyPDF2. Index rebuilds are triggered via `POST /api/reindex` or the frontend sidebar button.

### CAG (Cached Answer Graph)

SQLite-backed exact-match and semantic-similarity cache. Queries are matched by:
1. Exact string match
2. Case-insensitive match
3. Embedding cosine similarity (threshold ≥ 0.85)

High-reward responses (reward ≥ 0.70) are automatically cached after each interaction, creating a feedback loop where good responses are served instantly on subsequent similar queries. Batch-embedded keys keep lookups fast.

### GraphRAG

NetworkX directed knowledge graph with:
- **Entity resolution** — deduplication via embedding cosine similarity (threshold ≥ 0.92)
- **Leiden community detection** — via igraph/leidenalg when available, connected-components fallback otherwise
- **Multi-hop traversal** — up to 2 hops from matched entities, scores decay by 0.8 per hop
- **Qdrant in-memory** — entity vector + payload search when qdrant-client is installed, numpy cosine fallback otherwise

The graph engine provides co-occurrence bonuses, coherence penalties, and path distance weights that feed into CSWR's question-fit scoring component.

> **Implementation:** `backend/core/graph_engine.py`

### Web Search

Optional, **off by default**. Requires a [Tavily API key](https://tavily.com). Used only when local context is genuinely insufficient. Results carry a fixed 0.95 confidence and include source URLs for citation.

Enable via `web.enabled: true` in `backend/config.yaml` and set the `TAVILY_API_KEY` environment variable.

---

## RL-Based Fusion Routing

The system uses reinforcement learning to decide how much to trust each retrieval path for a given query. No static weights.

### Three-Stage Adaptive Policy

The `AdaptivePolicy` automatically transitions through RL stages as interaction data accumulates:

| Stage | Interactions | Method | Behavior |
|-------|-------------|--------|----------|
| **CQL** | 0–50 | Conservative Q-Learning (offline) | Stable defaults from pre-trained policy. No exploration. Ships with the repo. |
| **PPO** | 50–500 | Proximal Policy Optimization (online) | Weights start reflecting your usage patterns. Cautious exploration. |
| **DPO** | 500+ | Direct Preference Optimization | Preference-based refinement from high-reward vs low-reward episode pairs. |

### Observation Space (396 dimensions)

| Dims | Signal |
|------|--------|
| 384 | Query embedding (BGE-small-en-v1.5) |
| 3 | Top retrieval similarity scores from RAG |
| 1 | Average CSWR score across top results |
| 1 | Binary cache hit indicator |
| 1 | Graph connectivity signal (results / 10) |
| 1 | Normalized query length (words / 50) |
| 5 | Query type vector (factoid, how-to, conceptual, comparative, other) |

### Action Space (4 continuous values)

Raw logits → softmax → clamped to minimum 0.05 per source. No retrieval path is ever fully zeroed out.

### Heuristic Fallback

When the policy outputs near-uniform weights (max − min < 0.05), keyword heuristics take over:

| Query Pattern | RAG | CAG | Graph | Web |
|---------------|-----|-----|-------|-----|
| URLs, "look up" | 0.20 | 0.10 | 0.20 | 0.50 |
| "architecture", "design" | 0.20 | 0.10 | 0.60 | 0.10 |
| "what is", "explain" | 0.60 | 0.20 | 0.10 | 0.10 |
| Default | 0.40 | 0.20 | 0.30 | 0.10 |

### Reward Signal

Rewards come from the critique agent — a dedicated LLM call that scores each response on factual accuracy, proactivity, and helpfulness (each 0.0–1.0). The mean becomes the scalar reward logged to the replay buffer. No human annotation required.

> **Implementation:** `backend/agents/fusion_agent.py::compute_rl_weights`, `backend/rl/train_rl.py`, `backend/rl/train_ppo.py`, `backend/rl/train_dpo.py`

---

## Safety and Quality Layers

### Three-Phase Safety Gate

Every query passes through the safety agent before retrieval begins:

1. **Regex pre-filter** — fast pattern matching for prompt injection, SQL injection, template injection, XSS, jailbreak attempts
2. **OOD detection** — Mahalanobis distance with Ledoit-Wolf covariance shrinkage, fitted on the embedding distribution of indexed documents. Queries exceeding the threshold (default: 50.0) are flagged
3. **LLM safety classification** — binary SAFE/UNSAFE classification via Ollama for patterns that escape regex

### Self-Critique

The critique agent runs a **dedicated, separate LLM call** after generation (not inline — this was moved to a post-generation step for reliability). It returns structured JSON scores:

- **Factual accuracy** (0.0–1.0)
- **Proactivity** (0.0–1.0)
- **Helpfulness** (0.0–1.0)
- **Follow-up questions** (3 specific, non-generic suggestions)

Citation coverage is computed independently by counting `[1]`, `[2]`, etc. markers against substantive sentences.

### Tree-Structured Reasoning (ORPS)

For high-sensitivity queries (sensitivity > 0.7, as determined by query decomposition), the system runs **Outcome-Refining Process Supervision**:

1. Generate N candidates (default beam width: 3) with progressively higher temperature (0.2 → 0.8)
2. Score each via critique
3. Prune candidates below the reward threshold (0.3)
4. Optionally refine top candidates
5. Select the best by composite reward

Exploration trees are logged to the replay buffer for offline RL training.

### Faithfulness Checking

Individual claims can be verified against source chunks. A TTL cache (default 300s) prevents redundant LLM calls for the same claim within the same context window. Used on the hot path only when sensitivity exceeds the gate threshold.

> **Implementation:** `backend/core/reasoning.py`, `backend/core/critique.py`, `backend/agents/safety_agent.py`

---

## Dynamic Tool System

The tool registry dispatches to specialized tools based on query content. Tools follow the `BaseTool` protocol and are rate-limited (default: 10 calls per tool per 60 seconds).

| Tool | What It Does | Safety |
|------|-------------|--------|
| **Calculator** | AST-based math evaluation (no `eval`), unit conversions (length, mass, data, time, temperature) | Safe operators whitelist, no code execution |
| **Code Executor** | Sandboxed Python via subprocess — stdout/stderr capture, 10s timeout | Banned modules list (os, sys, subprocess, socket, etc.), import blocker injected at runtime |
| **Web Search** | Tavily API wrapper | Delegates to existing `tavily_search()`, config-gated |
| **API Bridge** | Generic REST API calls with URL validation | Configurable timeout (15s default) |

> **Implementation:** `backend/tools/registry.py`, `backend/tools/calculator.py`, `backend/tools/code_executor.py`

---

## Conversation Memory

Per-session state tracked across turns with entity extraction and anaphora resolution:

- **Entity extraction** — regex patterns identify business names, people, locations, and products from messages
- **Query expansion** — follow-up queries with pronouns ("what are their hours?") are automatically expanded using tracked entities ("what are their hours for The Blue Caboose in Kansas City?")
- **Demonstrative filtering** — "this project", "that algorithm" are recognized as self-contained and not expanded
- **Persistent user profile** — facts stored via explicit commands ("remember this: I prefer dark mode") or implicit statements. Stored in SQLite, injected into prompt context for personal queries

> **Implementation:** `backend/core/memory.py`, `backend/core/profile.py`

---

## Quick Start

### Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) running locally
- Node.js 18+ (frontend, optional)
- CUDA GPU optional (helps with embeddings and STIS engine)

### 1. Clone and install

```bash
git clone https://github.com/moonrunnerkc/rlfusion-orchestrator.git
cd rlfusion-orchestrator

python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Initialize environment

```bash
cp .env.example .env
./scripts/init_db.sh
```

### 3. Pull the LLM model

```bash
ollama pull dolphin-llama3:8b
```

> The model name is configured in `backend/config.yaml` under `llm.model`. Change it to any Ollama-compatible model.

### 4. Add your documents

Drop `.txt`, `.md`, or `.pdf` files into `data/docs/`. Subdirectories are scanned recursively.

```bash
cp ~/my-notes/*.md  data/docs/
cp ~/papers/*.pdf   data/docs/
```

The FAISS index builds automatically on first startup. To rebuild after adding documents:

```bash
curl -X POST http://localhost:8000/api/reindex
```

Or use the **Reindex Docs** button in the frontend sidebar.

> **Chunking:** 400 tokens per chunk, embedded with BGE-small-en-v1.5 (384 dims)
> **Index location:** `indexes/rag_index.faiss` (auto-generated, safe to delete and rebuild)

### 5. Start the backend

```bash
uvicorn backend.main:app --port 8000
```

Interactive API docs are available at:
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

Three profiles for different hardware:

```bash
# CPU-only
docker compose --profile cpu up

# GPU (requires nvidia-container-toolkit)
docker compose --profile gpu up

# ARM (Jetson, Apple Silicon, Snapdragon)
docker compose --profile arm up
```

Each profile includes the backend, frontend, and an Ollama container. Volumes are mounted for `data/`, `db/`, and `indexes/` so your documents and state persist.

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
| **Agent Pipeline** | Live status of each agent (safety → retrieval → fusion → generation → critique) as the query processes |
| **Fusion Meter** | Real-time visualization of RAG/CAG/Graph/Web weight distribution |
| **Monitoring Panel** | Weight history over time, reward tracking, system health indicators |
| **Chat Interface** | Multi-session chat with localStorage persistence, drag-and-drop file upload |
| **Settings Panel** | Runtime configuration toggles |
| **Connection Status** | WebSocket health indicator |

---

## RL Training

RLFusion ships with a pre-trained CQL policy (`models/rl_policy_cql.d3`, ~3.3 MB) that provides reasonable defaults out of the box. The system improves with use.

### Training commands

```bash
# Offline CQL (ships pre-trained)
python backend/rl/train_rl.py

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
| **LoRA SFT** | Fine-tune the base LLM on high-reward episodes via LoRA adapters, export to GGUF for Ollama |
| **GRPO** | Group Relative Policy Optimization for multi-agent coordination |
| **Federated Learning** | Privacy-preserving policy updates across instances — weight deltas are L2-clipped and Gaussian-noised before sharing |
| **MoE Routing** | Register specialized Ollama models per task type (code, critique, generation) for mixture-of-experts dispatch |

> **Implementation:** `backend/rl/fine_tune.py`, `backend/rl/train_ppo.py::train_grpo`, `backend/rl/federated.py`, `backend/core/model_router.py`

---

## Configuration

All runtime behavior is controlled by `backend/config.yaml`. Every key has safe defaults — the system starts with zero manual configuration.

```yaml
llm:
  model: dolphin-llama3:8b          # any Ollama-compatible model
  host: http://localhost:11434
  temperature: 0.72
  max_tokens: 8192

embedding:
  model: BAAI/bge-small-en-v1.5     # 384-dim embeddings
  device: cuda                       # or cpu

cswr:
  enabled: true
  top_k: 20
  pack_token_budget: 1800
  min_csw_score: 0.25
  answerability_threshold: 0.55
  stability_threshold: 0.7
  vector_weight: 0.4
  local_stability_weight: 0.3
  question_fit_weight: 0.2
  drift_penalty_weight: 0.1

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
  enabled: false
  max_results: 3
  search_timeout: 10

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
| `TAVILY_API_KEY` | No | _(empty)_ | Tavily API key for web search. Only needed if `web.enabled: true`. Free key at [tavily.com](https://tavily.com). |
| `RLFUSION_DEVICE` | No | `cpu` | Compute device: `cpu` or `cuda`. |
| `RLFUSION_FORCE_CPU` | No | `false` | Force CPU mode even if CUDA is available. |
| `OLLAMA_HOST` | No | `http://localhost:11434` | Ollama server URL. |
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
| `STIS_SEED` | `42` | Random seed for reproducibility |

---

## API Reference

| Method | Path | Rate Limit | Description |
|--------|------|------------|-------------|
| `POST` | `/chat` | 10/min | Query with fused response, weights, and reward |
| `WS` | `/ws` | — | Streaming chat with real-time pipeline status |
| `GET` | `/api/config` | 10/min | Current configuration |
| `PATCH` | `/api/config` | 10/min | Update config at runtime |
| `GET` | `/ping` | 10/min | Health check (GPU status, policy loaded) |
| `POST` | `/api/upload` | 10/min | Upload documents to `data/docs/` |
| `POST` | `/api/reindex` | 3/min | Rebuild RAG index |
| `DELETE` | `/api/reset` | 5/min | Wipe transient state (cache, episodes, replay) |
| `POST` | `/api/fine-tune` | 1/hour | Trigger LoRA SFT (requires `RLFUSION_ADMIN_KEY`) |
| `GET` | `/api/images/{path}` | — | Serve processed images |
| `GET` | `/metrics` | — | Prometheus metrics |

Full interactive documentation is auto-generated at `/docs` when the server is running.

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
  main.py                  # FastAPI entry, Orchestrator wiring, Prometheus metrics, WS pipeline
  config.py                # YAML config loader (cfg, PROJECT_ROOT, path helpers)
  config.yaml              # All runtime configuration with safe defaults
  agents/
    base.py                # BaseAgent protocol, PipelineState, QueryComplexity types
    orchestrator.py        # LangGraph DAG, complexity classification, prompt assembly
    retrieval_agent.py     # Parallel RAG/CAG/Graph/Web dispatch
    fusion_agent.py        # RL policy inference, heuristic fallback, context building
    critique_agent.py      # Dedicated LLM scoring, reward computation
    safety_agent.py        # Regex + OOD + LLM safety gate
  core/
    retrievers.py          # CSWR scoring, FAISS indexing, context packing, Tavily search
    fusion.py              # Weight normalization, context merging
    critique.py            # Critique parsing, STIS gate (detect_contradiction,
                           #   should_route_to_stis), safety checks, faithfulness
    stis_client.py         # STIS httpx client, axiom formatting, SQLite audit
    decomposer.py          # LLM query decomposition with heuristic fallback
    memory.py              # Entity extraction, anaphora resolution, session state
    profile.py             # Persistent user profile (SQLite)
    utils.py               # BGE embeddings, chunking, OOD Mahalanobis detection
    graph_engine.py        # NetworkX graph, Qdrant entity search, Leiden communities
    reasoning.py           # ORPS tree reasoning, faithfulness cache
    model_router.py        # MoE model selection per task type
    multimodal.py          # CLIP embeddings, PDF image extraction, cross-modal search
    scheduler.py           # Hardware profiling, quantization recommendation, task lanes
  rl/
    fusion_env.py          # Gymnasium env (396-dim obs, 4D continuous action)
    train_rl.py            # CQL offline training (d3rlpy)
    train_ppo.py           # PPO online, GRPO, AdaptivePolicy
    train_dpo.py           # DPO preference learning
    fine_tune.py           # LoRA SFT + GGUF export
    federated.py           # Differential privacy, FedAvg delta aggregation
    add_batch_episodes.py  # Seed replay buffer with synthetic episodes
    generate_training_data.py  # CoT trace extraction
  tools/
    base.py                # BaseTool protocol, ToolInput/ToolOutput types
    registry.py            # Thread-safe registry with per-tool rate limiting
    calculator.py          # AST-based math eval, unit conversions
    code_executor.py       # Sandboxed subprocess Python execution
    web_search.py          # Tavily wrapper (delegates to retrievers.tavily_search)
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
    components/
      AgentPipeline.tsx    # Live per-agent status visualization
      ChatInput.tsx        # Message input with file drag-and-drop
      ChatList.tsx         # Chat history sidebar
      ChatMessage.tsx      # Markdown-rendered message bubbles
      FusionMeter.tsx      # Real-time weight distribution display
      MonitoringPanel.tsx  # Weight history, reward tracking, system health
      SettingsPanel.tsx    # Runtime config toggles
      Sidebar.tsx          # Navigation and document management
      ConnectionStatus.tsx # WebSocket health indicator
      Header.tsx           # App header
models/
  rl_policy_cql.d3         # Pre-trained CQL policy (~3.3 MB)
scripts/
  init_db.sh               # SQLite database initialization
  grafana/dashboard.json   # Grafana dashboard template
  compatibility/
    fix_blackwell.sh       # NVIDIA Blackwell (RTX 50-series) CUDA fix
tests/                     # 544 tests across 9 files
  benchmarks/              # Ground-truth evaluation framework
    ragchecker.py          # Retrieval precision/recall/F1@k
    hotpotqa.py            # Multi-hop QA (exact-match + token-F1)
    truthfulqa.py          # Hallucination detection with trap questions
    runner.py              # Unified runner with 7-day regression detection
```

---

## Test Suite

**544 tests** across 9 test files. Run the full suite:

```bash
python -m pytest tests/ -v
```

| File | Tests | Coverage |
|------|-------|----------|
| `tests/test_core_units.py` | 168 | CSWR scoring, stability, fit, drift, packing, critique, fusion, memory, utils, decomposer, graph engine, reasoning, model router, fine-tuning, multimodal |
| `tests/test_tools.py` | 84 | BaseTool protocol, calculator (math + units), code executor (sandbox + banned modules), API bridge, web search, registry (rate limiting + dispatch) |
| `tests/test_agents.py` | 66 | Agent protocol conformance, safety/retrieval/fusion/critique agents, orchestrator routing, LangGraph DAG, prompt generation |
| `tests/test_phase4_rl.py` | 52 | ReplayFusionEnv, PPO training, DPO preference pairs, GRPO, AdaptivePolicy transitions |
| `tests/test_phase8_edge.py` | 50 | Hardware detection, quantization recommendation, lane scheduling, federated delta extraction/aggregation, DP noise |
| `tests/test_phase9_benchmarks.py` | 40 | RAGChecker, HotpotQA, TruthfulQA, benchmark runner, Prometheus metrics validation, Grafana dashboard schema |
| `tests/test_stis_engine.py` | 29 | Convergence invariants (dimension stability, centroid preservation, monotonicity), token sampling, schemas, config loading |
| `tests/test_stis_integration.py` | 29 | STIS client fallbacks (timeout, unreachable, HTTP errors), SQLite audit logging, orchestrator STIS wiring |
| `tests/test_stis_contradiction.py` | 26 | Contradiction detection thresholds, dual-condition routing logic, BGE embedding similarity |

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
| Ethics & Bias | 500 | **Yes** | 1.0 safety, fairness ≥ 0.983 | ~10.0s |

**Overall pass rate:** 100%

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
| Linux x86_64 + NVIDIA GPU | Full support (CUDA embeddings + STIS) |
| Linux x86_64 CPU-only | Full support (slower embeddings, no STIS) |
| Linux ARM64 (Jetson, etc.) | Docker profile available |
| macOS (Apple Silicon) | CPU mode via Docker ARM profile |
| NVIDIA Blackwell (RTX 50-series) | Supported — run `./scripts/compatibility/fix_blackwell.sh` if you hit cuBLAS errors |

### Blackwell GPU Fix

```bash
source venv/bin/activate
./scripts/compatibility/fix_blackwell.sh
```

Installs PyTorch nightly with proper Blackwell support. Only needed for RTX 5070/5080/5090.

---

## Known Limitations

- **Multimodal dependencies are optional** — CLIP, PyMuPDF, and Pillow must be installed separately for image processing. The system degrades gracefully without them.
- **Federated learning is local-only** — delta extraction, DP noise, and aggregation work, but no network transport layer exists for cross-instance communication yet.
- **Web search requires an external API key** — Tavily is the only supported provider. No fallback search engine.
- **STIS requires GPU** — the Qwen2.5-1.5B model needs ~3 GB VRAM. CPU fallback exists but is impractically slow for real-time use.
- **WebSocket sessions use connection IDs** — HTTP middleware generates correlation IDs, but WS sessions use their own identifiers.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and pull request guidelines.

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting. **Do not open public issues for security vulnerabilities.**

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## License

[MIT](LICENSE) — Copyright (c) 2025–2026 Bradley R. Kinnard
