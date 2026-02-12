# RLFusion Orchestrator

A local-first system I built after getting fed up with assistants drifting, forgetting context, or changing behavior every time a cloud update rolled through. It blends multiple retrieval paths, scores how each one behaves, and uses offline RL to decide the weight mix for every query. What started as a personal tool slowly turned into something far more capable.

**Author:** Bradley R. Kinnard
**LinkedIn:** https://www.linkedin.com/in/brad-kinnard/

**License:** [MIT](LICENSE)
**Whitepaper:** [WHITEPAPER.md](WHITEPAPER.md) — full technical write-up covering CSWR, CQL routing, self-critique rewards, and benchmark results.

If you use this work in your own project, research, or write-up, please cite this repo and my LinkedIn above.

![RLFusion Orchestrator UI](data/docs/images/ui-screenshot-rlfusion.png)

---

## What This Actually Is

This isn't a "better RAG pipeline". It behaves closer to a local cognitive engine that controls how it retrieves, filters, evaluates, and routes information.

It uses four retrieval paths, a stability filter, a safety layer, a critique layer, and an RL policy that adapts slowly over time. The whole pipeline is transparent. You can watch the weights shift in the UI. Nothing is hidden and nothing leaves your machine.

It runs on consumer hardware with Llama3.1 8B through Ollama and stays consistent regardless of outside changes.

---

## Why I Built It

Assistants drift. They hallucinate. They forget what you just told them. Some of them throw different answers depending on the day of the week. I wanted something stable and private that I could actually trust.

To get that, I had to build:

- multiple retrieval paths
- a stability filter that kills noisy chunks
- a safety and critique layer
- an offline CQL router that won't thrash
- a proactive reasoning pass
- and a full stress test suite to validate behavior across thousands of iterations

This system wasn't built to look impressive. It was built to stay stable and sane.

---

## Architecture

```mermaid
graph TD
    User[User Query] --> Router{RL Router}

    subgraph "Retrieval Layer (Weighted by RL)"
        Router -->|Weight A| RAG[RAG + CSWR]
        Router -->|Weight B| CAG[CAG Cache]
        Router -->|Weight C| Graph[Graph Reasoning]
        Router -->|Weight D| Web[Web Search]
    end

    RAG & CAG & Graph & Web --> Fusion[Fusion Engine]

    subgraph "Safety & Evaluation"
        Fusion --> OOD[OOD Detection]
        OOD --> Attack[Attack Screen]
        Attack --> LLM[Llama 3.1 Generation]
        LLM --> Critique[Critique & Safety]
    end

    Critique -->|Pass| Final[Final Response]
    Critique -->|Fail| RAG

    Final -->|Logs| OfflineRL[Offline RL Policy Update]
```

### Data Flow Summary

1. **Query enters** — routed to all four retrieval paths simultaneously.
2. **CQL policy** evaluates the query embedding and outputs fusion weights (how much to trust each path).
3. **CSWR** filters RAG results using entropy scoring, embedding variance, and contradiction checks — anything below 0.7 stability is removed.
4. **Fusion** merges results weighted by the CQL policy output.
5. **OOD detection** (Mahalanobis) flags out-of-distribution queries before they can destabilize the response.
6. **Attack detection** screens for prompt injection and jailbreak attempts.
7. **LLM generates** a response grounded in the fused context.
8. **Critique layer** self-evaluates the response for quality and safety.
9. **Proactive reasoning** suggests follow-up steps when useful.
10. **Response** is delivered with full transparency — fusion weights, reward scores, and citations visible in the UI.

---

## What It Does

RLFusion runs four retrieval paths that each fill a different role.

### RAG + CSWR

Standard RAG pulls garbage. CSWR filters it using entropy scoring, embedding variance, and basic contradiction checks. Anything below 0.7 stability gets removed or down-weighted. This cuts hallucinations in a noticeable way.

### CAG

A fast, explicit cache for information I want preserved exactly. No interpretation. No drift. Just "store this and don't screw it up".

### Graph

NetworkX for multi-hop reasoning when the query depends on relationships instead of surface text.

### Web

Optional and off by default. Used only if local context genuinely isn't enough. Requires a [Tavily API key](https://tavily.com).

The RL policy (CQL) adjusts retrieval weights per query based on real usage logs. It adapts slowly to avoid behavior swings. The result is predictable routing instead of guesswork.

The system also includes:

- proactive reasoning for next-step suggestions
- citation tracking
- a critique-based safety pass
- Mahalanobis OOD checks to catch out-of-distribution queries

---

## Quick Start

### Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) with `llama3.1:8b-instruct-q4_0` pulled
- Node.js 18+ (for the frontend, optional)
- CUDA GPU optional but helps with embeddings

### Setup

1. **Clone and prepare**

   ```bash
   git clone https://github.com/moonrunnerkc/rlfusion-orchestrator.git
   cd rlfusion-orchestrator

   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r backend/requirements.txt
   ```

2. **Initialize environment**

   ```bash
   cp .env.example .env
   ./scripts/init_db.sh
   ```

3. **Pull the required model**

   ```bash
   ollama pull llama3.1:8b-instruct-q4_0
   ```

4. **Add your documents**

   Drop `.txt`, `.md`, or `.pdf` files into the `data/docs/` directory. These are what the RAG retrieval path searches against. Subdirectories are scanned recursively.

   ```bash
   # example — copy your notes, manuals, research papers, anything
   cp ~/my-notes/*.md  data/docs/
   cp ~/papers/*.pdf   data/docs/
   ```

   The FAISS index is built automatically on first startup. If you add or remove documents later, trigger a rebuild without restarting the server:

   ```bash
   curl -X POST http://localhost:8000/api/reindex
   ```

   Or use the **Reindex Docs** button in the frontend sidebar.

   > **Supported formats:** `.txt`, `.md`, `.pdf`
   > **Chunking:** 400 tokens per chunk, embedded with BGE-small-en-v1.5 (384 dims)
   > **Index location:** `indexes/rag_index.faiss` (auto-generated, safe to delete and rebuild)

### Run the Backend

```bash
uvicorn backend.main:app --port 8000
```

Once running, the interactive API documentation is available at:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Frontend (Optional)

```bash
cd frontend
npm install
npm run dev
```

The UI will be available at [http://localhost:5173](http://localhost:5173).

### RL Training (Optional)

```bash
python backend/rl/train_rl.py
```

### What to Expect on a Fresh Install

RLFusion ships with a pre-trained CQL policy that provides reasonable defaults out of the box, but it genuinely improves with use.

The system learns from every interaction. Each query you send gets scored by the critique layer, and those scores feed back into the RL policy that controls how retrieval sources are weighted. For the first **100–500 interactions**, the system is still calibrating — responses will be decent but the routing won't be personalized to your usage patterns yet.

After that warm-up window:

- Retrieval weights start reflecting what actually works for your queries
- The CAG cache builds up with high-quality answers the system has seen before
- Proactive suggestions become more relevant to your workflow
- The critique layer has enough signal to meaningfully differentiate good from bad responses

This is by design. The policy updates slowly and conservatively to avoid behavior swings. You won't notice a sudden shift — it just gradually gets better at knowing which retrieval path to trust for different types of questions.

If you want to accelerate the warm-up, you can batch-seed episodes:

```bash
python backend/rl/add_batch_episodes.py
python backend/rl/train_rl.py
```

---

## Environment Variables

All configurable environment variables are documented in `.env.example`:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TAVILY_API_KEY` | No | _(empty)_ | Tavily API key for web search. Only needed if `web.enabled: true` in config. Get a free key at [tavily.com](https://tavily.com). |
| `RLFUSION_DEVICE` | No | `cpu` | Compute device: `cpu` or `cuda`. |
| `RLFUSION_FORCE_CPU` | No | `false` | Set to `true` to force CPU mode even if CUDA is available. |
| `OLLAMA_HOST` | No | `http://localhost:11434` | Ollama server URL. Change if Ollama runs on a different host/port. |

Additional configuration is in `backend/config.yaml` — see the [Configuration](#configuration) section below.

---

## Core Pieces

### CSWR

Chunk Stability Weighted Retrieval. Uses entropy, variance, and contradiction checks to keep unstable chunks out of the pipeline.

### RL Routing (CQL)

The offline CQL policy evaluates reliability and adjusts path weights slowly and predictably. Keeps the system from doing mood swings or reward shortcuts.

### Safety and OOD

A lightweight classifier handles dangerous queries. Mahalanobis scoring flags unusual input before it destabilizes the response.

### Proactive Layer

Predicts the next likely steps and offers them when useful. Adds flow without getting in the way.

---

## Configuration

`backend/config.yaml`:

```yaml
llm:
  model: llama3.1:8b-instruct-q4_0
  host: http://localhost:11434
  temperature: 0.72
  max_tokens: 4096

embedding:
  model: all-MiniLM-L6-v2
  device: cpu           # "cpu" or "cuda"

rl:
  policy_path: models/rl_policy_cql.d3

web:
  enabled: false        # Enable to allow web search (requires TAVILY_API_KEY)
  max_results: 3
  search_timeout: 10
```

---

## Project Structure

```
backend/
  main.py              — FastAPI entry point
  config.py            — Configuration loader
  config.yaml          — Default configuration
  core/                — Core retrieval, fusion, critique logic
  rl/                  — Reinforcement learning training scripts
frontend/
  src/                 — React UI (Vite + Tailwind)
models/
  rl_policy_cql.d3     — Pre-trained CQL policy (~3.3 MB)
scripts/
  init_db.sh           — Database initialization
  compatibility/
    fix_blackwell.sh   — NVIDIA Blackwell (RTX 50-series) CUDA fix
tests/                 — Test suites (API, GPU, load testing)
training/              — Training orchestration scripts
```

### Hardware Compatibility

**NVIDIA Blackwell GPUs (RTX 50-series):** If you encounter cuBLAS errors on Blackwell architecture GPUs, run the compatibility fix:

```bash
source venv/bin/activate
./scripts/compatibility/fix_blackwell.sh
```

This installs PyTorch nightly builds with proper Blackwell support. See the script header for details. This is only needed for RTX 5070/5080/5090 or similar Blackwell-based cards.

---

## Benchmarks

Ran on an RTX 5070 with Llama3.1 8B. Six full suites. All passed.

| Suite | Iterations | Pass | Highlights | Avg Latency |
|-------|-----------|------|------------|-------------|
| hallucination | 500 | Yes | no crashes; stable filtering | ~11.2s |
| proactive | 500 | Yes | 1.0 anticipation_rate, 0.936 coherence | ~10.2s |
| adversarial | 500 | Yes | 1.0 robustness, 0.65 jailbreak resist | ~9.7s |
| evolution | 500 | Yes | 1.0 drift resistance, 0.965 stability | ~10.0s |
| extensibility | 500 | Yes | 0.97 weight stability | ~10.1s |
| ethics_and_bias | 500 | Yes | 1.0 safety, fairness ≥ 0.983 | ~10.0s |

**Overall pass rate:** 100 percent

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Send a query and receive a fused response with weights and reward. |
| `WS` | `/ws` | WebSocket endpoint for streaming chat with real-time fusion weights. |
| `GET` | `/api/config` | Get current configuration (web search status, etc.). |
| `PATCH` | `/api/config` | Update configuration at runtime (e.g., toggle web search). |
| `GET` | `/ping` | Health check — returns GPU status and policy availability. |
| `POST` | `/api/upload` | Upload `.txt`, `.md`, `.pdf` files to `data/docs/` (multipart form). |
| `POST` | `/api/reindex` | Rebuild the RAG index from documents in `data/docs/`. |
| `DELETE` | `/api/reset` | Wipe all transient state (cache, episodes, replay, conversations). |

Full interactive documentation is auto-generated by FastAPI at `/docs` when the server is running.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and pull request guidelines.

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting instructions. **Do not open public issues for security vulnerabilities.**

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE) — Copyright (c) 2025-2026 Bradley R. Kinnard
