# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend (FastAPI, Python 3.10+)

```bash
# Run the server (loads both GGUF models at startup; GPU model needs CUDA-built llama-cpp-python)
uvicorn backend.main:app --port 8000 --reload

# Initialize the SQLite cache/replay/episodes DB (required once before first run)
./scripts/init_db.sh
```

Both Qwen2.5-1.5B (CPU triage, Q4_K_M) and Llama-3.1-8B (GPU generation, Q8_0) GGUF files must be pre-downloaded into `models/` before the backend will start. See README §"Download model artifacts".

The backend refuses to boot without `RLFUSION_ADMIN_KEY` set (>= 32 chars). Mutating endpoints (`/api/reset`, `/api/config`, `/api/upload`, `/api/reindex`, `/api/fine-tune`) and `/metrics` require it as a bearer token.

For Blackwell (RTX 50-series), llama-cpp-python must be compiled with `-DCMAKE_CUDA_ARCHITECTURES=100;120`. See `scripts/compatibility/fix_blackwell.sh` and the Dockerfile.

### Tests

```bash
# Default suite (excludes GPU-only tests; safe in CI)
pytest tests/ -v --tb=short -m "not gpu"

# GPU-only tests (require CUDA)
pytest tests/ -v --tb=short -m "gpu"

# Single file / single test class / single test
pytest tests/test_core_units.py -v
pytest tests/test_core_units.py::TestScoreChunks -v
```

`pytest.ini_options` in `pyproject.toml` defines three markers (`gpu`, `slow`, `integration`) and sets `testpaths = ["tests", "backend/tests"]`.

### Lint / format

```bash
black backend/ tests/
isort backend/ tests/ --profile black
flake8 backend/ tests/ --ignore=E501,W503     # E501 owned by black; W503 conflicts with PEP8
```

### Frontend (React + Vite + TS, Node 18+)

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
npm run build      # tsc -b && vite build
npm run lint       # eslint
```

### RL training

No pre-trained policy ships with the repo (the v1 checkpoint was trained on a broken reward signal and was deleted in v2.0.0). Train one from real chat episodes when you have enough data:

```bash
python -m backend.rl.train_rl --seed 42      # offline CQL on the live episodes table
```

`scripts/eval_ragas.py` evaluates the resulting policy against a small RAGAs-style benchmark. See `RELEASES.md` for the v2 changelog.

### Docker

```bash
docker compose --profile gpu up    # requires nvidia-container-toolkit
docker compose --profile cpu up    # CPU fallback, slow
```

Models, data, and the SQLite DB are bind-mounted from the host. `models/` mounts read-only; the container runs as a non-root user.

## Architecture

The "big picture" sits in the README. A few cross-cutting facts are not obvious from reading any single file.

### Asymmetric dual-model orchestrator (`backend/core/asymmetric_llm.py`)

Two GGUF models live in one Python process behind a single orchestrator:

- **CPU triage worker** (Qwen2.5-1.5B Q4_K_M, `n_gpu_layers=0`) handles JSON-shaped tasks: `intent_parse`, `safety_check`, `cag_lookup`, `graph_trigger`, `obs_build`.
- **GPU executor** (Llama-3.1-8B Q8_0, `n_gpu_layers=-1`) handles free-form generation: `generation`, `critique`, `faithfulness`.

`route_task()` is wired but the live request path does not call it yet; chat traffic currently goes straight to the GPU executor via `backend/core/model_router.py`. See RELEASES.md "Known follow-ups".

Triage calls have a JSON contract enforced via `json_repair` (one corrective-prompt retry on failure) and assume short structured output. Generation calls have no JSON safety net.

OOM on the GPU executor triggers a temporary CPU offload + log; recovery on next successful load. Per-model locks make access thread-safe.

### Pipeline order (must be preserved end-to-end)

Wired in `backend/main.py` (`/chat`, WS `/ws`) and assembled as a LangGraph DAG in `backend/agents/orchestrator.py`:

1. CAG lookup: SHA-256 + semantic similarity returns immediately on a strong hit (sub-millisecond fast path). Thresholds live under `cag.*` in `backend/config.yaml`.
2. Safety agent: regex pre-filter (lifted into `backend/api/chunk_safety.py` so retrievers and fusion reuse it), then Mahalanobis OOD, then LLM safety classify.
3. Intent decomposition (CPU triage, structured JSON).
4. GraphRAG traversal (NetworkX, Leiden communities, 2-hop, score decay 0.8/hop).
5. Observation vector build (394 dims: 384 embed + 10 features, see `backend/rl/fusion_env.py`).
6. RL policy inference produces 2D softmax weights `[cag, graph]` with a 0.05 floor on each.
7. Fusion (`backend/agents/fusion_agent.build_fusion_context`) screens chunks for prompt injection, CSWR re-ranks graph results, then merges by weight.
8. LLM generation (GPU executor).
9. Critique agent (GPU executor, separate LLM call, not inline) produces the reward.
10. High-reward responses cached back into CAG.
11. Episode (query, weights, reward, ...) appended to SQLite replay buffer.

The orchestrator exposes both a full-pipeline path (for `/chat`) and a `prepare → stream → finalize` split (for WS `/ws` streaming).

### CSWR (Chunk Stability Weighted Retrieval)

`backend/core/retrievers.py::score_chunks` and `build_pack`. Five-axis composite score with domain-adaptive stability thresholds (general 0.70, tech 0.65, code 0.55) detected by keyword matching. Thresholds are periodically recalibrated from replay episodes via `compute_domain_quantiles()`.

### Frontend ↔ backend contract

`frontend/src/types/contracts.ts` mirrors the shapes returned by `/chat` and the WS `done` message. If you change either on the backend, update `contracts.ts` in the same change. `tests/test_asymmetric.py` has a contract-alignment test that catches drift.

### Config is single-source

`backend/config.yaml` is authoritative for all runtime tunables (CSWR weights, RL stage cutoffs, retrieval paths, fusion defaults, CAG thresholds, CORS origins, etc.). Loader: `backend/config.py` exposes `cfg`, `PROJECT_ROOT`, path helpers, and `get_inference_config()` which layers env overrides on top.

Env overrides worth knowing about: `INFERENCE_ENGINE` (`llama_cpp_dual` default), `RLFUSION_DEVICE`, `RLFUSION_FORCE_CPU`, `RLFUSION_ADMIN_KEY` (required at boot; gates all mutating endpoints and `/metrics`).

### SQLite is the persistence layer

`db/rlfo_cache.db` (created by `scripts/init_db.sh`) holds CAG entries, episodes, the RL replay buffer, conversation history, and the persistent user profile. No external DB. Wiping it via `DELETE /api/reset` clears transient state but keeps documents.

### `data/docs/` drives the knowledge graph

Drop `.txt`/`.md`/`.pdf` files there. The entity graph in `data/entity_graph.json` is rebuilt from document content at indexing/reindex time. Do not edit it by hand. `POST /api/reindex` is the supported way to refresh it. Uploads are hash-prefixed in `data/docs/` so a poisoned reupload cannot take the slot of a trusted document.

## Conventions worth knowing

- Python formatter is Black, line length 88; isort uses the Black-compatible profile. `target-version = py310` in `pyproject.toml`.
- Most modules have a `# Author: Bradley R. Kinnard` header. Preserve it when editing.
- The codebase finished migrating from the older 4-path architecture (RAG + Web + CAG + Graph) to 2 paths (CAG + Graph) in v2.0.0. A few stale comments still mention the old paths; new code targets 2-path. Do not reintroduce `faiss` or `tavily` to the hot path. `web.enabled` defaults to `false`.

---

# Style, Code & Content Rules

These rules are enforced at all times, in every response, every commit, every file. No exceptions.

## Writing & Output
- No em dashes anywhere. Use commas, colons, semicolons, parentheses, or separate sentences. Applies to prose, code, comments, docs, everything.
- Spoken register. Contractions always (don't, won't, it's). Fragments fine when clear. Vary sentence length.
- Default to the shortest accurate answer. No headers or formatting unless the content genuinely requires it.
- No topic sentences, no concluding summaries, no wrap-ups. Stop when done.
- Drop filler ("that", "which", "in order to" when removable).
- Never end by offering to continue, expand, or go deeper. No closing questions, no "want me to dig in" offers. Last sentence carries info, not solicits a reply.
- Never restate the same point in different words. One analogy, one pass. If two sentences carry the same insight, kill one.
- No labeled sections in conversational responses. No "Topic:" markers, no bold-as-header, no "label: line" topic transitions.
- Multi-question responses: weave into one flow with natural bridges, not labeled breaks.
- All reasoning is invisible. Output shows results, never process. Never label what you're doing ("One assumption:", "Counter-argument:", "Let me analyze").

## Voice
- Peer register. No deference, no lecturing, no performative thinking.
- Assumed context. Never re-explain what's already known. Match the reader's level, not a general audience.
- Specificity as proof. Prove competence with precise details (numbers, names, tradeoffs), not volume.
- Answer first, justify only if needed or asked.

## Code
- Reads as human-written. Zero AI tells. No generic variable names, no over-commented obvious logic, no boilerplate filler.
- Named exports only. No default exports. (TS/JS)
- kebab-case filenames. (TS/JS) Python keeps existing snake_case.
- Strict TypeScript. No `any`.
- Full JSDoc on all public functions. (Python uses Google-style docstrings per existing convention.)
- 300-line file limit. Decompose past that.
- Tests validate real behavior, not wiring.
- No mocks for things testable directly. Never mock the thing under test.
- Error messages include what failed and what to do about it.
- DRY at 3 repetitions, not before.
- SOLID, pragmatic not dogmatic.
- Every function has at least one test.
- Test names describe behavior, not implementation. No test should require reading the implementation to understand what it verifies.
- Integration tests over unit tests when the boundary is the point.
- Verify correctness before presenting.

## Build Guides & Docs
- Architecture, strategy, steps, tooling choices, reasoning.
- No raw code blocks unless a small snippet is genuinely necessary for clarity.

## README
- Section order: title + one-line description, badges row, "What This Does" (3 sentences max), install/quick start, usage examples, architecture (if complex), API reference (if applicable), contributing, license.
- No AI tells. No "empowering developers" language. No feature walls. No symmetrical pros/cons blocks.
- Every claim backed by a number or a link.
- Write like a human who built the thing, not a marketing team.

## Claims & Accuracy
- No fabricated metrics. Ever.
- Accuracy over hype.
- Every public claim verifiable with evidence.
- For research-style answers: state findings as facts in your own words, collect all sources at the bottom under a single `Sources:` line. No inline citations, no "[source]" markers, no parenthetical links in the body.
