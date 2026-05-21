# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend (FastAPI, Python 3.10+)

```bash
# Run the server (loads both GGUF models at startup; GPU model needs CUDA-built llama-cpp-python)
uvicorn backend.main:app --port 8000 --reload

# Initialize the SQLite cache/replay/episodes DB (required once before first run)
./scripts/init_db.sh

# STIS contradiction-resolution microservice (optional, separate process, port 8100)
python -m stis_engine
```

Both Qwen2.5-1.5B (CPU triage, Q4_K_M) and Llama-3.1-8B (GPU generation, Q8_0) GGUF files must be pre-downloaded into `models/` before the backend will start. See README §"Download model artifacts".

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
pytest tests/test_stis_engine.py::test_dimension_stability -v
```

`pytest.ini_options` in `pyproject.toml` defines three markers (`gpu`, `slow`, `integration`) and sets `testpaths = ["tests", "backend/tests"]`. Top-level files like `test_attack_detection.py`, `test_multimodal.py`, `test_proactive_critique.py`, `test_proactive_prompting.py` are NOT in `testpaths`. Run them by explicit path.

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

Pre-trained CQL policy ships at `models/rl_policy_cql.d3`. Retraining is optional.

```bash
python backend/rl/train_rl.py             # offline CQL (d3rlpy)
python scripts/retrain_fusion.py          # 500 synthetic episodes for 2-path
python backend/rl/train_ppo.py            # online PPO (after ~50 real interactions)
python backend/rl/train_dpo.py            # DPO (after ~500 preference pairs)
python backend/rl/add_batch_episodes.py   # seed the replay buffer with synthetic data
```

`run_all.sh` runs a full baseline × ablation × seed sweep (8 × 12 × 5). Set `QUICK=1` for a smoke run.

### Docker

```bash
docker compose --profile gpu up    # requires nvidia-container-toolkit
docker compose --profile cpu up    # CPU fallback, slow
```

Models in `./models/` and data in `./data/`, `./db/` are bind-mounted.

## Architecture

The "big picture" sits in the README. A few cross-cutting facts are not obvious from reading any single file.

### Asymmetric dual-model pipeline (`backend/core/asymmetric_llm.py`)

Two GGUF models live in one Python process behind a single orchestrator:

- **CPU triage worker** (Qwen2.5-1.5B Q4_K_M, `n_gpu_layers=0`) handles: `intent_parse`, `safety_check`, `cag_lookup`, `graph_trigger`, `obs_build`.
- **GPU executor** (Llama-3.1-8B Q8_0, `n_gpu_layers=-1`) handles: `generation`, `critique`, `stis_deep`, `faithfulness`.

When adding a new LLM call, decide which worker before writing it. Triage calls have a JSON contract enforced via `json_repair` (one corrective-prompt retry on failure) and assume short structured output. Generation calls have no JSON safety net.

OOM on the GPU executor triggers a temporary CPU offload + log; recovery on next successful load. Per-model locks make access thread-safe.

### Pipeline order (must be preserved end-to-end)

Wired in `backend/main.py` (`/chat`, WS `/ws`) and assembled as a LangGraph DAG in `backend/agents/orchestrator.py`:

1. CAG lookup: SHA-256 + semantic similarity ≥ 0.85 returns immediately (< 5 ms fast path).
2. Safety agent (CPU triage): regex, then Mahalanobis OOD, then LLM safety classify.
3. Intent decomposition (CPU triage, structured JSON).
4. GraphRAG traversal (NetworkX, Leiden communities, 2-hop, score decay 0.8/hop).
5. Observation vector build (394 dims, see `backend/rl/fusion_env.py`).
6. RL policy inference produces 2D softmax weights `[cag, graph]` clamped to ≥ 0.05 each.
7. Fusion (`backend/core/fusion.py`) merges contexts by weight.
8. Contradiction check: only routes to STIS if `cag_vs_graph_cos_sim < 0.40` AND `best_cswr < 0.70` (both required; see `backend/core/critique.py::should_route_to_stis`).
9. LLM generation (GPU executor). If STIS handled this turn, the LLM streaming loop is skipped entirely.
10. Critique agent (GPU executor, separate LLM call, not inline) produces the reward.
11. High-reward responses (≥ 0.70) cached into CAG.
12. Episode (query, weights, reward, ...) appended to SQLite replay buffer.

The orchestrator exposes both a full-pipeline path (for `/chat`) and a `prepare → stream → finalize` split (for WS `/ws` streaming).

### CSWR (Chunk Stability Weighted Retrieval)

`backend/core/retrievers.py::score_chunks` and `build_pack`. Five-axis composite score with domain-adaptive stability thresholds (general 0.70, tech 0.65, code 0.55) detected by keyword matching. Thresholds are periodically recalibrated from replay episodes via `compute_domain_quantiles()`. CSWR's `best_score` is half of the dual-condition STIS gate, so changes here ripple into STIS routing.

### STIS engine (`stis_engine/`)

Separate FastAPI process on port 8100, independent of the backend's main asyncio loop. It lazy-loads Qwen2.5-1.5B fp16 on first `/generate`, auto-unloads after 120s idle. The convergence loop in `swarm.py` enforces three invariants (dimension stability, centroid in convex hull, monotonic similarity). Tests under `tests/test_stis_engine.py` fail if any are broken.

The backend calls STIS via `backend/core/stis_client.py` (httpx + SQLite audit). Any STIS failure (timeout, HTTP error, unreachable) falls back silently to normal LLM generation. Never raise out of the client.

### Frontend ↔ backend contract

`frontend/src/types/contracts.ts` mirrors the exact shapes returned by `/chat` and the WS `done` message. If you change any of these on the backend, update `contracts.ts` in the same change. `tests/test_asymmetric.py` has a contract-alignment test that catches drift.

### Config is single-source

`backend/config.yaml` is authoritative for all runtime tunables (CSWR weights, STIS thresholds, RL stage cutoffs, retrieval paths, fusion defaults, beam width, faithfulness gate, etc.). Loader: `backend/config.py` exposes `cfg`, `PROJECT_ROOT`, path helpers, and `get_inference_config()` which layers env overrides on top.

Env overrides worth knowing about: `INFERENCE_ENGINE` (`llama_cpp_dual` default), `RLFUSION_DEVICE`, `RLFUSION_FORCE_CPU`, `RLFUSION_ADMIN_KEY` (required to call `POST /api/fine-tune`).

### SQLite is the persistence layer

`db/rlfo_cache.db` (created by `scripts/init_db.sh`) holds CAG entries, episodes, the RL replay buffer, the STIS audit table (`stis_resolutions`), and the persistent user profile. No external DB. Wiping it via `DELETE /api/reset` clears transient state but keeps documents.

### `data/docs/` drives the knowledge graph

Drop `.txt`/`.md`/`.pdf` files there. The entity graph in `data/entity_graph.json` is rebuilt from document content at indexing/reindex time. Do not edit it by hand. `POST /api/reindex` is the supported way to refresh it.

## Conventions worth knowing

- Python formatter is Black, line length 88; isort uses the Black-compatible profile. `target-version = py310` in `pyproject.toml`.
- Most modules have a `# Author: Bradley R. Kinnard` header. Preserve it when editing.
- The codebase is mid-migration: an older 4-path architecture (RAG + Web + CAG + Graph) was reduced to 2 paths (CAG + Graph). Some comments still reference the old paths (`no_rag`, `no_web` ablations in `run_all.sh`, `rag_weight` column in the episodes table). New code targets 2-path. Do not reintroduce `faiss` or `tavily` to the hot path. `web.enabled` defaults to `false`.
- `.claude/` is gitignored except `commands/`. The shared OCR slash-command stubs ship with the repo; per-user state (settings, transcripts, caches) stays local.

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

<!-- Managed by `ocr init`. Mirror in AGENTS.md. Do not hand-edit. -->
<!-- OCR:START -->
## Open Code Review Instructions

These instructions are for AI assistants handling code review in this project.

Always open `.ocr/skills/SKILL.md` when the request:
- Asks for code review, PR review, or feedback on changes
- Mentions "review my code" or similar phrases
- Wants multi-perspective analysis of code quality
- Asks to map, organize, or navigate a large changeset

Use `.ocr/skills/SKILL.md` to learn:
- How to run the 8-phase review workflow
- How to generate a Code Review Map for large changesets
- Available reviewer personas and their focus areas
- Session management and output format

Keep this managed block so `ocr init` can refresh the instructions.

<!-- OCR:END -->
