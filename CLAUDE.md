# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend (FastAPI, Python 3.10+)

```bash
# Run the server (loads both GGUF models at startup; GPU model needs CUDA-built llama-cpp-python)
RLFUSION_ADMIN_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(48))") \
  uvicorn backend.main:app --port 8000 --reload

# Initialize the SQLite cache/replay/episodes DB (required once before first run)
./scripts/init_db.sh
```

Both Qwen2.5-1.5B (CPU triage, Q4_K_M) and Llama-3.1-8B (GPU generation, Q8_0) GGUF files must be pre-downloaded into `models/` before the backend will start. See README §"Download model artifacts". A `models/CHECKSUMS.txt` manifest is verified at boot; either populate it (`cd models && sha256sum *.gguf > CHECKSUMS.txt`) or set `model_integrity.verify_at_boot: false` in `backend/config.yaml`.

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

`pytest.ini_options` in `pyproject.toml` defines three markers (`gpu`, `slow`, `integration`) and sets `testpaths = ["tests", "backend/tests"]`. Tests that hit admin endpoints set `RLFUSION_ALLOW_WEAK_ADMIN_KEY=1` so short fixture keys are accepted; outside tests, the server refuses to boot with anything shorter than 32 chars.

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

There is no pre-trained policy in the repo any more. The trainer reads the live replay buffer (`db/rlfo_cache.db`) and produces `models/rl_policy_cql.d3`. Run it after roughly 50 real chat turns:

```bash
python -m backend.rl.train_rl                  # offline CQL (d3rlpy)
python -m backend.rl.train_rl --seed 42 --epochs 1   # smoke check
```

`run_all.sh` runs a CQL-only seed sweep over the live replay buffer. Set `QUICK=1` for a 1-seed smoke run. `scripts/sweep_baselines.sh` is a placeholder for the broader multi-baseline sweep, which depends on `train_rl.py` growing `--algo` / `--ablation` flags.

### Docker

```bash
docker compose --profile gpu up    # requires nvidia-container-toolkit
docker compose --profile cpu up    # CPU fallback, slow
```

Models in `./models/` and data in `./data/`, `./db/` are bind-mounted. The image runs as the `rlfusion` user with `cap_drop: [ALL]` and `no-new-privileges`. Models are mounted read-only.

## Architecture

The "big picture" sits in the README. A few cross-cutting facts are not obvious from reading any single file.

### Asymmetric dual-model pipeline (`backend/core/asymmetric_llm.py`)

Two GGUF models live in one Python process behind a single orchestrator:

- **CPU triage worker** (Qwen2.5-1.5B Q4_K_M, `n_gpu_layers=0`) handles: `intent_parse`, `safety_check`, `cag_lookup`, `graph_trigger`, `obs_build`.
- **GPU executor** (Llama-3.1-8B Q8_0, `n_gpu_layers=-1`) handles: `generation`, `critique`, `faithfulness`.

When adding a new LLM call, decide which worker before writing it. Triage calls have a JSON contract enforced via `json_repair` (one corrective-prompt retry on failure) and assume short structured output. Generation calls have no JSON safety net.

OOM on the GPU executor triggers a temporary CPU offload + log. After `_oom_recovery_threshold` successful CPU-fallback calls, the orchestrator force-reloads the GPU model on the next call. The admin-only `POST /api/reset_gpu` endpoint forces an immediate reload. Per-model locks make access thread-safe.

Note: `route_task()` is currently not wired into the live request path. The orchestrator routes work through fusion_agent / critique_agent / main.py directly; the class is loaded but reserved for the future routing wire-up tracked in RELEASES.md "Known follow-ups".

### Pipeline order (must be preserved end-to-end)

Wired in `backend/main.py` (`/chat`, WS `/ws`) and assembled as a LangGraph DAG in `backend/agents/orchestrator.py`:

1. CAG lookup: SHA-256 + semantic similarity returns immediately (< 5 ms fast path). All thresholds live in `cag.*` keys in `backend/config.yaml`.
2. Safety agent (CPU triage): regex, then Mahalanobis OOD, then LLM safety classify.
3. Intent decomposition (CPU triage, structured JSON).
4. GraphRAG traversal (NetworkX, Leiden communities, 2-hop, score decay 0.8/hop).
5. Observation vector build (394 dims = 384-d embedding + 10 retrieval features). Single source of truth: `backend/rl/obs_builder.py`.
6. RL policy inference. Raw action is projected via `project_to_simplex()` (same fn used by `FusionEnv.step`) to enforce a 0.05 floor per path.
7. Fusion (`backend/agents/fusion_agent.py::build_fusion_context`) merges contexts by weight. Every retrieved chunk passes through the prompt-injection scrubber (`backend/api/injection_filter.py`) before slot allocation. Surviving chunks are wrapped in `BEGIN/END UNTRUSTED RETRIEVED CONTEXT` delimiters.
8. LLM generation (GPU executor).
9. Critique agent (GPU executor, separate LLM call, not inline) produces the reward.
10. High-reward responses (≥ `cag.reinsert_reward_threshold`) re-inserted into CAG.
11. Episode (query, weights, reward, obs_features, policy_weights, effective_weights, from_cache, had_empty_path) appended to SQLite replay buffer.

The orchestrator exposes both a full-pipeline path (for `/chat`) and a `prepare → stream → finalize` split (for WS `/ws` streaming).

### CSWR (Chunk Stability Weighted Retrieval)

`backend/core/retrievers.py::score_chunks` and `build_pack`. Five-axis composite score with domain-adaptive stability thresholds (general 0.70, tech 0.65, code 0.55) detected by keyword matching. Thresholds are periodically recalibrated from replay episodes via `compute_domain_quantiles()`.

### Frontend ↔ backend contract

`frontend/src/types/contracts.ts` mirrors the exact shapes returned by `/chat` and the WS `done` message. If you change any of these on the backend, update `contracts.ts` in the same change. `tests/test_asymmetric.py` has a contract-alignment test that catches drift.

### Config is single-source

`backend/config.yaml` is authoritative for all runtime tunables (CSWR weights, RL stage cutoffs, retrieval paths, fusion defaults, beam width, faithfulness gate, CAG thresholds, CORS allowlist, WS rate limits, etc.). Loader: `backend/config.py` exposes `cfg`, `PROJECT_ROOT`, path helpers, and `get_inference_config()` which layers env overrides on top.

Env overrides worth knowing about: `INFERENCE_ENGINE` (`llama_cpp_dual` default), `RLFUSION_DEVICE`, `RLFUSION_FORCE_CPU`, `RLFUSION_ADMIN_KEY` (required to boot; gates `POST /api/fine-tune`, `PATCH /api/config`, `POST /api/upload`, `POST /api/reindex`, `DELETE /api/reset`, `POST /api/reset_gpu`, `GET /metrics`, and the WS endpoint), `RLFUSION_ALLOW_WEAK_ADMIN_KEY` (tests only; bypasses the 32-char floor).

### SQLite is the persistence layer

`db/rlfo_cache.db` (created by `scripts/init_db.sh`) holds CAG entries, episodes, the RL replay buffer, and the persistent user profile. No external DB. Wiping it via `DELETE /api/reset` clears transient state but keeps documents.

### `data/docs/` drives the knowledge graph

Drop `.txt`/`.md`/`.pdf` files there. The entity graph in `data/entity_graph.json` is rebuilt from document content at indexing/reindex time. Do not edit it by hand. `POST /api/reindex` is the supported way to refresh it. Uploaded files are content-addressed (`<sha-prefix>_<stem>.<ext>`) so a poisoned upload cannot clobber existing docs.

## Conventions worth knowing

- Python formatter is Black, line length 88; isort uses the Black-compatible profile. `target-version = py310` in `pyproject.toml`.
- Most modules have a `# Author: Bradley R. Kinnard` header. Preserve it when editing.
- The codebase finished a migration from a 4-path architecture (RAG + Web + CAG + Graph) to 2 paths (CAG + Graph). Do not reintroduce `faiss` or `tavily` to the hot path. `web.enabled` defaults to `false`.
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
