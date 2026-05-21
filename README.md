# RLFusion Orchestrator

Local-first RAG chatbot where an offline-trained CQL policy picks per-query weights for two retrieval paths (CAG cache + GraphRAG) and a dedicated critique LLM produces the reward signal.

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![CUDA 12.1+](https://img.shields.io/badge/cuda-12.1%2B-76B900)
![Inference: llama-cpp-python](https://img.shields.io/badge/inference-llama--cpp--python-orange)
![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000)

**Author:** Bradley R. Kinnard ([LinkedIn](https://www.linkedin.com/in/brad-kinnard/))

<img width="1921" height="917" alt="RLFusion UI" src="https://github.com/user-attachments/assets/51043408-e3b9-4729-bef8-184e96073169" />

---

## What this does

Two retrieval sources serve every chat turn: a SHA-256/embedding-keyed answer cache (CAG) and an entity-graph + document-chunk store (GraphRAG). A CQL policy trained offline on past episodes outputs a 2-D softmax `[w_cag, w_graph]` for the current query embedding, and the orchestrator fuses both contexts using those weights before calling the local Llama 3.1 8B generator. Every response gets scored by a separate critique LLM call, the reward is logged as a new episode, and high-reward turns are written back into CAG.

The repo is built for one user, on one box, with no cloud calls. The fusion meter in the UI shows the live `[w_cag, w_graph]` split for every turn.

---

## Quick start

```bash
# 1. SQLite cache + replay buffer
./scripts/init_db.sh

# 2. (optional) override the inference engine and model
cp .env.example .env
# edit .env to set INFERENCE_MODEL=<your-installed-ollama-model>
# or leave it unset to let the resolver auto-pick

# 3. Backend (FastAPI on :8000)
uvicorn backend.main:app --port 8000 --reload

# 4. Frontend (Vite on :5173)
cd frontend && npm install && npm run dev
```

**Engine resolution.** The default config requests `llama_cpp_dual` (the
dual-GGUF asymmetric path). If those GGUFs aren't on disk,
`backend/core/engine_detect.resolve_inference_config()` falls back to
ollama at `http://localhost:11434` and picks the highest-scoring
installed model (general chat preferred, 15-40 GB band preferred). Pin
a specific model with `INFERENCE_MODEL=<name>` in `.env`. The active
engine and model show up on `/ping` as `engine_resolution`. No model
name is hard-coded in the committed config.

**Embedding device.** `embedding.device: auto` in `backend/config.yaml`
turns into cuda if available, mps on Apple Silicon, cpu everywhere
else. Force with `RLFUSION_DEVICE` or `RLFUSION_FORCE_CPU`.

The v1 pre-trained CQL policy was trained on a broken reward signal and
has been deleted (see [overhaul-plan.md](overhaul-plan.md) Phase 3). The
orchestrator falls back to heuristic weights when no policy is present.
Train a fresh one with `python backend/rl/train_rl.py` once the
`episodes` table has at least a few hundred real chat turns.

For Blackwell (RTX 50-series) GPUs, llama-cpp-python must be compiled with
`-DCMAKE_CUDA_ARCHITECTURES=100;120`. See `scripts/compatibility/fix_blackwell.sh`.

---

## Architecture

The hot path, in order, lives in `backend/main.py::websocket_endpoint`
and `backend/agents/orchestrator.py`:

1. **Memory expand + complexity classify** — `step_preprocess()` runs the
   regex-based classifier and short-term memory enrichment.
2. **Safety gate** — `SafetyAgent` runs three tiers: pattern regex, OOD
   Mahalanobis distance against the embedding distribution, then a keyword
   blocklist. LLM calls do not sit on this path.
3. **Retrieval** — `retrieve()` in `backend/core/retrievers.py` checks CAG
   first; a strong cache hit (score ≥ 0.90) skips graph entirely. Otherwise
   the graph path merges entity traversal (`GraphEngine.hybrid_search`)
   with semantic doc-chunk matches.
4. **Fusion** — `FusionAgent.act()` embeds the query, runs the CQL policy
   to produce `[w_cag, w_graph]`, clamps each weight to ≥ 0.05, and builds
   the LLM context using CSWR-scored chunks.
5. **Generation** — Llama 3.1 8B streams tokens via llama-cpp-python with
   layers pinned to GPU. CAG-fast-path turns skip generation entirely.
6. **Critique + reward logging** — a separate LLM call scores the response
   (factual / proactivity / helpfulness, 0–1 each), the average reward is
   logged into the `episodes` table, and turns with reward ≥ 0.70 are
   cached back into CAG.

### Models

| Worker         | Model                              | Quant   | Device |
|----------------|------------------------------------|---------|--------|
| Triage         | Qwen 2.5 1.5B Instruct             | Q4_K_M  | CPU    |
| Generator      | Llama 3.1 8B Instruct              | Q8_0    | GPU    |

The asymmetric dual-model orchestrator (`backend/core/asymmetric_llm.py`)
loads both at startup with per-model thread locks and an OOM fallback that
re-loads the GPU model in CPU mode. `route_task()` exists but is not
wired into the live request path yet; it is reserved for Phase 5 of the
overhaul plan.

### CSWR — Chunk Stability Weighted Retrieval

`score_chunks()` in `backend/core/retrievers.py` ranks graph results on:

```
CSW = 0.35·vector + 0.25·local_stability + 0.20·question_fit
    + 0.10·drift_penalty + 0.10·project_coherence
```

Stability thresholds adapt to detected domain (`general` 0.70 / `tech`
0.65 / `code` 0.55) and are periodically recalibrated from logged
episodes via `compute_domain_quantiles()`.

### Frontend ↔ backend contract

`frontend/src/types/contracts.ts` mirrors the shapes returned by `/chat`
and the WebSocket `done` message. Any change to those shapes on the
backend must update `contracts.ts` in the same commit. `tests/test_asymmetric.py`
has a contract-alignment test that catches drift.

---

## API

| Method | Path              | Notes                                                    |
|--------|-------------------|----------------------------------------------------------|
| POST   | `/chat`           | Single-shot chat with full pipeline                       |
| WS     | `/ws`             | Streaming chat with per-step pipeline status               |
| GET    | `/ping`           | Health probe + boot id + device/model info              |
| GET    | `/metrics`        | Prometheus metrics                                       |
| POST   | `/api/upload`     | Upload .txt/.md/.pdf to `data/docs/`                     |
| POST   | `/api/reindex`    | Rebuild chunk metadata and entity graph                  |
| DELETE | `/api/reset`      | Wipe cache, episodes, replay, conversations              |
| POST   | `/api/fine-tune`  | LoRA SFT on high-reward episodes (admin key required)     |
| GET/PATCH | `/api/config`  | Read/update runtime web config                           |

---

## Tests

```bash
# Default (excludes GPU-only tests)
pytest tests/ -v --tb=short -m "not gpu"

# GPU-only
pytest tests/ -v --tb=short -m "gpu"
```

`pytest.ini_options` in `pyproject.toml` defines `gpu`, `slow`, and
`integration` markers and sets `testpaths = ["tests", "backend/tests"]`.

---

## RL training

The repo ships with no pre-trained CQL policy. The orchestrator falls
back to heuristic 2-path weights until you train one.

```bash
# (one-time) migrate the episodes table from 4-path to 2-path
python3 scripts/migrate_episodes_to_two_path.py

# train CQL on whatever real chat episodes you have accumulated
python backend/rl/train_rl.py

# optional: seed the replay buffer by replaying a batch of queries
python backend/rl/add_batch_episodes.py
```

`backend/rl/fusion_env.py::FusionEnv.step()` now calls the live local
generator and the same `critique()` function the chat path uses, so
training reward and serving reward come from the same scorer. The
training run is bounded by data quality, not data quantity — see
Levine et al. 2020 (arXiv 2005.01643) on CQL with small high-quality
datasets.

---

## Benchmarks

```bash
# 50-query RAGAs-style eval over data/docs/rlfusion/
python3 scripts/eval_ragas.py

# CI-friendly: skip the LLM, score fused context as the answer
python3 scripts/eval_ragas.py --dry-run
```

The script compares three fusion strategies (uniform `[0.5, 0.5]`,
heuristic `_heuristic_weights()`, learned CQL) on the 50-question set
at `data/benchmarks/ragas_qa.jsonl`. It reports three metrics per
strategy:

| metric | what it measures |
|--------|------------------|
| `context_relevance` | does the fused context contain the ground-truth span (substring OR embedding cosine ≥ that)? |
| `answer_relevance`  | embedding cosine between the generated answer and the reference answer |
| `faithfulness`      | fraction of reference-answer content tokens that appear in the fused context |

These are simplified, local proxies for the RAGAs metrics in
Es et al. 2023 (arXiv 2309.15217); they do not require an LLM judge so
they are reproducible from a single command. Each run also dumps the
full per-query results to `tests/results/ragas_<timestamp>.json`.

The "learned" column falls back to uniform `[0.5, 0.5]` when no
`models/rl_policy_cql.d3` is present, which is the default state of the
repo after the overhaul. Train a policy with `python backend/rl/train_rl.py`
and re-run the eval to populate it with real numbers.

### Reference run (v2.0.0, dry-run, empty CAG)

Captured locally on macOS / CPU, no GPU, no LLM (`--dry-run`), CAG
cache empty, freshly-built entity graph over `data/docs/rlfusion/`
(15 chunks, 227 entities):

| strategy   |   n | context_relevance | answer_relevance | faithfulness |
|------------|----:|------------------:|-----------------:|-------------:|
| uniform    |  50 |             0.799 |            0.614 |        0.685 |
| heuristic  |  50 |             0.799 |            0.614 |        0.685 |
| learned    |  50 |             0.799 |            0.614 |        0.685 |

All three are identical here on purpose: with an empty CAG cache the
graph path is the only thing producing context, and the per-strategy
slot allocation has no effect when CAG is empty. The heuristic policy
does produce three distinct weight vectors per query (`cag` ∈
`{0.3, 0.5, 0.6}`); the metric collapse is downstream of fusion, not
upstream. To get meaningful differentiation between strategies, run
the eval without `--dry-run` (real LLM) and against a CAG cache
populated by some actual chat traffic.

Raw per-query JSON for that run: `tests/results/ragas_1779338015.json`.

## Docker

```bash
docker compose --profile gpu up    # requires nvidia-container-toolkit
docker compose --profile cpu up    # CPU fallback, slow
```

`./models/`, `./data/`, and `./db/` are bind-mounted.

---

## Configuration

`backend/config.yaml` is authoritative for all runtime tunables (CSWR
weights, RL stage cutoffs, retrieval paths, fusion defaults, beam width,
faithfulness gate, etc.). `backend/config.py` exposes `cfg`,
`PROJECT_ROOT`, path helpers, and `get_inference_config()`, which layers
env overrides on top.

Env overrides worth knowing about: `INFERENCE_ENGINE`
(`llama_cpp_dual` default), `RLFUSION_DEVICE`, `RLFUSION_FORCE_CPU`,
`RLFUSION_ADMIN_KEY` (required to call `POST /api/fine-tune`).

---

## Persistence

`db/rlfo_cache.db` (created by `scripts/init_db.sh`) holds CAG entries,
episodes, the RL replay buffer, conversation history, and the user
profile. No external DB. `DELETE /api/reset` clears the transient
tables but keeps documents.

---

## What changed in v2 (overhaul)

This branch removed roughly 11k lines of dead or self-referential code:
the STIS sub-token-swarm engine, the hand-typed ontology contradiction
trigger, the unwired multimodal/CLIP path, the network-less federated
learning scaffold, the ORPS tree-search reasoning module, the DPO+GRPO
trainers, the FAISS-era `retrieve_rag()` path, and a stack of fake
"benchmark" files that ran 10-question self-graded sanity checks. See
[overhaul-plan.md](overhaul-plan.md) for the full record.

---

## Contributing

Pull requests welcome for bug fixes and small features. For larger
changes, open an issue first to discuss scope. No CLA. MIT license.

If you build on this, cite the repo and the LinkedIn above.

---

## License

MIT. See [LICENSE](LICENSE).
