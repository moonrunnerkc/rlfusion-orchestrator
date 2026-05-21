# Release notes

## v2.0.0 — 2026-05-20

Net diff against the v1 tip (commit `7fe6210`):

```
75 files changed, 1399 insertions(+), 13258 deletions(-)
```

That's a 11,859 line net reduction. The headline goal of v2 was honesty:
make every README claim trace to a runnable code path, delete what does
not earn its complexity, and replace the broken offline-RL reward signal
with the same scorer the chat path uses.

### Removed

- **`stis_engine/`** — the "Sub-Token Intuition Swarm" microservice and
  all callers. The swarm was N copies of one model with 0.008-scale
  Gaussian noise; it converged in 1–2 iterations because the agents
  started nearly identical. The contradiction trigger fired against a
  hand-typed 6-fact ontology that itself contradicted the rest of the
  codebase. Deletes ~3 GB of VRAM headroom and ~600 lines of glue.
- **`data/ontology.json`** + `_load_ontology_facts` / `detect_contradiction`
  / `should_route_to_stis` in `critique.py`.
- **`backend/core/multimodal.py`** (594 lines) + CLIP/PyMuPDF deps + the
  `/api/images/{path}` endpoint. Not on the hot path; not chat-critical.
- **`backend/rl/federated.py`** (415 lines) — delta extraction, DP noise,
  FedAvg, but no network transport.
- **`backend/core/reasoning.py`** (437 lines) — ORPS beam-search tree
  reasoning gated behind a sensitivity heuristic that rarely fired.
- **`backend/rl/train_dpo.py`** + the GRPO function in `train_ppo.py`.
  Optimizing on top of a broken reward compounds error.
- **Fake academic benchmarks**: `tests/benchmarks/{hotpotqa,truthfulqa,
  ragchecker}.py` and `tests/test_phase9_benchmarks.py` — each was a
  10-question self-referential RLFO sanity check that shared a name with
  the real benchmark.
- **Dead FAISS RAG path**: `retrieve_rag()`, `retrieve_rag_structured()`,
  `build_rag_index()`, `get_rag_index()`, and the FAISS index machinery.
- **STIS UI**: `App.tsx` STIS panel, `AgentPipeline.tsx` STIS detail,
  `/ping` STIS health probe.
- **Stale RL scripts**: `train_supervised.py`, `train_from_db.py`,
  `train_real_episodes.py`, `generate_training_data.py` (all 4-path).
- **Stale top-level test files**: `test_attack_detection.py`,
  `test_multimodal.py`, `test_proactive_critique.py`,
  `test_proactive_prompting.py`, `train_with_cpu_embeddings.py`.
- **The v1 pre-trained CQL policy**: `models/rl_policy_cql.d3` — trained
  on the broken reward signal documented below.
- **`STIS_ARCHITECTURE.md`** and stale doc claims in `README.md`,
  `WHITEPAPER.md`, `CSWR.md`. The whitepaper now carries an
  ARCHIVED-as-historical-context header.

### Added / changed

- **CSWR on the hot path** — `FusionAgent.build_fusion_context()` now
  runs `score_chunks()` on the merged graph result set and gates by
  `cswr.min_csw_score` (default 0.25). CAG entries still go through the
  raw `score >= 0.85` threshold.
- **`FusionEnv.step()` reward parity** — calls the live
  `engine.generate()` and `critique()` so training reward and serving
  reward come from the same scorer. `RLFUSION_ENV_DRY_RUN=1` skips the
  generator for CI-only smoke testing.
- **2-path episodes schema** — `init_db.sh` no longer creates
  `rag_weight`; `scripts/migrate_episodes_to_two_path.py` drops the
  column from existing DBs. `log_episode_to_replay_buffer()` detects
  whichever schema is in place.
- **`backend/rl/train_rl.py` rewritten** — loads 2-path episodes, trains
  CQL with 2D actions, evals against the live `FusionEnv` between
  epochs, saves to `models/rl_policy_cql.d3` on improvement.
- **`scripts/eval_ragas.py`** — 3-strategy × 3-metric RAGAs-style
  evaluation over `data/benchmarks/ragas_qa.jsonl` (50 manually-curated
  Q&A pairs grounded in the project's own docs at
  `data/docs/rlfusion/`). Reports `context_relevance`,
  `answer_relevance`, `faithfulness`. No LLM judge needed.
- **README rewrite** — drops the STIS/multimodal/100%-pass-rate
  headline framing; describes the actual `safety → retrieve → fuse →
  generate → critique` path; adds the Benchmarks section and the v2
  changelog.

### Why the v1 RL policy is gone

`FusionEnv.step()` in v1 generated `context[:400] + "Next Steps boilerplate"`
as the "response", then critiqued the boilerplate. The CQL policy
optimized weights that scored well against truncated-context-plus-boilerplate,
which has nothing to do with the real Llama 3.1 8B output the chat path
emits. Any policy derived from that signal was learning a different
problem than the one it served at inference. v2 deletes the checkpoint
and the orchestrator falls back to heuristic weights until a new policy
is trained from real chat episodes via `python backend/rl/train_rl.py`.

### Known follow-ups

- The Benchmarks table's `learned` row is empty by design until a
  fresh CQL policy is trained against real chat episodes.
- The episodes table's `rag_weight` column lingers on existing v1
  databases until `scripts/migrate_episodes_to_two_path.py` is run.
- The asymmetric LLM orchestrator's `route_task()` is loaded but not
  wired into the live request path. Phase 5 of the overhaul plan covers
  it as an A/B decision after the eval table is filled.
- 384-dim raw embeddings in the 394-dim observation space remain
  undertrained relative to dataset size; deferred until the dataset
  grows.
- Mahalanobis OOD threshold (`50.0`) is a magic number; deferred until
  there is real production telemetry to calibrate against.
