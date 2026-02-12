# RLFusion Orchestrator: Offline Reinforcement Learning for Multi-Path Retrieval Fusion in Local LLM Systems

**Bradley R. Kinnard**
December 2025

---

## Abstract

Current retrieval-augmented generation (RAG) systems use static weighting to combine retrieval sources. When the query distribution shifts or the source quality varies, these fixed weights produce unpredictable outputs. This paper describes RLFusion Orchestrator, a system that replaces static fusion with an offline reinforcement learning policy trained via Conservative Q-Learning (CQL). The system routes queries across four retrieval paths — stability-filtered vector search, an exact-match cache, a knowledge graph, and optional web search — and learns a weighting function from logged interaction data. A stability filter (CSWR) removes low-confidence chunks before fusion, and a self-critique mechanism generates reward signals without human annotation. The system runs entirely on consumer hardware using Llama 3.1 8B through Ollama. Evaluation across six stress-test suites (3,000 total iterations) shows 100% suite pass rate, 0.97 weight stability, 1.0 drift resistance, and 0.65 jailbreak resistance.

---

## 1. Problem Statement

Retrieval-augmented generation has a fundamental stability problem. The standard approach — embed a query, pull the top-k nearest chunks, stuff them into a prompt — works fine until it doesn't. The failure modes are well-documented:

- **Noisy chunks** that score high on vector similarity but contain contradictory or tangential information.
- **Context pollution** where irrelevant retrievals bias the generated response.
- **Behavioral drift** across sessions, caused by non-deterministic retrieval ordering and temperature-dependent generation.

Most systems address these by tuning hyperparameters manually or adding a reranker on top of the retrieval stack. Both approaches are reactive. They fix individual failure cases rather than adapting the retrieval strategy itself.

The harder problem is multi-source fusion. When a system has access to multiple retrieval paths — vector search, a cache, a knowledge graph, web search — someone has to decide how much to trust each one for a given query. Static weights break down immediately. A factoid question should lean on the cache. A multi-hop reasoning question should lean on the graph. A question about recent events needs the web. No fixed ratio works across all of these.

RLFusion was built to solve this specific problem: **learn a query-conditioned weighting function from usage data, without requiring online interaction or human labeling.**

---

## 2. System Architecture

### 2.1 Overview

The system operates as a pipeline with six stages:

1. **Query decomposition** — an LLM call classifies the query's intent, expected answer shape, key entities, and sensitivity level.
2. **Parallel retrieval** — four independent retrieval paths execute concurrently.
3. **Stability filtering (CSWR)** — vector search results are scored for local stability, question fit, and drift, then filtered.
4. **RL-based fusion** — a CQL policy maps query embeddings to retrieval weights.
5. **Safety screening** — OOD detection (Mahalanobis distance) and attack detection run before generation.
6. **Generation with self-critique** — the LLM generates a response and simultaneously produces a structured self-evaluation with reward scores.

The backend is a FastAPI application. The frontend is a React SPA that displays fusion weights in real time. The system communicates via WebSocket for streaming responses.

### 2.2 Retrieval Paths

**RAG + CSWR.** Standard FAISS-based vector search using BGE-small-en-v1.5 embeddings (384 dimensions). Raw results pass through the CSWR filter described in Section 3. Documents are chunked at 400 tokens, stored in a flat L2 index.

**CAG (Cached Answer Graph).** An SQLite-backed exact-match and semantic-similarity cache. Queries are matched first by exact string, then by case-insensitive match, then by embedding cosine similarity (threshold ≥ 0.85). High-reward responses (reward ≥ 0.70) are automatically cached after each interaction, creating a feedback loop where good responses are served instantly on subsequent similar queries.

**Graph Retrieval.** A NetworkX directed graph loaded from a JSON ontology file. Entities are matched to the query via embedding similarity, then the graph is traversed up to two hops. Results carry scores that decay by a factor of 0.8 per hop, naturally preferring tightly connected information.

**Web Search.** Tavily API integration, disabled by default. Used only when local context is insufficient. Results carry a fixed confidence of 0.95 and are formatted with source URLs for citation.

### 2.3 Query Decomposition

Before retrieval, the query is analyzed by the LLM to produce a structured profile:

```json
{
  "primary_intent": "explain|compare|troubleshoot|list|design|summarize",
  "required_facts": ["specific facts needed"],
  "key_entities": ["entities mentioned"],
  "temporal_focus": "past|current|future|null",
  "expected_shape": "definition|list|step-by-step|code|comparison",
  "sensitivity_level": 0.0-1.0
}
```

This profile feeds directly into the CSWR scoring function (Section 3), where it controls the question-fit component. A heuristic fallback handles cases where the LLM fails to return valid JSON — intent is inferred from keyword patterns, entities from capitalization rules.

---

## 3. Chunk Stability Weighted Retrieval (CSWR)

CSWR is the mechanism that distinguishes this system's vector search from standard top-k retrieval. The problem it solves: FAISS returns chunks ranked by vector distance, but distance alone says nothing about whether a chunk is *stable* — whether it sits in a coherent neighborhood of the document, whether it actually addresses the question, and whether it drifts away from surrounding context.

### 3.1 Scoring Components

Each retrieved chunk receives a composite score from four weighted components:

$$\text{CSW} = w_v \cdot S_{\text{vector}} + w_s \cdot S_{\text{stability}} + w_f \cdot S_{\text{fit}} + w_d \cdot S_{\text{drift}}$$

Default weights: $w_v = 0.4$, $w_s = 0.3$, $w_f = 0.2$, $w_d = 0.1$.

**Vector score** ($S_{\text{vector}}$): Transformed FAISS L2 distance, computed as $\frac{1}{1 + d}$ where $d$ is the raw distance. Ranges from 0 to 1, with 1 indicating an exact match.

**Local stability** ($S_{\text{stability}}$): Cosine similarity between a chunk's embedding and its neighbors in the document. If a chunk is semantically distant from its surrounding chunks, it is likely a fragment or noise. Boundary chunks (first or last in a document) receive a 0.15 penalty since they often contain headers, footers, or partial sentences.

**Question fit** ($S_{\text{fit}}$): Measures how well a chunk matches the decomposed query profile. Computed from three sub-signals:
- Entity coverage: fraction of `key_entities` from the query profile found in the chunk text (weight 0.4).
- Fact coverage: fraction of `required_facts` found (weight 0.3).
- Shape match: whether structural indicators in the text match the expected answer shape — e.g., numbered lists for "list" queries, `def`/`class` keywords for "code" queries (weight 0.2 + 0.1 bonus).

**Drift penalty** ($S_{\text{drift}}$): Penalizes chunks that sit in unstable neighborhoods. When a chunk's average neighbor similarity drops below 0.5, a penalty proportional to the deficit is applied. Fully isolated chunks (both neighbors below 0.5) receive a 1.5x severity multiplier.

### 3.2 Domain-Adaptive Thresholds

Stability thresholds are not static. The system maintains per-domain quantile statistics:

- **Tech domain** (detected via keywords like "gpu", "model", "embedding"): stability threshold 0.65
- **Code domain** (detected via keywords like "function", "class", "def"): stability threshold 0.55
- **General domain**: stability threshold 0.70

These quantiles can be recalibrated from logged episodes using `compute_domain_quantiles()`, which recalculates the 25th, 50th, and 75th percentile stability scores per domain and writes them back to the config.

### 3.3 Context Packing

After scoring and filtering, chunks are packed into context windows using a budget-aware algorithm. A center chunk drives the pack; adjacent chunks are pulled in if they meet stability thresholds (≥ 0.65 for main content, CSW ≥ 0.4 for supporting content). The total token budget defaults to 1,800 tokens per pack, and up to four packs are selected per query.

Each pack is then checked for *answerability* — the LLM is asked whether the packed context can answer the user's question, with a binary yes/no and a confidence score. Packs below the answerability threshold (default 0.55) are dropped. If all packs fail, the highest-scoring pack is retained as a fallback.

---

## 4. RL-Based Fusion Routing

### 4.1 Why Offline RL

Online RL is impractical here. Each "step" in this system involves an LLM call, retrieval across multiple backends, and a user-facing response. Exploration in production — deliberately trying bad retrieval weight combinations to learn — is unacceptable.

Offline RL sidesteps this. The policy learns from logged interactions: queries, the weights that were applied, the fused context, and the reward signal from the critique layer. No live exploration is needed.

### 4.2 Conservative Q-Learning (CQL)

The system uses CQL (Kumar et al., 2020) via the d3rlpy library. CQL addresses the distributional shift problem in offline RL — the tendency of Q-learning to overestimate the value of state-action pairs that are underrepresented in the training data. It does this by adding a conservative regularization term that penalizes Q-values for out-of-distribution actions.

Configuration:
- Actor and critic learning rates: $3 \times 10^{-4}$
- Conservative weight: 5.0
- Batch size: 256
- Action scaler: MinMaxActionScaler (normalizes actions to [-1, 1])
- Encoder: two-layer MLP (384 → 256 → 256 → 4)

### 4.3 State-Action Space

**Observation space** (396 dimensions):
- 384-dimensional query embedding from BGE-small-en-v1.5
- 3 top retrieval similarity scores from the RAG path
- 1 average CSWR score across top results
- 1 binary cache hit indicator (1.0 if CAG returned results)
- 1 graph connectivity signal (number of graph results / 10)
- 1 normalized query length (word count / 50)
- 5-dimensional query type vector (factoid, how-to, conceptual, comparative, other)

**Action space** (4 continuous values in [-1, 1]):
Raw logits passed through softmax and clamped to a minimum of 0.05 per source, preventing any retrieval path from being fully zeroed out.

### 4.4 Reward Signal

Rewards come from the self-critique layer (Section 5.2). Each response generates scores for factual accuracy, proactivity, helpfulness, and citation coverage, averaged into a scalar reward in [0, 1]. No human annotation is required.

### 4.5 Training Loop

The training loop runs offline against a replay buffer stored in SQLite:

1. Load episodes from the `episodes` table.
2. Construct `MDPDataset` by embedding queries and extracting weights/rewards.
3. Train CQL for up to 50 epochs, 1,000 gradient steps per epoch.
4. After each epoch, evaluate against the FusionEnv for 10 episodes.
5. Save the model checkpoint if the evaluation reward improves by more than 0.01.
6. Early stop after 3 epochs without improvement.

The trained policy is saved as a PyTorch state dict at `models/rl_policy_cql.d3`. At inference time, a lightweight wrapper loads the encoder and output layers and runs forward passes without the full d3rlpy dependency.

### 4.6 Heuristic Fallback

When the CQL policy outputs near-uniform weights (max - min < 0.05), the system falls back to keyword-based heuristics:

| Query Pattern | RAG | CAG | Graph | Web |
|---------------|-----|-----|-------|-----|
| URLs, "look up" | 0.20 | 0.10 | 0.20 | 0.50 |
| "architecture", "design", "workflow" | 0.20 | 0.10 | 0.60 | 0.10 |
| "what is", "explain", "describe" | 0.60 | 0.20 | 0.10 | 0.10 |
| Default | 0.40 | 0.20 | 0.30 | 0.10 |

This prevents the system from behaving unpredictably on queries the policy hasn't seen enough examples of.

---

## 5. Safety and Quality Layers

### 5.1 Out-of-Distribution Detection

The system uses Mahalanobis distance with Ledoit-Wolf covariance shrinkage to detect OOD queries. The detector is fitted on the embedding distribution of indexed documents. At query time, any query whose Mahalanobis distance exceeds the threshold (default: 50.0) is flagged.

Ledoit-Wolf shrinkage is used instead of raw covariance because the embedding space is 384-dimensional and the number of training documents may be small. Without shrinkage, the covariance matrix is often singular or near-singular, making the Mahalanobis distance numerically unstable.

### 5.2 Self-Critique

Rather than using a separate evaluator model, the system appends a structured self-critique instruction to the generation prompt. The LLM scores its own output on four axes:

- **Factual accuracy** (0.00–1.00)
- **Proactivity** (0.00–1.00)
- **Helpfulness** (0.00–1.00)
- **Citation coverage** (0.00–1.00, computed separately from inline citation counts)

The critique block is parsed from the response via regex, the scores are extracted, and the critique text is stripped before the response reaches the user. The final reward is either the explicit "Final reward" score or the mean of the three sub-scores.

Citation coverage is computed independently by counting `[1]`, `[2]`, etc. markers and dividing by the number of substantive sentences (> 20 characters) in the response. This provides a cross-check against the LLM's self-reported score.

### 5.3 Attack Detection

A lightweight safety classifier runs each query through the LLM with a binary classification prompt (SAFE/UNSAFE). The classifier is trained on patterns covering illegal activity, harm, jailbreak attempts, malware requests, and harassment. Flagged queries are blocked before they reach the generation stage.

### 5.4 Faithfulness Checking

Individual claims can be verified against source chunks. The LLM is shown the source material and asked whether a specific claim is SUPPORTED or not. This is used in evaluation suites but not in the hot path of normal query processing, since the latency cost of per-claim verification is too high for interactive use.

---

## 6. Conversation Memory

The system maintains per-session conversation state with entity tracking, topic stacking, and anaphora resolution.

### 6.1 Entity Extraction

Regex-based extraction identifies four entity types from user messages:
- **Business** (e.g., "a restaurant called The Blue Caboose")
- **Person** (e.g., "someone named John Smith")
- **Location** (e.g., "in Kansas City, MO")
- **Product** (e.g., "the RTX 5070")

Extracted entities are stored in the session's `active_entities` dictionary, keyed by type. The most recent entity of each type overwrites the previous one.

### 6.2 Query Expansion

When a follow-up query contains anaphoric references ("it", "they", "the restaurant") or is very short (≤ 4 words), the system expands it using tracked entities. A question like "what are their hours?" after a conversation about a restaurant is expanded to "what are their hours for The Blue Caboose in Kansas City?"

The expansion logic is rule-based, not LLM-driven, to avoid adding latency. Business-related queries pull business and location entities. Person-related queries pull person entities. Ambiguous queries fall back to the most recently mentioned entity of any type.

### 6.3 Persistent User Profile

Users can store facts about themselves across sessions using explicit commands ("remember this: I prefer dark mode") or implicit statements ("I like Python more than JavaScript"). These are stored in SQLite and injected into the prompt context when the query references personal information. The system distinguishes between memory requests and actual questions using pattern matching, preventing questions like "do you remember my name?" from being treated as storage commands.

---

## 7. Evaluation

All benchmarks were run on a single machine with an NVIDIA RTX 5070 and Llama 3.1 8B (Q4 quantized) through Ollama. Each test suite ran 500 iterations. Total wall-clock time: 46 minutes.

### 7.1 Results

| Suite | Metric | Value |
|-------|--------|-------|
| Hallucination | Error rate | 0.0% |
| Hallucination | Avg latency (p50) | 11,663 ms |
| Proactive | Anticipation rate | 1.000 |
| Proactive | Chain coherence | 0.936 |
| Adversarial | Robustness score | 1.000 |
| Adversarial | Jailbreak resistance | 0.650 |
| Evolution | Drift resistance | 1.000 |
| Evolution | Temporal stability | 0.965 |
| Extensibility | Weight stability | 0.970 |
| Ethics & Bias | Safety score | 1.000 |
| Ethics & Bias | Overall fairness | 0.984 |
| Ethics & Bias | Gender bias score | 0.983 |
| Ethics & Bias | Political bias score | 0.985 |

Pass rate across all suites: **6/6 (100%)**.

Peak memory usage: 1,386 MB (hallucination suite).

### 7.2 Discussion

**Jailbreak resistance at 0.65** is the weakest metric. This is expected — the safety classifier relies on the same 8B parameter LLM that the attacker is trying to manipulate. A dedicated safety model or a prompt-engineering approach with negative examples would likely improve this, but at the cost of additional latency.

**Latency around 10–11 seconds per query** reflects the cost of running a local 8B model for both query decomposition and response generation. This is not competitive with cloud-based systems, but the design goal was privacy and stability, not speed. Upgrading to a faster quantization format or a smaller model would reduce latency linearly.

**Weight stability at 0.97** indicates the CQL policy produces consistent routing decisions across repeated evaluations of the same query. The 3% instability comes from the softmax temperature and the stochastic elements in retrieval scoring, not from policy variance.

---

## 8. Implementation Details

### 8.1 Embedding Model

BGE-small-en-v1.5 (BAAI) was chosen over the originally planned all-MiniLM-L6-v2 for production use. Both produce 384-dimensional embeddings. BGE-small was selected because it consistently ranked higher on MTEB benchmarks for retrieval tasks at the time of development and showed better stability under the CSWR scoring regime.

### 8.2 Database Schema

The system uses a single SQLite database (`db/rlfo_cache.db`) with three tables:

- **`cache`** — key-value store for CAG. Keys are query strings, values are cached responses, scores are confidence levels.
- **`episodes`** — replay buffer for RL training. Stores query, response, reward, individual path weights, fused context, and proactive suggestions.
- **`user_profile`** — persistent user facts, keyed by a hash-based identifier, categorized by type (preferences, identity, work, personality, general).

### 8.3 Hot-Reload Configuration

The `config.yaml` file can be modified at runtime through the `/api/config` PATCH endpoint. Currently, only the web search toggle is exposed through this interface. The config is loaded into memory at startup and mutated in place — there is no config-watching or polling mechanism.

### 8.4 CQL Policy Wrapper

At inference time, the full d3rlpy library is not loaded. Instead, a lightweight PyTorch wrapper (`CQLPolicyWrapper`) loads only the encoder and mu (mean action) layers from the saved state dict. The forward pass is:

```
observation → Linear(384, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 4) → clamp(-1, 1)
```

This reduces import time and memory footprint compared to loading the full CQL algorithm class.

---

## 9. Limitations

**Single-user design.** The system was built for personal use. Multi-user deployment would require session isolation, auth, and per-user replay buffers. The current architecture stores all episodes and profile data in a shared SQLite file.

**LLM-dependent reward signal.** The self-critique reward comes from the same model that generates the response. This creates a self-reinforcing loop — the model will rarely rate its own output as terrible. The reward signal is useful for relative comparisons across different weight configurations, but it should not be treated as a ground-truth quality measure.

**No online adaptation.** The CQL policy is trained offline and deployed as a static artifact. The replay buffer grows continuously as the system is used, but policy updates require manually rerunning the training script. Periodic retraining (e.g., weekly) would be a straightforward extension.

**Latency floor.** Every query requires at least two LLM calls (decomposition and generation) plus embedding computation and retrieval. On an 8B model with Q4 quantization, this puts the minimum latency around 8–10 seconds. There is no batching or speculative decoding.

**FAISS flat index.** The vector index uses `IndexFlatL2`, which performs exact search. This is fine for document collections under ~100k chunks but will not scale to millions without switching to an approximate index (IVF, HNSW).

---

## 10. Related Work

**RAG** (Lewis et al., 2020) established the pattern of retrieval-augmented generation. RLFusion builds on this by adding stability filtering and learned weighting, addressing the "garbage in, garbage out" problem that affects naive RAG.

**RLHF** (Ouyang et al., 2022) uses online RL with human feedback. RLFusion's approach is cheaper — it uses automated self-critique as a reward proxy and trains offline, avoiding the cost and complexity of human annotation pipelines.

**CQL** (Kumar et al., 2020) provides the theoretical foundation for the offline RL component. The conservative regularizer is what prevents the policy from learning to exploit out-of-distribution actions — critical when the replay buffer is small and non-exhaustive.

**Self-Reflection in LLMs** (Shinn et al., 2023; Madaan et al., 2023) explores the idea of LLMs evaluating their own outputs. RLFusion's critique layer is a simplified version of this, using structured scoring templates rather than free-form reflection.

---

## 11. Conclusion

RLFusion Orchestrator demonstrates that offline RL can replace static weights in multi-path retrieval systems, even when the training signal comes from automated self-critique rather than human annotation. The CSWR filter provides a principled mechanism for rejecting low-quality retrieval results before they reach the LLM, and the CQL policy learns query-conditioned routing that adapts to actual usage patterns.

The system is designed for stability over speed, privacy over convenience, and transparency over abstraction. Every fusion weight, every retrieval score, and every critique output is visible to the user in real time.

The source code is available at [github.com/moonrunnerkc/rlfusion-orchestrator](https://github.com/moonrunnerkc/rlfusion-orchestrator) under the MIT license.

---

## References

Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative Q-Learning for Offline Reinforcement Learning. *NeurIPS 2020*.

Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.

Madaan, A., Tandon, N., Gupta, P., et al. (2023). Self-Refine: Iterative Refinement with Self-Feedback. *NeurIPS 2023*.

Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS 2022*.

Shinn, N., Cassano, F., Gopinath, A., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *NeurIPS 2023*.
