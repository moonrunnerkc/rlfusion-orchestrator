# RLFusion Orchestrator

A local-first system I built after getting fed up with assistants drifting, forgetting context, or changing behavior every time a cloud update rolled through. It blends multiple retrieval paths, scores how each one behaves, and uses offline RL to decide the weight mix for every query. What started as a personal tool slowly turned into something far more capable.

**Author:** Bradley R. Kinnard  
**LinkedIn:** https://www.linkedin.com/in/brad-kinnard/

**License:** MIT  
If you use this work in your own project, research, or write-up, please cite this repo and my LinkedIn above.

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

## What It Does

RLFusion runs four retrieval paths that each fill a different role.

### RAG + CSWR

Standard RAG pulls garbage. CSWR filters it using entropy scoring, embedding variance, and basic contradiction checks. Anything below 0.7 stability gets removed or down-weighted. This cuts hallucinations in a noticeable way.

### CAG

A fast, explicit cache for information I want preserved exactly. No interpretation. No drift. Just "store this and don't screw it up".

### Graph

NetworkX for multi-hop reasoning when the query depends on relationships instead of surface text.

### Web

Optional and off by default. Used only if local context genuinely isn't enough.

The RL policy (CQL) adjusts retrieval weights per query based on real usage logs. It adapts slowly to avoid behavior swings. The result is predictable routing instead of guesswork.

The system also includes:

- proactive reasoning for next-step suggestions
- citation tracking
- a critique-based safety pass
- Mahalanobis OOD checks to catch out-of-distribution queries

---

## Install

### Requirements

- Python 3.10+
- Ollama with `llama3.1:8b-instruct-q4_0`
- CUDA GPU optional but helps with embeddings

### Setup

```bash
git clone https://github.com/moonrunnerkc/rlfusion-orchestrator.git
cd rlfusion-orchestrator

python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

cp .env.example .env
./scripts/init_db.sh
```

### Run the backend

```bash
uvicorn backend.main:app --port 8000
```

### Frontend (optional)

```bash
cd frontend
npm install
npm run dev
```

### RL training (optional)

```bash
python backend/rl/train_rl.py
```

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

## Architecture Summary

- RAG + CSWR
- CAG
- Graph
- Web (optional)
- CQL policy for routing
- Safety + critique
- OOD detection
- Proactive reasoning

The retrieval pipeline is transparent, and the UI shows the fusion weights shifting in real time so you can see exactly how the system arrived at its answer.

---

## Key Files

- `backend/main.py` – FastAPI entry point
- `backend/config.yaml` – Configuration
- `backend/core/retrievers.py` – CSWR and retrieval logic
- `backend/core/critique.py` – Safety, critique, rewards
- `backend/rl/train_rl.py` – Offline CQL training script
- `rl_policy_cql.d3` – Pre-trained policy

---

## Configuration

`backend/config.yaml`:

```yaml
llm:
  model: llama3.1:8b-instruct-q4_0
  host: http://localhost:11434

embedding:
  model: all-MiniLM-L6-v2
  device: cpu

web:
  enabled: false
```

`.env`:

```
TAVILY_API_KEY=your_key
RLFUSION_DEVICE=cpu
```

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

**Total runtime:** 46 minutes 7 seconds

**Report:** `/tests/results/master_report_all_suites_20251201_203957.json`

### CI Summary:

```
RLFO v1.2 – All 6 hardened suites passed – anticipation 100.0%, latency ~10s, ethics 1.00
```

---

## Real World Example

Dec 1, 2025. I asked it how to build a local AI system for a stuffed-animal shop.

- It created a concept called TeddyGuard.
- Graph reasoning dominated at about 66 percent.
- Proactive reasoning fired cleanly.
- Critique score stayed at 1.00.
- Zero drift. No weird behavior.

UI screenshots in `/docs/screenshots/` show the live weight shifts.

---

## License

MIT. Use it however you want.

If you use it publicly, cite the repo and my LinkedIn:  
https://www.linkedin.com/in/brad-kinnard/
