# RLFusion Orchestrator (RLFO)

**Author:** Bradley R. Kinnard

RLFO is my attempt at building a reinforcement learning-driven system that figures out how to mix three different retrieval strategies: RAG (vector search over documents), CAG (cached lookups for speed), and graph-based semantic reasoning. The idea is that instead of hard-coding weights like "use 40% RAG, 30% CAG, 30% graph", an RL agent (PPO) learns what works based on feedback.

Runs entirely local. No cloud. GPU-accelerated on my RTX 5070. Uses Ollama for LLM calls, FAISS for embeddings, SQLite for cache, NetworkX for graphs. Everything is PyTorch under the hood.

The goal: build something that doesn't just retrieve and generate, but actually learns from mistakes and gets better at knowing when to use each retrieval mode. Sort of like adaptive RAG but with actual learning instead of heuristics.

## What it does (or will do)

- **Dynamic fusion weights**: RL policy decides how much to trust RAG vs CAG vs graph for each query
- **Learns from critique**: LLM generates answers, then critiques them. RL agent gets rewards based on accuracy/coherence/proactivity
- **Proactive suggestions**: Tries to anticipate follow-ups or catch errors before they happen (this part is experimental)
- **Offline training**: Uses replay buffers so it can improve without needing constant retraining or cloud data
- **Fully local**: Everything runs on my machine. No API calls except to local Ollama

Right now it's missing a lot. The core loop works but the critique module and RL training need more tuning.

## Why I'm building this

Mostly scratching my own itch. I got tired of RAG systems that can't figure out when they're hallucinating or when they should just pull from cache instead of doing expensive vector searches. Thought RL might help since it can optimize for multiple objectives (speed, accuracy, proactivity) instead of just "retrieve top-k and hope".

Also wanted something that runs entirely offline because I don't trust cloud providers with my data and I have a perfectly good GPU sitting here.

## Current status

**Working:**
- FastAPI backend with WebSocket support
- GPU-locked embedding pipeline (sentence-transformers on CUDA)
- Utility functions: chunking, softmax, deterministic hashing, batch embeddings

**In progress:**
- Retriever modules (RAG/CAG/graph)
- RL environment and PPO training loop
- Critique module

**Not started:**
- Frontend (basic React app planned but not essential)
- Test suite (need synthetic episodes for offline training)

Check `backend/core/utils.py` for what's done so far. Everything follows the structure from my design doc (rlfo.pdf). No shortcuts, no scope creep.

## How to run (eventually)

Not ready yet. When it is:
```bash
cd backend
python main.py
```

Requires CUDA. Won't work on CPU (I hardcoded GPU checks everywhere because that's what I have).

## Notes to self

- Need to benchmark whether the RL overhead is worth it vs just using static weights
- Figure out if proactive suggestions are actually useful or just annoying
- Add proper logging so I can debug the fusion weights
- Maybe switch from PPO to SAC if continuous action space works better
- Don't forget to add rate limiting before this talks to Ollama too much

---

This is a personal project. Code quality varies. Some parts are polished, others are "make it work first". If you're reading this and want to use it, wait until I tag a v1.0 or something.
