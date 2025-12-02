# RLFusion Orchestrator

> A multi-source retrieval fusion engine with offline reinforcement learning.
> Originally built for personal offline use, now open-sourced for public benefit.

**Author:** Bradley R. Kinnard
**License:** MIT

---

## Overview

This is a local-first AI assistant that combines multiple retrieval sources and uses reinforcement learning to dynamically weight them based on query type. It runs entirely on your hardware with Llama3.1 8B via Ollama.

Four retrieval sources get blended together and an RL policy decides the mix:

- **RAG** — vector search with a stability filter I call CSWR (more on that below)
- **CAG** — a local cache for stuff you explicitly want remembered
- **Graph** — NetworkX graph for when you need multi-hop connections between concepts
- **Web** — optional, off by default, for when local context isn't enough

The policy is trained offline using Conservative Q-Learning (CQL).

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) with llama3.1:8b-instruct-q4_0 model
- Optional: CUDA-capable GPU for faster embeddings

### Installation

```bash
# Clone the repository
git clone https://github.com/moonrunnerkc/rlfusion-orchestrator.git
cd rlfusion-orchestrator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r backend/requirements.txt

# Copy environment config
cp .env.example .env
# Edit .env to add your API keys if using web search

# Initialize the database (run the setup script)
./scripts/init_db.sh
```

### Running

```bash
# Start the backend
uvicorn backend.main:app --port 8000

# Optional: Start the frontend
cd frontend && npm install && npm run dev

# Optional: Retrain the policy
python backend/rl/train_rl.py
```

---

## The CSWR thing

Standard RAG retrieval pulls garbage chunks all the time, especially from PDFs with bad formatting. CSWR (Chunk Stability Weighted Retrieval) is my fix — it runs a second pass on retrieved chunks and scores them for stability before they get anywhere near the prompt.

Two signals: token entropy (if it's all over the place, probably junk) and embedding variance across sentences (if a chunk contradicts itself halfway through, kill it). Anything below 0.7 gets downweighted or dropped entirely.

Costs maybe 5% more compute. Killed something like 15-20% of the hallucinations I was seeing on messy research papers. Worth it.

---

## Architecture

The CQL policy adjusts how much weight goes to each source depending on the query. Graph gets more weight when chasing relationships between things. RAG dominates for document lookups. CAG handles explicitly cached content.

There's also:
- A **proactive layer** that tries to guess what you might ask next
- **Citation tracking** so you can see which chunks contributed to an answer
- A **safety classifier** that refuses dangerous requests (see `backend/core/critique.py`)
- **OOD detection** using Mahalanobis distance for graceful fallback

### Key Files

```
backend/main.py            — FastAPI entry point
backend/config.yaml        — Configuration (customize here)
backend/core/retrievers.py — CSWR lives here
backend/core/critique.py   — Safety and reward logic
backend/rl/train_rl.py     — CQL training script
rl_policy_cql.d3           — Pre-trained model
```

---

## Configuration

Edit `backend/config.yaml` to customize:

```yaml
llm:
  model: llama3.1:8b-instruct-q4_0
  host: http://localhost:11434

embedding:
  model: all-MiniLM-L6-v2
  device: cpu  # or "cuda" for GPU

web:
  enabled: false  # Set true and add TAVILY_API_KEY to .env for web search
```

### Environment Variables

Create a `.env` file from `.env.example`:

```bash
# Required only if web search is enabled
TAVILY_API_KEY=your_key_here

# Optional device override
RLFUSION_DEVICE=cpu
```

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## License

MIT. Do whatever you want with it.

– Brad
