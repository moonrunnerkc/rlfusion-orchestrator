# RLFusion Orchestrator

**Author:** Bradley R. Kinnard

## Overview

RLFusion Orchestrator combines retrieval-augmented generation (RAG), context-augmented generation (CAG), knowledge graph reasoning, and live web search into a unified system. Instead of using fixed weights, a PPO-based RL agent learns optimal source fusion strategies based on automated response critique. The system integrates **Context Stability Weighted Retrieval (CSWR)**, an 8-phase hallucination-resistant retrieval pipeline that scores chunks based on stability, answerability, and semantic drift.

All components run locally with GPU acceleration. No external API dependencies except for the local Ollama instance.

## Key Features

- **Learned Fusion Weights**: PPO policy dynamically adjusts RAG/CAG/Graph/Web weights per query
- **CSWR Integration**: Hallucination-resistant retrieval with stability scoring and context packs
- **Automated Critique**: LLM-as-judge evaluates responses and generates training rewards
- **Offline RL Training**: Train on pre-computed episodes without live environment interaction
- **Proactive Suggestions**: System anticipates follow-up queries based on conversation context
- **Multi-Mode Generation**: Chat, build, and evaluation modes with tailored system prompts
- **WebSocket Streaming**: Real-time response generation with weight transparency

## Architecture

```
backend/
├── main.py                 # FastAPI server with WebSocket endpoints
├── config.yaml             # System configuration
├── requirements.txt        # Python dependencies
├── core/
│   ├── retrievers.py       # RAG, CAG, Graph, Web retrieval + CSWR
│   ├── fusion.py           # Multi-source context fusion
│   ├── critique.py         # LLM-based response evaluation
│   ├── utils.py            # Embeddings, chunking, softmax
│   ├── profile.py          # User memory and profile management
│   └── decomposer.py       # Query intent and entity extraction
└── rl/
    ├── fusion_env.py       # Gymnasium environment for RL training
    ├── train_rl.py         # PPO training script
    └── generate_training_data.py  # Synthetic episode generation

training/
└── training_rl.py          # Standalone training entry point

data/
├── docs/                   # Document collection for RAG indexing
└── synthetic_episodes/     # Training episodes with pre-computed rewards

db/
└── rlfo_cache.db          # SQLite database for CAG and episode storage

indexes/
├── rag_index.faiss        # FAISS vector index
└── metadata.json          # Chunk metadata for RAG

frontend/                   # React application (optional)
```

## Requirements

- **GPU**: CUDA-capable GPU (tested on RTX 40/50 series)
- **Python**: 3.10+
- **Ollama**: Local LLM server with Qwen2.5 or compatible model
- **Storage**: ~10GB for embeddings, index, and dependencies

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/rlfusion-orchestrator.git
cd rlfusion-orchestrator
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Install and Configure Ollama

```bash
# Install Ollama (https://ollama.ai)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull model
ollama pull qwen2.5:72b

# Start Ollama server
ollama serve
```

### 5. Configure System

Edit `backend/config.yaml`:

```yaml
llm:
  host: "http://localhost:11434"
  model: "qwen2.5:72b"

paths:
  docs: "data/docs"
  index: "indexes/rag_index.faiss"
  db: "db/rlfo_cache.db"

rl:
  policy_path: "rl_policy.zip"

cswr:
  top_k: 20
  stability_threshold: 0.65
  answerability_threshold: 0.70

web:
  enabled: false  # Set to true after adding API key
  api_key: "YOUR_TAVILY_API_KEY_HERE"  # Get free key at https://tavily.com
```

**Optional: Enable Web Search**

To enable live web search via Tavily:
1. Get a free API key at [https://tavily.com](https://tavily.com)
2. Add your key to `backend/config.yaml` under `web.api_key`
3. Set `web.enabled: true`

## Usage

### Index Documents

Place PDF, TXT, or Markdown files in `data/docs/`, then build the RAG index:

```bash
cd backend
PYTHONPATH=.. python -c "from core.retrievers import build_rag_index; build_rag_index()"
```

This creates `indexes/rag_index.faiss` and `indexes/metadata.json`.

### Start Server

```bash
cd backend
PYTHONPATH=.. python main.py
```

Server runs on `http://localhost:8000` with WebSocket support at `ws://localhost:8000/ws`.

### Chat Mode

Send queries via WebSocket:

```json
{
  "query": "How does CSWR improve retrieval quality?",
  "mode": "chat"
}
```

The system retrieves from all sources, applies learned fusion weights, generates a response, and streams it back with weight transparency and critique scores.

### Build Mode

For generative system design tasks:

```json
{
  "query": "Design a distributed caching layer",
  "mode": "build"
}
```

Build mode encourages novel synthesis rather than factual recall.

### Train RL Policy

Generate synthetic training episodes:

```bash
cd backend/rl
PYTHONPATH=../.. python generate_training_data.py --episodes 500
```

Train the fusion policy:

```bash
cd training
./run_training.sh --episodes 1000 --seed 42
```

Training logs appear in `training/logs/` and the policy saves to `rl_policy.zip`.

### Evaluate System

Test with known queries:

```bash
cd backend
PYTHONPATH=.. python -m core.critique
```

Or use the `/chat` endpoint with test cases.

## CSWR Pipeline

Context Stability Weighted Retrieval operates in 8 phases:

1. **Query Decomposition**: Extract intent, entities, and expected answer structure
2. **Vector Search**: Retrieve top-k candidates via FAISS similarity
3. **Stability Scoring**: Measure chunk coherence with local neighborhood
4. **Context Packs**: Build 3-chunk windows around high-stability centers
5. **Answerability**: Filter packs that can address query intent
6. **Structured Formatting**: Annotate chunks with scores and metadata
7. **Observability**: Log all retrieval decisions for analysis
8. **Config-Driven**: All thresholds tunable via `config.yaml`

CSWR reduces hallucinations by prioritizing stable, answerable content over raw similarity scores.

## API Endpoints

- `GET /ping` - Health check
- `POST /chat` - Synchronous query endpoint
- `WS /ws` - WebSocket streaming with real-time weights
- `GET /api/config` - Get current configuration
- `PATCH /api/config` - Update configuration (e.g., enable/disable web search)

## Development

Run tests:

```bash
cd backend
PYTHONPATH=.. pytest
```

Format code:

```bash
ruff format .
```

Check types:

```bash
mypy backend/
```

## Performance

- **Embedding**: 384-dim sentence-transformers on GPU (~2ms per query)
- **Retrieval**: CSWR with 20 candidates (~50ms)
- **Fusion**: RL policy inference on CPU (~5ms)
- **Generation**: Streaming via Ollama (depends on model size)

Total latency: ~200-500ms for first token, then real-time streaming.

## Citation

If you use this system in research, please cite:

```bibtex
@software{rlfusion2024,
  title={RLFusion Orchestrator: RL-Optimized Multi-Source Retrieval Fusion},
  author={Kinnard, Bradley R.},
  year={2024},
  url={https://github.com/yourusername/rlfusion-orchestrator}
}
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

Built with: FastAPI, PyTorch, Stable-Baselines3, FAISS, Ollama, Sentence-Transformers, NetworkX.
