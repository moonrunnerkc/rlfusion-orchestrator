# Author: Bradley R. Kinnard
# generate_training_data.py - Synthetic training episodes + CoT trace extraction
# Phase 5: adds chain-of-thought training data from high-reward replay episodes

import json
import random
import argparse
import sqlite3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

DOMAIN_TEMPLATES = {
    "technical_docs": {
        "weight": 0.15, "optimal_sources": {"rag": 0.6, "cag": 0.2, "graph": 0.15, "web": 0.05},
        "queries": ["What is {concept}?", "Explain {concept} in detail", "How does {concept} work?",
                    "Define {concept}", "Describe the {concept} implementation"],
        "concepts": ["reinforcement learning", "neural networks", "transformers", "attention mechanism",
                     "gradient descent", "backpropagation", "convolutional networks", "recurrent networks",
                     "LSTM", "GRU", "PPO algorithm", "Q-learning", "policy gradients", "actor-critic",
                     "transfer learning", "fine-tuning", "embeddings", "vector databases", "FAISS",
                     "semantic search", "RAG architecture", "knowledge graphs", "graph databases",
                     "Neo4j", "ontologies", "RDF", "SPARQL", "web scraping", "HTTP protocols"]
    },
    "architecture": {
        "weight": 0.20, "optimal_sources": {"rag": 0.15, "cag": 0.10, "graph": 0.6, "web": 0.15},
        "queries": ["How does the {system} architecture work?", "Explain the {system} workflow",
                    "What components make up {system}?", "Describe the {system} design pattern",
                    "How do components interact in {system}?"],
        "concepts": ["microservices", "event-driven systems", "message queues", "pub/sub patterns",
                     "REST APIs", "GraphQL", "WebSocket connections", "real-time systems",
                     "distributed systems", "load balancing", "caching layers", "database sharding",
                     "CQRS pattern", "event sourcing", "saga pattern", "service mesh",
                     "API gateway", "reverse proxy", "container orchestration", "Kubernetes architecture",
                     "Docker networking", "service discovery", "circuit breakers", "retry patterns"]
    },
    "web_current": {
        "weight": 0.10, "optimal_sources": {"rag": 0.15, "cag": 0.10, "graph": 0.15, "web": 0.6},
        "queries": ["Look up {url}", "Tell me about {website}", "What's on {url}?",
                    "Browse {website} and summarize", "Get information from {url}"],
        "concepts": ["https://github.com/openai/gpt-4", "https://huggingface.co/models",
                     "https://pytorch.org/docs", "https://www.tensorflow.org", "https://docs.python.org",
                     "https://arxiv.org/list/cs.AI/recent", "https://news.ycombinator.com",
                     "https://stackoverflow.com/questions", "https://www.reddit.com/r/MachineLearning",
                     "https://paperswithcode.com"]
    },
    "cached_frequent": {
        "weight": 0.10, "optimal_sources": {"rag": 0.15, "cag": 0.6, "graph": 0.15, "web": 0.10},
        "queries": ["What is the capital of {place}?", "Who created {thing}?", "When was {event}?",
                    "Quick fact about {topic}", "Tell me about {person}"],
        "concepts": ["France", "Python programming", "the internet", "World War II", "Einstein",
                     "Shakespeare", "Leonardo da Vinci", "Tesla", "the moon landing", "Bitcoin",
                     "quantum computing", "DNA", "gravity", "photosynthesis", "evolution", "the solar system"]
    },
    "coding": {
        "weight": 0.15, "optimal_sources": {"rag": 0.4, "cag": 0.15, "graph": 0.3, "web": 0.15},
        "queries": ["How to implement {task} in Python?", "Show me {task} code example",
                    "Best practices for {task}", "Debug {problem}", "Optimize {task} performance"],
        "concepts": ["binary search", "quicksort", "merge sort", "dynamic programming", "memoization",
                     "recursion", "async/await", "threading", "multiprocessing", "REST API",
                     "database connection pooling", "caching strategy", "error handling", "logging",
                     "testing", "CI/CD pipeline", "Docker deployment"]
    },
    "research": {
        "weight": 0.15, "optimal_sources": {"rag": 0.4, "cag": 0.10, "graph": 0.20, "web": 0.3},
        "queries": ["Latest research on {topic}", "What are recent advances in {field}?",
                    "Survey of {topic} methods", "State of the art in {field}", "Compare {topic} approaches"],
        "concepts": ["large language models", "retrieval augmented generation", "few-shot learning",
                     "prompt engineering", "chain of thought", "constitutional AI", "RLHF",
                     "multimodal models", "vision transformers", "diffusion models",
                     "reinforcement learning from human feedback", "AI safety", "alignment",
                     "interpretability", "explainable AI", "federated learning"]
    },
    "troubleshooting": {
        "weight": 0.10, "optimal_sources": {"rag": 0.35, "cag": 0.10, "graph": 0.4, "web": 0.15},
        "queries": ["How to fix {error}?", "Why is {problem} happening?", "Troubleshoot {issue}",
                    "Debug {error} in {system}", "Resolve {problem}"],
        "concepts": ["memory leak", "segmentation fault", "null pointer exception", "connection timeout",
                     "rate limiting", "authentication error", "CORS error", "database deadlock",
                     "race condition", "infinite loop", "stack overflow", "out of memory", "permission denied"]
    },
    "conceptual": {
        "weight": 0.05, "optimal_sources": {"rag": 0.35, "cag": 0.20, "graph": 0.30, "web": 0.15},
        "queries": ["What is the philosophy of {concept}?", "Explain the theory behind {concept}",
                    "Why does {concept} matter?", "Implications of {concept}", "Future of {concept}"],
        "concepts": ["artificial general intelligence", "consciousness", "free will", "ethics in AI",
                     "technological singularity", "human-AI collaboration", "digital privacy",
                     "open source", "decentralization", "automation"]
    }
}


def generate_episode(domain: str, template: dict, episode_id: int) -> dict:
    query_template = random.choice(template["queries"])
    concept = random.choice(template["concepts"])
    query = query_template.format(
        concept=concept, system=concept, url=concept if "http" in str(concept) else f"https://{concept}.com",
        website=concept, place=concept, thing=concept, event=concept, topic=concept,
        person=concept, task=concept, problem=concept, field=concept, error=concept, issue=concept
    )
    optimal = template["optimal_sources"]
    noise = {k: random.gauss(0, 0.05) for k in optimal}
    noisy = {k: max(0.05, min(0.85, v + noise[k])) for k, v in optimal.items()}
    total = sum(noisy.values())
    weights = {k: v / total for k, v in noisy.items()}
    quality = 1.0 - sum(abs(weights[k] - optimal[k]) for k in optimal) / 2
    reward = 0.75 + (0.25 * quality)
    return {"id": episode_id, "query": query, "domain": domain, "optimal_weights": weights,
            "reward": round(reward, 3), "ground_truth_response": f"Response about {concept}"}


def generate_dataset(num_episodes: int = 100000, output_path: str = "data/synthetic_episodes/training_100k.jsonl"):
    output_file = Path(__file__).parents[2] / output_path
    output_file.parent.mkdir(parents=True, exist_ok=True)

    episodes, episode_id = [], 0
    for domain, template in DOMAIN_TEMPLATES.items():
        count = int(num_episodes * template["weight"])
        for _ in range(count):
            episodes.append(generate_episode(domain, template, episode_id))
            episode_id += 1

    with open(output_file, 'w') as f:
        for ep in episodes:
            f.write(json.dumps(ep) + '\n')

    print(f"Generated {len(episodes)} episodes -> {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--output", type=str, default="data/synthetic_episodes/training_100k.jsonl")
    parser.add_argument("--cot", action="store_true", help="Generate CoT traces from replay buffer")
    parser.add_argument("--cot-output", type=str, default="data/synthetic_episodes/cot_traces.jsonl")
    parser.add_argument("--cot-min-reward", type=float, default=0.8)
    parser.add_argument("--cot-limit", type=int, default=1000)
    args = parser.parse_args()

    if args.cot:
        extract_cot_traces(
            min_reward=args.cot_min_reward,
            output_path=args.cot_output,
            max_traces=args.cot_limit,
        )
    else:
        generate_dataset(args.episodes, args.output)


# ── Phase 5: Chain-of-Thought trace extraction ──────────────────────────

def extract_cot_traces(
    min_reward: float = 0.8,
    output_path: str = "data/synthetic_episodes/cot_traces.jsonl",
    max_traces: int = 1000,
    db_path: str | None = None,
) -> Path:
    """Extract high-reward episodes from the replay buffer and format as
    chain-of-thought training examples for SFT fine-tuning.

    Each trace includes the original query, the high-reward response, and
    a synthesized reasoning chain that explains the retrieval and fusion
    decisions that led to the answer.
    """
    project_root = Path(__file__).parents[2]
    resolved_db = Path(db_path) if db_path else project_root / "db" / "rlfo_cache.db"
    out_file = project_root / output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not resolved_db.exists():
        print(f"No database at {resolved_db}, generating synthetic CoT traces instead")
        return _generate_synthetic_cot(out_file, max_traces)

    conn = sqlite3.connect(str(resolved_db))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='episodes'")
    if not cursor.fetchone():
        conn.close()
        print("No episodes table found, generating synthetic CoT traces")
        return _generate_synthetic_cot(out_file, max_traces)

    cursor.execute(
        "SELECT query, response, reward, rag_weight, cag_weight, graph_weight, fused_context "
        "FROM episodes WHERE reward >= ? ORDER BY reward DESC LIMIT ?",
        (min_reward, max_traces),
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print(f"No episodes with reward >= {min_reward}, generating synthetic CoT traces")
        return _generate_synthetic_cot(out_file, max_traces)

    traces = []
    for query, response, reward, rag_w, cag_w, graph_w, context in rows:
        # build a reasoning chain from the episode metadata
        weights = {"rag": rag_w or 0.0, "cag": cag_w or 0.0, "graph": graph_w or 0.0}
        dominant = max(weights, key=weights.get)
        web_w = max(0.0, 1.0 - sum(weights.values()))

        chain = (
            f"Step 1: Analyze the query to determine intent and required information.\n"
            f"Step 2: Route retrieval with weights: RAG={rag_w:.2f}, CAG={cag_w:.2f}, "
            f"Graph={graph_w:.2f}, Web={web_w:.2f}. Primary source: {dominant}.\n"
            f"Step 3: Score and filter retrieved chunks using CSWR stability weighting.\n"
            f"Step 4: Fuse context from all sources weighted by RL policy.\n"
            f"Step 5: Generate response grounded in fused context.\n"
            f"Step 6: Self-critique confirmed reward={reward:.2f}."
        )

        traces.append({
            "query": query,
            "reasoning_chain": chain,
            "response": response[:2000] if response else "",
            "reward": reward,
            "weights": {**weights, "web": web_w},
        })

    with open(out_file, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")

    print(f"Extracted {len(traces)} CoT traces (reward >= {min_reward}) -> {out_file}")
    return out_file


def _generate_synthetic_cot(out_file: Path, count: int) -> Path:
    """Fallback: create synthetic CoT traces when no replay data exists."""
    traces = []
    for i in range(min(count, 500)):
        domain = random.choice(list(DOMAIN_TEMPLATES.keys()))
        template = DOMAIN_TEMPLATES[domain]
        query_t = random.choice(template["queries"])
        concept = random.choice(template["concepts"])
        query = query_t.format(
            concept=concept, system=concept, url=concept,
            website=concept, place=concept, thing=concept,
            event=concept, topic=concept, person=concept,
            task=concept, problem=concept, field=concept,
            error=concept, issue=concept,
        )
        opt = template["optimal_sources"]
        dominant = max(opt, key=opt.get)

        chain = (
            f"Step 1: Query about '{concept}' in domain '{domain}'.\n"
            f"Step 2: Optimal retrieval routing: {', '.join(f'{k}={v:.2f}' for k, v in opt.items())}. "
            f"Primary: {dominant}.\n"
            f"Step 3: CSWR filters low-stability chunks.\n"
            f"Step 4: Fuse weighted context.\n"
            f"Step 5: Generate grounded response.\n"
            f"Step 6: Self-critique targets reward >= 0.85."
        )
        traces.append({
            "query": query,
            "reasoning_chain": chain,
            "response": f"Response about {concept} using {dominant} retrieval.",
            "reward": 0.85 + random.uniform(0, 0.15),
            "weights": opt,
        })

    with open(out_file, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")

    print(f"Generated {len(traces)} synthetic CoT traces -> {out_file}")
    return out_file
