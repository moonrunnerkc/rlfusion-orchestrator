# Author: Bradley R. Kinnard
# generate_training_data.py - Synthetic training episodes for 4D fusion policy

import json
import random
import argparse
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
    args = parser.parse_args()
    generate_dataset(args.episodes, args.output)
