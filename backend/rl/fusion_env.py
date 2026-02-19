# Author: Bradley R. Kinnard
# fusion_env.py - Gymnasium env for multi-source fusion weight learning
# Originally built for personal offline use, now open-sourced for public benefit.
# Obs: 396 dims (384 embed + 12 features). Action: 4D logits -> softmax+clip(0.05)

import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from backend.core.critique import critique
from backend.core.fusion import fuse_context
from backend.core.retrievers import retrieve
from backend.core.utils import embed_text

# Project root for config access
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FusionEnv(gym.Env):
    """Gymnasium environment for learning retrieval fusion weights."""
    metadata = {"render_modes": ["human"]}
    NUM_SOURCES = 4

    def __init__(self, config_path: str = "backend/config.yaml") -> None:
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.NUM_SOURCES,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(396,), dtype=np.float32)
        self.current_query: str = ""
        self.query_embedding: Optional[np.ndarray] = None
        self.retrieval_results: Optional[Dict[str, Any]] = None
        self.conversation_history: list[str] = []
        self.config_path = PROJECT_ROOT / config_path

    def _generate_response(self, context: str) -> str:
        base = context[:400] if context and len(context) > 400 else (context or "Based on available information about RLFO...")
        proactive = """

## Next Steps

Based on your query, you might also want to consider:
• Review the fusion weight configuration for your use case
• Check the retrieval source priorities in your config
• Monitor critique scores to track response quality over time

## Follow-up Questions

You might find these related topics helpful:
• How does the RL policy adapt to user feedback?
• What are the default weight distributions for each source?
"""
        return f"{base}{proactive}"

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        query = options.get("query") if options else "What is RLFusion Orchestrator?"
        self.current_query = query
        self.conversation_history.append(query)
        if len(self.conversation_history) > 3:
            self.conversation_history = self.conversation_history[-3:]

        self.query_embedding = embed_text(query)
        self.retrieval_results = retrieve(query)

        embed = self.query_embedding
        rag_scores = [r.get("score", 0) for r in self.retrieval_results.get("rag", [])[:3]]
        top3_sim = rag_scores + [0.0] * (3 - len(rag_scores))
        cswr = [r.get("csw_score", r.get("score", 0)) for r in self.retrieval_results.get("rag", [])[:3]]
        avg_cswr = [np.mean(cswr) if cswr else 0.0]
        cache_hit = [1.0 if self.retrieval_results.get("cag", []) else 0.0]
        graph_degree = [len(self.retrieval_results.get("graph", [])) / 10.0]
        query_len = [len(query.split()) / 50.0]

        q_lower = query.lower()
        query_type = [0.0] * 5
        if any(w in q_lower for w in ["what is", "who is", "when", "where"]):
            query_type[0] = 1.0
        elif any(w in q_lower for w in ["how to", "how do", "steps"]):
            query_type[1] = 1.0
        elif any(w in q_lower for w in ["why", "explain", "concept"]):
            query_type[2] = 1.0
        elif any(w in q_lower for w in ["compare", "difference", "vs"]):
            query_type[3] = 1.0
        else:
            query_type[4] = 1.0

        features = np.array(top3_sim + avg_cswr + cache_hit + graph_degree + query_len + query_type, dtype=np.float32)
        observation = np.concatenate([embed, features]).astype(np.float32)
        return observation, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        weights = torch.softmax(torch.tensor(action), dim=0)
        weights = torch.clamp(weights, min=0.05)
        weights = weights / weights.sum()
        self.last_applied_weights = weights.numpy()
        w_rag, w_cag, w_graph, w_web = self.last_applied_weights.tolist()

        fusion_output = fuse_context(
            self.current_query,
            self.retrieval_results["rag"],
            self.retrieval_results["cag"],
            self.retrieval_results["graph"]
        )
        fused_context = fusion_output["context"]
        response = self._generate_response(fused_context)
        critique_result = critique(self.current_query, fused_context, response)
        reward = critique_result["reward"]

        info = {
            "weights": {"rag": float(w_rag), "cag": float(w_cag), "graph": float(w_graph), "web": float(w_web)},
            "reward": float(reward), "query": self.current_query, "response": response, "critique": critique_result,
        }

        # Rebuild a 396-dim observation matching reset() shape:
        # 384 embed + 3 top_sim + 1 avg_cswr + 1 cache_hit + 1 graph_degree + 1 query_len + 5 query_type
        embed = self.query_embedding
        rag_scores = [r.get("score", 0) for r in self.retrieval_results.get("rag", [])[:3]]
        top3_sim = rag_scores + [0.0] * (3 - len(rag_scores))
        cswr = [r.get("csw_score", r.get("score", 0)) for r in self.retrieval_results.get("rag", [])[:3]]
        avg_cswr = [float(np.mean(cswr)) if cswr else 0.0]
        cache_hit = [1.0 if self.retrieval_results.get("cag", []) else 0.0]
        graph_degree = [len(self.retrieval_results.get("graph", [])) / 10.0]
        query_len = [len(self.current_query.split()) / 50.0]

        q_lower = self.current_query.lower()
        query_type = [0.0] * 5
        if any(w in q_lower for w in ["what is", "who is", "when", "where"]):
            query_type[0] = 1.0
        elif any(w in q_lower for w in ["how to", "how do", "steps"]):
            query_type[1] = 1.0
        elif any(w in q_lower for w in ["why", "explain", "concept"]):
            query_type[2] = 1.0
        elif any(w in q_lower for w in ["compare", "difference", "vs"]):
            query_type[3] = 1.0
        else:
            query_type[4] = 1.0

        features = np.array(top3_sim + avg_cswr + cache_hit + graph_degree + query_len + query_type, dtype=np.float32)
        observation = np.concatenate([embed, features]).astype(np.float32)
        return observation, float(reward), True, False, info

    def render(self) -> None:
        print(f"[FusionEnv] Query: {self.current_query}")

    def close(self) -> None:
        pass
