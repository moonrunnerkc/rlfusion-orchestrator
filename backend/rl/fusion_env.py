# Author: Bradley R. Kinnard
# fusion_env.py - Gymnasium env for 2-path (CAG + Graph) fusion weight learning
# Obs: 394 dims (384 embed + 10 features). Action: 2D logits -> softmax.
# Upgraded from 4-path (RAG/CAG/Graph/Web) in Step 5 of production upgrade.

import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from backend.core.critique import critique
from backend.core.fusion import fuse_context
from backend.core.retrievers import retrieve
from backend.core.utils import embed_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FusionEnv(gym.Env):
    """Gymnasium environment for learning 2-path fusion weights (CAG + Graph)."""
    metadata = {"render_modes": ["human"]}
    NUM_SOURCES = 2

    def __init__(self, config_path: str = "backend/config.yaml") -> None:
        super().__init__()
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.NUM_SOURCES,), dtype=np.float32,
        )
        # 384 embed + 10 features = 394
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(394,), dtype=np.float32,
        )
        self.current_query: str = ""
        self.query_embedding: Optional[np.ndarray] = None
        self.retrieval_results: Optional[Dict[str, Any]] = None
        self.conversation_history: list[str] = []
        self.config_path = PROJECT_ROOT / config_path

    def _build_observation(self) -> np.ndarray:
        """Build 394-dim observation from embedding + retrieval features."""
        embed = self.query_embedding

        # CAG features
        cag_results = self.retrieval_results.get("cag", [])
        cache_hit = 1.0 if cag_results else 0.0
        cag_top_score = float(cag_results[0].get("score", 0)) if cag_results else 0.0

        # Graph features
        graph_results = self.retrieval_results.get("graph", [])
        graph_density = len(graph_results) / 10.0
        graph_scores = [g.get("score", 0) for g in graph_results[:3]]
        graph_top3 = graph_scores + [0.0] * (3 - len(graph_scores))

        # Query features
        query_len = len(self.current_query.split()) / 50.0
        q_lower = self.current_query.lower()
        query_type = [0.0] * 3
        if any(w in q_lower for w in ["what is", "who is", "define", "explain"]):
            query_type[0] = 1.0  # factual, favors cache
        elif any(w in q_lower for w in ["how does", "architecture", "design", "relationship"]):
            query_type[1] = 1.0  # relational, favors graph
        else:
            query_type[2] = 1.0  # general

        features = np.array(
            [cache_hit, cag_top_score, graph_density]
            + graph_top3
            + [query_len]
            + query_type,
            dtype=np.float32,
        )
        return np.concatenate([embed, features]).astype(np.float32)

    def _generate_response(self, context: str) -> str:
        """Synthetic response for offline training episodes."""
        base = context[:400] if context and len(context) > 400 else (
            context or "Based on available information about RLFO..."
        )
        proactive = (
            "\n\n## Next Steps\n\n"
            "Based on your query, you might also want to consider:\n"
            "- Review the fusion weight configuration for your use case\n"
            "- Check the retrieval source priorities in your config\n"
            "- Monitor critique scores to track response quality over time\n"
        )
        return f"{base}{proactive}"

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        query = options.get("query") if options else "What is RLFusion Orchestrator?"
        self.current_query = query
        self.conversation_history.append(query)
        if len(self.conversation_history) > 3:
            self.conversation_history = self.conversation_history[-3:]

        self.query_embedding = embed_text(query)
        self.retrieval_results = retrieve(query)

        observation = self._build_observation()
        return observation, {}

    def step(
        self, action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        weights = torch.softmax(torch.tensor(action), dim=0)
        weights = torch.clamp(weights, min=0.05)
        weights = weights / weights.sum()
        self.last_applied_weights = weights.numpy()
        w_cag, w_graph = self.last_applied_weights.tolist()

        fusion_output = fuse_context(
            self.current_query,
            [],  # no RAG results
            self.retrieval_results.get("cag", []),
            self.retrieval_results.get("graph", []),
        )
        fused_context = fusion_output["context"]
        response = self._generate_response(fused_context)
        critique_result = critique(self.current_query, fused_context, response)
        reward = critique_result["reward"]

        info = {
            "weights": {"cag": float(w_cag), "graph": float(w_graph)},
            "reward": float(reward),
            "query": self.current_query,
            "response": response,
            "critique": critique_result,
        }

        observation = self._build_observation()
        return observation, float(reward), True, False, info

    def render(self) -> None:
        print(f"[FusionEnv] Query: {self.current_query}")

    def close(self) -> None:
        pass
