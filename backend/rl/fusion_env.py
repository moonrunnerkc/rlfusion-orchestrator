# backend/rl/fusion_env.py
# Fusion RL environment — single-step, reward-from-critique
# Brad Kinnard — Nov 2025

import gymnasium as gym
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import torch

# Import from backend.core modules
from backend.core.utils import embed_text
from backend.core.retrievers import retrieve
from backend.core.fusion import fuse_context
from backend.core.critique import critique


class FusionEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config_path: str = "config.yaml"):
        super().__init__()

        # Action = 3 weights in [-1, 1] → later normalized to [0,1] and softmaxed
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation = 384-dim query embedding (all-MiniLM-L6-v2)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(384,), dtype=np.float32
        )

        self.current_query: str = ""
        self.query_embedding: np.ndarray | None = None
        self.retrieval_results: Dict[str, Any] | None = None
        self.config_path = Path(__file__).parents[2] / config_path

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        query = options.get("query") if options else "What is RLFusion Orchestrator?"
        self.current_query = query

        self.query_embedding = embed_text(query)  # (384,) numpy array
        self.retrieval_results = retrieve(query)

        obs = self.query_embedding.astype(np.float32)
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Convert action [-1,1] → [0,1] → softmax to sum to 1.0
        weights = np.clip(action, -1.0, 1.0)
        weights = (weights + 1.0) / 2.0
        total = weights.sum()
        weights = weights / total if total > 0 else np.array([0.33, 0.33, 0.34])

        rag_w, cag_w, graph_w = weights.tolist()

        # Fuse contexts using current weights
        fusion_output = fuse_context(
            self.current_query,
            self.retrieval_results["rag"],
            self.retrieval_results["cag"],
            self.retrieval_results["graph"]
        )

        fused = fusion_output["context"]

        # During training we use stored ground-truth response from dataset
        response = getattr(self, "ground_truth_response", "RLFO response")

        # Real reward from self-critique module
        critique_result = critique(self.current_query, fused, response)
        reward = critique_result["reward"]

        done = True
        truncated = False

        info = {
            "weights": {"rag": float(rag_w), "cag": float(cag_w), "graph": float(graph_w)},
            "reward": float(reward),
            "query": self.current_query
        }

        obs = self.query_embedding.astype(np.float32)
        return obs, float(reward), done, truncated, info

    def render(self):
        print(f"[FusionEnv] Query: {self.current_query}")

    def close(self):
        pass
