"""
Gymnasium environment for training reinforcement learning agents on multi-source fusion.

This environment enables agents to learn optimal weighting strategies for combining
retrieval results from multiple sources (RAG, CAG, Graph, Web) in a retrieval-augmented
generation pipeline. Rewards are based on response quality critique scores.

Author: Bradley R. Kinnard
License: MIT
"""

import gymnasium as gym
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from backend.core.critique import critique
from backend.core.fusion import fuse_context
from backend.core.retrievers import retrieve
from backend.core.utils import embed_text


class FusionEnv(gym.Env):
    """
    Single-step episodic environment for learning multi-source fusion weights.

    The agent observes query embeddings with conversation context and outputs
    weights for four retrieval sources. Rewards are computed via automated
    critique of the fused response quality.

    Attributes:
        NUM_SOURCES: Number of retrieval sources (RAG, CAG, Graph, Web)
        action_space: Continuous weights in [-1, 1] for each source
        observation_space: 3-turn query context (384 dims × 3)
    """

    metadata = {"render_modes": ["human"]}
    NUM_SOURCES = 4

    def __init__(self, config_path: str = "config.yaml") -> None:
        """
        Initialize the fusion environment.

        Args:
            config_path: Path to configuration file (relative to project root)
        """
        super().__init__()

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.NUM_SOURCES,),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(384 * 3,),
            dtype=np.float32
        )

        self.current_query: str = ""
        self.query_embedding: Optional[np.ndarray] = None
        self.retrieval_results: Optional[Dict[str, Any]] = None
        self.conversation_history: list[str] = []
        self.config_path = Path(__file__).parents[2] / config_path

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed for reproducibility
            options: Optional dict containing 'query' key for custom query

        Returns:
            Tuple of (observation, info_dict)
        """
        super().reset(seed=seed)

        query = options.get("query") if options else "What is RLFusion Orchestrator?"
        self.current_query = query

        self.conversation_history.append(query)
        if len(self.conversation_history) > 3:
            self.conversation_history = self.conversation_history[-3:]

        self.query_embedding = embed_text(query)
        self.retrieval_results = retrieve(query)

        recent_queries = self.conversation_history[-3:]
        padded_queries = recent_queries + [""] * (3 - len(recent_queries))
        observation = np.concatenate([embed_text(q) for q in padded_queries]).astype(np.float32)

        return observation, {}

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step of the environment.

        Args:
            action: Array of weights in [-1, 1] for each source

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        weights = np.clip(action, -1.0, 1.0)
        weights = (weights + 1.0) / 2.0
        weights_sum = weights.sum()

        num_sources = self.action_space.shape[0]
        if weights_sum > 0:
            weights = weights / weights_sum
        else:
            weights = np.full(num_sources, 1.0 / num_sources)

        weight_rag, weight_cag, weight_graph, weight_web = weights.tolist()

        fusion_output = fuse_context(
            self.current_query,
            self.retrieval_results["rag"],
            self.retrieval_results["cag"],
            self.retrieval_results["graph"]
        )

        fused_context = fusion_output["context"]

        response = getattr(self, "ground_truth_response", "Default response")

        critique_result = critique(self.current_query, fused_context, response)
        reward = critique_result["reward"]

        terminated = True
        truncated = False

        info = {
            "weights": {
                "rag": float(weight_rag),
                "cag": float(weight_cag),
                "graph": float(weight_graph),
                "web": float(weight_web)
            },
            "reward": float(reward),
            "query": self.current_query
        }

        recent_queries = self.conversation_history[-3:]
        padded_queries = recent_queries + [""] * (3 - len(recent_queries))
        observation = np.concatenate([embed_text(q) for q in padded_queries]).astype(np.float32)

        return observation, float(reward), terminated, truncated, info

    def render(self) -> None:
        """
        Render the current environment state.

        Displays the current query being processed.
        """
        print(f"[FusionEnv] Query: {self.current_query}")

    def close(self) -> None:
        """
        Clean up environment resources.

        This environment has no persistent resources to clean up.
        """
        pass
