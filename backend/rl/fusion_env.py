# Author: Bradley R. Kinnard
# fusion_env.py - Gymnasium env for 2-path (CAG + Graph) fusion weight learning
# Obs: 394 dims (384 embed + 10 features). Action: 2D logits -> softmax.

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from backend.agents.fusion_agent import build_fusion_context
from backend.core.critique import critique
from backend.core.retrievers import retrieve
from backend.core.utils import embed_text
from backend.rl.obs_builder import OBS_DIM, build_observation, project_to_simplex

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FusionEnv(gym.Env):
    """Contextual-bandit environment for 2-path fusion weights (CAG + Graph).

    Every `step()` returns terminal=True, so this is a contextual bandit
    in gym clothing rather than a full-horizon RL problem. Migration from
    CQL to AWAC, CRR, or behaviour cloning is tracked in RELEASES.md
    "Known follow-ups" and out of scope for the 2026-05-21 remediation.

    Each call:
      1. Projects the action onto the [w_cag, w_graph] simplex via
         `project_to_simplex` (same fn the live serving path uses).
      2. Builds the fused context exactly the way the live pipeline does
         via `build_fusion_context()` (CSWR rerank, injection scrub).
      3. Calls the real local generator (Llama 3.1 8B by default) for a
         response unless RLFUSION_ENV_DRY_RUN is set, in which case it
         falls back to the fused context as the "response" so training
         can smoke-test on CPU-only boxes.
      4. Calls `critique()` to produce the reward signal: the same call
         that runs at inference, so training reward and serving reward
         come from the same scorer.
    """
    metadata = {"render_modes": ["human"]}
    NUM_SOURCES = 2

    def __init__(self, config_path: str = "backend/config.yaml") -> None:
        super().__init__()
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.NUM_SOURCES,), dtype=np.float32,
        )
        # 384 embed + 10 features = 394
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32,
        )
        self.current_query: str = ""
        self.query_embedding: Optional[np.ndarray] = None
        self.retrieval_results: Optional[Dict[str, Any]] = None
        self.conversation_history: list[str] = []
        self.config_path = PROJECT_ROOT / config_path
        # dry-run flag for CPU-only/test environments
        self._dry_run = os.environ.get("RLFUSION_ENV_DRY_RUN", "").lower() in ("1", "true")

    def _build_observation(self) -> np.ndarray:
        """Build 394-dim observation. Delegates to the canonical builder."""
        return build_observation(
            self.current_query,
            self.query_embedding,
            self.retrieval_results,
        )

    def _generate_response(self, fused_context: str) -> str:
        """Run the real local generator unless dry-run is active."""
        if self._dry_run:
            return fused_context or "[dry-run: no context]"

        from backend.core.model_router import get_engine
        engine = get_engine()
        try:
            return engine.generate(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. Use the retrieved context "
                            "to answer the user's question accurately. Be concise."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{fused_context}\n\nQuestion: {self.current_query}"
                        ),
                    },
                ],
                temperature=0.2, num_predict=400, num_ctx=4096,
            )
        except Exception as exc:
            logger.warning("FusionEnv generator call failed (%s); using fallback", exc)
            return fused_context or "[generator unavailable]"

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
        # F1.5: train and serve share the same simplex projection.
        self.last_applied_weights = project_to_simplex(np.asarray(action).flatten())
        w_cag, w_graph = self.last_applied_weights.tolist()

        # build context exactly the way the live pipeline does (with CSWR re-rank)
        fused_context = build_fusion_context(
            self.retrieval_results or {"cag": [], "graph": []},
            self.last_applied_weights,
            query=self.current_query,
        )

        response = self._generate_response(fused_context)
        critique_result = critique(self.current_query, fused_context, response)
        reward = critique_result["reward"]

        info = {
            "weights": {"cag": float(w_cag), "graph": float(w_graph)},
            "reward": float(reward),
            "query": self.current_query,
            "response": response,
            "fused_context": fused_context,
            "critique": critique_result,
        }

        observation = self._build_observation()
        return observation, float(reward), True, False, info

    def render(self) -> None:
        print(f"[FusionEnv] Query: {self.current_query}")

    def close(self) -> None:
        pass
