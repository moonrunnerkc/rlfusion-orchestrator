# Author: Bradley R. Kinnard
"""Online PPO trainer with CQL warm-start for fusion weight learning.

Augments the offline CQL policy with online PPO fine-tuning. The CQL
checkpoint provides a conservative starting point; PPO then adapts via
real interaction feedback. Supports GRPO-style group ranking for
multi-agent coordination (Phase 1 agents evaluated as a group).

Key pieces:
    train_ppo_online()     - main training loop with configurable batches
    generate_cql_demos()   - extract expert demos from CQL for warm-start
    train_grpo()           - group relative policy optimization
    PPOPolicyWrapper       - adapts SB3 PPO for inference (handles dim mismatch)
    AdaptivePolicy         - selects CQL/PPO/DPO based on interaction count
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import ClassVar

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents.base import RLPolicy
from backend.config import cfg
from backend.core.utils import embed_text
from backend.rl.fusion_env import FusionEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config defaults ──────────────────────────────────────────────────────
_rl_cfg = cfg.get("rl", {})
_ppo_cfg = _rl_cfg.get("ppo", {})
_DEFAULT_BATCH = _ppo_cfg.get("batch_size", 50)
_DEFAULT_CLIP = _ppo_cfg.get("clip_range", 0.2)
_DEFAULT_ENT = _ppo_cfg.get("ent_coef", 0.01)
_DEFAULT_GAE = _ppo_cfg.get("gae_lambda", 0.95)
_DEFAULT_EPOCHS = _ppo_cfg.get("n_epochs", 10)
_DEFAULT_LR = float(_rl_cfg.get("learning_rate", 3e-4))
_DEFAULT_GAMMA = float(_rl_cfg.get("gamma", 0.99))
_WARM_START_THRESHOLD = _rl_cfg.get("adaptive_warmup", {}).get("cql_until", 50)


# ── Replay environment (skips live retrieval/critique) ───────────────────

class ReplayFusionEnv(FusionEnv):
    """Replays stored episodes for offline PPO training. No external calls."""

    def __init__(self, episodes: list[dict[str, object]]) -> None:
        super().__init__()
        if not episodes:
            raise ValueError("ReplayFusionEnv needs at least one episode")
        self._episodes = episodes
        self._idx = 0
        self._current_reward = 0.0
        self._optimal_action: np.ndarray | None = None

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        ep = self._episodes[self._idx % len(self._episodes)]
        self._idx += 1
        self.current_query = str(ep.get("query", ""))
        self._current_reward = float(ep.get("reward", 0.5))  # type: ignore[arg-type]

        # build optimal action from stored weights
        weights = ep.get("optimal_weights", ep.get("weights", {}))
        if isinstance(weights, dict):
            self._optimal_action = np.array([
                weights.get("rag", 0.25),
                weights.get("cag", 0.25),
                weights.get("graph", 0.25),
                weights.get("web", 0.25),
            ], dtype=np.float32)
        else:
            self._optimal_action = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)

        # 396-dim obs: 384 embed + 12 feature stub
        emb = embed_text(self.current_query)
        features = np.zeros(12, dtype=np.float32)
        features[6] = len(self.current_query.split()) / 50.0  # query_len slot
        obs = np.concatenate([emb, features]).astype(np.float32)
        return obs, {}

    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        # reward = how close the action is to the optimal weights
        if self._optimal_action is not None:
            dist = float(np.linalg.norm(action - self._optimal_action))
            imitation = 1.0 - min(dist / 2.83, 1.0)
            reward = imitation * self._current_reward
        else:
            reward = self._current_reward

        obs = np.zeros(396, dtype=np.float32)
        return obs, reward, True, False, {"query": self.current_query}


# ── CQL demo generation ─────────────────────────────────────────────────

def generate_cql_demos(
    cql_path: Path,
    queries: list[str] | None = None,
    n_demos: int = 200,
) -> list[dict[str, object]]:
    """Extract expert demonstrations from a trained CQL checkpoint.

    Loads the CQL actor weights, predicts actions for a bank of queries,
    and returns episode dicts suitable for ReplayFusionEnv.
    """
    if not cql_path.exists():
        logger.warning("CQL checkpoint not found at %s, returning empty demos", cql_path)
        return []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(str(cql_path), weights_only=False, map_location=device)
    policy_weights = state_dict.get("policy", state_dict)

    # rebuild the CQL actor (same arch as CQLPolicyWrapper in main.py)
    encoder = nn.Sequential(
        nn.Linear(384, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
    ).to(device)
    mu = nn.Linear(256, 4).to(device)

    try:
        encoder[0].weight.data = policy_weights["_encoder._layers.0.weight"]
        encoder[0].bias.data = policy_weights["_encoder._layers.0.bias"]
        encoder[2].weight.data = policy_weights["_encoder._layers.2.weight"]
        encoder[2].bias.data = policy_weights["_encoder._layers.2.bias"]
        mu.weight.data = policy_weights["_mu.weight"]
        mu.bias.data = policy_weights["_mu.bias"]
    except KeyError as exc:
        logger.error("CQL weight key mismatch: %s", exc)
        return []

    encoder.eval()
    mu.eval()

    if queries is None:
        queries = _default_query_bank()

    demos: list[dict[str, object]] = []
    with torch.no_grad():
        for i in range(min(n_demos, len(queries))):
            q = queries[i % len(queries)]
            emb = embed_text(q).reshape(1, -1)
            obs_t = torch.tensor(emb, dtype=torch.float32, device=device)
            action = torch.clamp(mu(encoder(obs_t)), -1.0, 1.0).cpu().numpy().flatten()

            # convert action logits to weights for the episode
            exp_w = np.exp(action)
            weights = exp_w / exp_w.sum()

            demos.append({
                "query": q,
                "reward": 0.75,  # conservative CQL baseline reward
                "optimal_weights": {
                    "rag": float(weights[0]),
                    "cag": float(weights[1]),
                    "graph": float(weights[2]),
                    "web": float(weights[3]),
                },
            })

    logger.info("Generated %d CQL demos from %s", len(demos), cql_path)
    return demos


def _default_query_bank() -> list[str]:
    """Diverse query bank for demo generation and GRPO training."""
    return [
        "What is reinforcement learning?",
        "How does the CSWR algorithm filter chunks?",
        "Explain transformer attention mechanism",
        "Look up https://github.com/trending",
        "What is the capital of France?",
        "How to implement binary search in Python?",
        "Compare CQL and PPO for offline RL",
        "Latest research on retrieval augmented generation",
        "How do microservices communicate?",
        "Debug a NullPointerException in Java",
        "What is the RLFusion Orchestrator architecture?",
        "How does graph retrieval work?",
        "Explain the actor-critic algorithm",
        "Best practices for API rate limiting",
        "What is Conservative Q-Learning?",
        "How to optimize database queries?",
        "Describe the event sourcing pattern",
        "What is out-of-distribution detection?",
        "How does the safety agent block attacks?",
        "Explain Mahalanobis distance for OOD",
        "What is federated learning?",
        "How to build a knowledge graph from documents?",
        "Explain the Leiden community detection algorithm",
        "What is direct preference optimization?",
        "Compare RAG and CAG retrieval paths",
    ]


# ── PPO training ─────────────────────────────────────────────────────────

def load_episodes_from_db(db_path: Path) -> list[dict[str, object]]:
    """Load training episodes from the SQLite replay buffer."""
    if not db_path.exists():
        logger.warning("DB not found: %s", db_path)
        return []

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='episodes'")
    if not cursor.fetchone():
        conn.close()
        return []

    cursor.execute(
        "SELECT query, reward, rag_weight, cag_weight, graph_weight FROM episodes ORDER BY id"
    )
    rows = cursor.fetchall()
    conn.close()

    episodes: list[dict[str, object]] = []
    for query, reward, rag_w, cag_w, graph_w in rows:
        web_w = max(0.0, 1.0 - (rag_w + cag_w + graph_w))
        episodes.append({
            "query": query,
            "reward": float(reward),
            "optimal_weights": {
                "rag": float(rag_w),
                "cag": float(cag_w),
                "graph": float(graph_w),
                "web": float(web_w),
            },
        })

    logger.info("Loaded %d episodes from %s", len(episodes), db_path)
    return episodes


def create_ppo_model(
    env: gym.Env,
    learning_rate: float = _DEFAULT_LR,
    clip_range: float = _DEFAULT_CLIP,
    ent_coef: float = _DEFAULT_ENT,
    gae_lambda: float = _DEFAULT_GAE,
    n_epochs: int = _DEFAULT_EPOCHS,
    batch_size: int = _DEFAULT_BATCH,
    gamma: float = _DEFAULT_GAMMA,
    device: str = "cpu",
) -> PPO:
    """Create a fresh PPO model with Phase 4 hyperparameters."""
    vec_env = DummyVecEnv([lambda: env])
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=min(2048, max(batch_size * 4, 128)),
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        device=device,
    )
    return model


def evaluate_policy(
    model: PPO,
    env: gym.Env,
    n_eval: int = 10,
) -> dict[str, float]:
    """Run policy over n_eval episodes and return aggregate stats."""
    rewards: list[float] = []
    for _ in range(n_eval):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, _, _, _ = env.step(action)
        rewards.append(float(reward))

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
    }


def train_ppo_online(
    episodes: list[dict[str, object]] | None = None,
    total_timesteps: int = 50000,
    batch_size: int = _DEFAULT_BATCH,
    warm_start: bool = True,
    save_path: Path | None = None,
) -> Path:
    """Train PPO online with optional CQL warm-start.

    If warm_start is True and no episodes are provided, generates CQL demos
    for pre-training. Returns the path to the saved model.
    """
    cql_path = PROJECT_ROOT / "models" / "rl_policy_cql.d3"

    # gather training episodes
    if episodes is None:
        db_path = PROJECT_ROOT / _rl_cfg.get("paths", {}).get("db", "db/rlfo_cache.db")
        if not db_path.is_absolute():
            db_path = PROJECT_ROOT / str(db_path)
        episodes = load_episodes_from_db(db_path)

    if not episodes and warm_start:
        episodes = generate_cql_demos(cql_path)

    if not episodes:
        logger.error("No training episodes available, aborting PPO training")
        raise ValueError("No training episodes. Populate the replay buffer first.")

    # build replay env and model
    env = ReplayFusionEnv(episodes)
    model = create_ppo_model(env, batch_size=batch_size)

    # optionally warm-start from existing PPO checkpoint
    existing_ppo = PROJECT_ROOT / "models" / "rl_policy_ppo.zip"
    if warm_start and existing_ppo.exists():
        logger.info("Warm-starting from existing PPO checkpoint: %s", existing_ppo)
        vec_env = DummyVecEnv([lambda: ReplayFusionEnv(episodes)])
        model = PPO.load(str(existing_ppo), env=vec_env, device="cpu")

    logger.info(
        "PPO training: %d episodes, %d timesteps, batch=%d, clip=%.2f, ent=%.3f",
        len(episodes), total_timesteps, batch_size,
        float(model.clip_range(1.0)) if callable(model.clip_range) else float(model.clip_range),  # type: ignore[arg-type]
        float(model.ent_coef),
    )

    model.learn(total_timesteps=total_timesteps)

    # evaluate
    eval_stats = evaluate_policy(model, env)
    logger.info(
        "PPO eval: mean=%.3f std=%.3f min=%.3f max=%.3f",
        eval_stats["mean_reward"], eval_stats["std_reward"],
        eval_stats["min_reward"], eval_stats["max_reward"],
    )

    # save
    if save_path is None:
        save_path = PROJECT_ROOT / "models" / "rl_policy_ppo.zip"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    logger.info("PPO model saved to %s", save_path)
    return save_path


# ── GRPO: Group Relative Policy Optimization ─────────────────────────────

def rank_candidates_grpo(
    model: PPO,
    env: ReplayFusionEnv,
    n_candidates: int = 4,
) -> list[tuple[np.ndarray, float]]:
    """Generate n_candidates weight vectors and rank by pipeline reward.

    Returns (action, reward) pairs sorted by reward descending.
    Used internally by train_grpo to compute group-relative advantages.
    """
    obs, _ = env.reset()
    candidates: list[tuple[np.ndarray, float]] = []

    for _ in range(n_candidates):
        action, _ = model.predict(obs, deterministic=False)
        _, reward, _, _, _ = env.step(action)
        candidates.append((action.copy(), float(reward)))
        obs, _ = env.reset()

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def train_grpo(
    model: PPO,
    episodes: list[dict[str, object]],
    group_size: int = 4,
    n_iterations: int = 100,
    learning_rate: float = 1e-4,
) -> PPO:
    """Group Relative Policy Optimization for multi-agent fusion.

    For each query, generates group_size candidate weight vectors,
    evaluates them, and updates the policy using group-relative
    advantages. This prevents any single agent from dominating
    the fusion at the expense of overall quality.
    """
    if not episodes:
        raise ValueError("GRPO needs episodes to train on")

    env = ReplayFusionEnv(episodes)
    policy_net = model.policy
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

    logger.info("GRPO: %d iterations, group_size=%d, lr=%.1e", n_iterations, group_size, learning_rate)

    for iteration in range(n_iterations):
        total_loss = 0.0
        n_updates = 0

        for ep in episodes:
            obs, _ = env.reset()
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            # generate a group of candidate actions
            actions: list[np.ndarray] = []
            log_probs: list[torch.Tensor] = []
            rewards: list[float] = []

            for _ in range(group_size):
                dist = policy_net.get_distribution(obs_tensor)
                sample = dist.sample()
                lp = dist.log_prob(sample)
                actions.append(sample.detach().cpu().numpy().flatten())
                log_probs.append(lp)

                _, reward, _, _, _ = env.step(actions[-1])
                rewards.append(float(reward))
                obs, _ = env.reset()
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            # group-relative advantage (GRPO core idea)
            mean_r = float(np.mean(rewards))
            std_r = max(float(np.std(rewards)), 1e-6)
            advantages = [(r - mean_r) / std_r for r in rewards]

            # policy gradient with group advantage
            loss = torch.zeros(1)
            for lp, adv in zip(log_probs, advantages):
                loss = loss - (lp.sum() * adv)
            loss = loss / group_size

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            n_updates += 1

        avg_loss = total_loss / max(n_updates, 1)
        if (iteration + 1) % 10 == 0:
            logger.info("GRPO iter %d/%d | avg_loss=%.4f", iteration + 1, n_iterations, avg_loss)

    logger.info("GRPO training complete")
    return model


# ── Policy wrappers for inference ────────────────────────────────────────

class PPOPolicyWrapper:
    """Adapts SB3 PPO for the RLPolicy protocol used by FusionAgent.

    Handles dimension mismatch: FusionEnv trains on 396 dims, but
    compute_rl_weights passes 384-dim embeddings at inference. Pads
    with zeros if needed.
    """
    _NAME: ClassVar[str] = "ppo"

    def __init__(self, model: PPO) -> None:
        self._model = model
        obs_shape = model.observation_space.shape  # type: ignore[union-attr]
        self._expected_dims: int = int(obs_shape[0])  # type: ignore[index]

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = np.array(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        actual_dims = obs.shape[-1]
        if actual_dims < self._expected_dims:
            pad = np.zeros((obs.shape[0], self._expected_dims - actual_dims), dtype=np.float32)
            obs = np.concatenate([obs, pad], axis=-1)
        elif actual_dims > self._expected_dims:
            obs = obs[..., :self._expected_dims]

        action, _ = self._model.predict(obs, deterministic=True)
        return action


class AdaptivePolicy:
    """Selects between CQL, PPO, and DPO policies based on interaction count.

    Implements the RLPolicy protocol. Thresholds are configurable via
    config.yaml under rl.adaptive_warmup. The default progression:
        0-50 interactions:   CQL (conservative, stable)
        50-500 interactions: PPO (if available, else CQL)
        500+ interactions:   DPO (if available, else PPO, else CQL)
    """

    def __init__(
        self,
        cql_policy: RLPolicy | None = None,
        ppo_policy: PPOPolicyWrapper | None = None,
        dpo_policy: RLPolicy | None = None,
        cql_threshold: int = _WARM_START_THRESHOLD,
        dpo_threshold: int = 500,
    ) -> None:
        self._cql = cql_policy
        self._ppo = ppo_policy
        self._dpo = dpo_policy
        self._cql_threshold = cql_threshold
        self._dpo_threshold = dpo_threshold
        self._interaction_count = 0

    @property
    def interaction_count(self) -> int:
        return self._interaction_count

    @property
    def active_policy_name(self) -> str:
        """Which policy is currently active based on interaction count."""
        if self._interaction_count >= self._dpo_threshold and self._dpo is not None:
            return "dpo"
        if self._interaction_count >= self._cql_threshold and self._ppo is not None:
            return "ppo"
        if self._cql is not None:
            return "cql"
        if self._ppo is not None:
            return "ppo"
        return "fallback"

    def predict(self, obs: np.ndarray) -> np.ndarray:
        self._interaction_count += 1
        name = self.active_policy_name

        if name == "dpo" and self._dpo is not None:
            return self._dpo.predict(obs)
        if name == "ppo" and self._ppo is not None:
            return self._ppo.predict(obs)
        if self._cql is not None:
            return self._cql.predict(obs)

        # absolute fallback: uniform weights
        batch = obs.shape[0] if obs.ndim > 1 else 1
        return np.full((batch, 4), 0.25, dtype=np.float32)

    def reset_count(self) -> None:
        """Reset interaction counter (e.g., after retraining)."""
        self._interaction_count = 0


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Online PPO training for fusion weights")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total training timesteps")
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH, help="PPO batch size")
    parser.add_argument("--no-warm-start", action="store_true", help="Skip CQL warm-start")
    parser.add_argument("--grpo", action="store_true", help="Use GRPO group ranking mode")
    parser.add_argument("--grpo-group-size", type=int, default=4, help="GRPO candidate group size")
    parser.add_argument("--grpo-iterations", type=int, default=100, help="GRPO training iterations")
    parser.add_argument("--db", type=str, default="db/rlfo_cache.db", help="Episode database path")
    parser.add_argument("--save", type=str, default=None, help="Model save path")
    args = parser.parse_args()

    db_path = PROJECT_ROOT / args.db
    episodes = load_episodes_from_db(db_path) if db_path.exists() else []

    save_path = Path(args.save) if args.save else PROJECT_ROOT / "models" / "rl_policy_ppo.zip"

    if args.grpo:
        if not episodes:
            cql_path = PROJECT_ROOT / "models" / "rl_policy_cql.d3"
            episodes = generate_cql_demos(cql_path)
        if not episodes:
            logger.error("No episodes for GRPO training")
            sys.exit(1)

        env = ReplayFusionEnv(episodes)
        model = create_ppo_model(env, batch_size=args.batch_size)
        model = train_grpo(
            model, episodes,
            group_size=args.grpo_group_size,
            n_iterations=args.grpo_iterations,
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        logger.info("GRPO model saved to %s", save_path)
    else:
        train_ppo_online(
            episodes=episodes if episodes else None,
            total_timesteps=args.timesteps,
            batch_size=args.batch_size,
            warm_start=not args.no_warm_start,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
