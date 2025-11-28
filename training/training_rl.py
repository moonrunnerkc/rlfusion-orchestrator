"""
Reinforcement learning training script for multi-source fusion weight optimization.

This module implements offline RL training using Proximal Policy Optimization (PPO)
to learn optimal fusion weights for combining RAG, CAG, Graph, and Web retrieval sources.
Training uses pre-computed episodes with known rewards to avoid expensive online evaluation.

Author: Bradley R. Kinnard
License: MIT
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from backend.core.utils import embed_text
from backend.rl.fusion_env import FusionEnv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """
    Load system configuration from YAML file.

    Returns:
        Dictionary containing configuration parameters
    """
    config_path = Path(__file__).parent.parent / "backend" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_episodes(data_path: Path) -> List[Dict[str, Any]]:
    """
    Load training episodes from JSON files in specified directory.

    Each episode must contain: query, fused_context, response, and reward.
    Invalid or incomplete episodes are logged and skipped.

    Args:
        data_path: Directory containing episode JSON files

    Returns:
        List of valid episode dictionaries
    """
    episodes = []

    if not data_path.exists():
        logger.warning(f"Data path does not exist: {data_path}")
        return episodes

    json_files = list(data_path.glob("*.json"))
    logger.info(f"Found {len(json_files)} episode files")

    required_keys = ["query", "fused_context", "response", "reward"]

    for episode_file in json_files:
        try:
            with open(episode_file, "r") as f:
                episode = json.load(f)

            if all(key in episode for key in required_keys):
                episodes.append(episode)
            else:
                missing = [k for k in required_keys if k not in episode]
                logger.warning(f"Episode {episode_file.name} missing keys: {missing}")

        except Exception as e:
            logger.error(f"Failed to load {episode_file.name}: {e}")

    logger.info(f"Loaded {len(episodes)} valid episodes")
    return episodes


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for all libraries to ensure reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Set random seed to {seed} for reproducibility")


def build_observation(queries: List[str]) -> np.ndarray:
    """
    Build observation vector from conversation history.

    Creates 3-turn context by concatenating embeddings of recent queries.
    Pads with empty strings if fewer than 3 queries available.

    Args:
        queries: List of recent query strings (up to 3)

    Returns:
        Flattened observation array of shape (1152,) = 384 * 3
    """
    recent_queries = queries[-3:] if len(queries) >= 3 else queries
    padded_queries = recent_queries + [""] * (3 - len(recent_queries))

    embeddings = [embed_text(q) for q in padded_queries]
    observation = np.concatenate(embeddings)

    return observation.reshape(1, -1)


def normalize_weights(action: np.ndarray) -> np.ndarray:
    """
    Normalize action weights to valid probability distribution.

    Applies softmax transformation to ensure weights sum to 1.0
    and are all non-negative.

    Args:
        action: Raw action weights from policy

    Returns:
        Normalized weight array summing to 1.0
    """
    weights = action.flatten() if isinstance(action, np.ndarray) else np.array(action)
    exp_weights = np.exp(weights)
    normalized = exp_weights / np.sum(exp_weights)

    return normalized


def train_policy(
    episodes: List[Dict[str, Any]],
    config: Dict[str, Any],
    num_episodes: int,
    policy_path: Path,
    log_dir: Path
) -> None:
    """
    Train PPO policy using offline episodes with pre-computed rewards.

    Performs weight exploration and tracks statistics without online
    environment evaluation. Policy is saved periodically and at completion.

    Args:
        episodes: List of training episode dictionaries
        config: System configuration
        num_episodes: Number of training episodes to run
        policy_path: Path to save trained policy
        log_dir: Directory for training logs
    """
    logger.info("=" * 60)
    logger.info("OFFLINE TRAINING MODE")
    logger.info("Using pre-computed rewards from synthetic episodes")
    logger.info("No live retrieval or LLM calls will be made")
    logger.info("=" * 60)

    env = FusionEnv()
    vectorized_env = DummyVecEnv([lambda: env])

    if policy_path.exists():
        logger.info(f"Loading existing policy from {policy_path}")
        model = PPO.load(str(policy_path), env=vectorized_env)
    else:
        logger.info("Initializing new PPO policy")
        model = PPO(
            "MlpPolicy",
            vectorized_env,
            policy_kwargs={"net_arch": [256, 256, 128]},
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gae_lambda=0.95,
            gamma=0.99,
            clip_range=0.2,
            verbose=0
        )

    log_dir.mkdir(parents=True, exist_ok=True)
    logger_configured = configure(str(log_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(logger_configured)

    episode_rewards = []
    episode_weights = []
    conversation_history = []

    logger.info(f"Starting training on {len(episodes)} synthetic episodes")

    for episode_idx in range(num_episodes):
        if not episodes:
            logger.warning("No episodes available, cannot train")
            break

        episode = random.choice(episodes)
        query = episode.get("query", "")
        reward = episode.get("reward", 0.0)

        if not query or not query.strip():
            continue

        conversation_history.append(query)
        if len(conversation_history) > 3:
            conversation_history = conversation_history[-3:]

        observation = build_observation(conversation_history)
        action, _ = model.predict(observation, deterministic=False)
        normalized_weights = normalize_weights(action)

        episode_rewards.append(reward)
        episode_weights.append(normalized_weights)

        if (episode_idx + 1) % 100 == 0:
            recent_rewards = episode_rewards[-100:]
            recent_weights = np.array(episode_weights[-100:])

            mean_reward = np.mean(recent_rewards)
            mean_rag = np.mean(recent_weights[:, 0])
            mean_cag = np.mean(recent_weights[:, 1])
            mean_graph = np.mean(recent_weights[:, 2])
            mean_web = np.mean(recent_weights[:, 3]) if recent_weights.shape[1] > 3 else 0.0

            logger.info(
                f"Episode {episode_idx + 1}/{num_episodes} | "
                f"Reward: {mean_reward:.3f} | "
                f"Weights RAG:{mean_rag:.2f} CAG:{mean_cag:.2f} "
                f"Graph:{mean_graph:.2f} Web:{mean_web:.2f}"
            )

    logger.info("Offline sampling complete - saving policy")
    model.save(str(policy_path))
    logger.info(f"Policy saved to {policy_path}")

    if episode_rewards:
        final_mean = np.mean(episode_rewards)
        final_std = np.std(episode_rewards)
        logger.info(f"Training complete. Mean reward: {final_mean:.3f} (std: {final_std:.3f})")

    if episode_weights:
        final_weights = np.mean(episode_weights, axis=0)
        logger.info(
            f"Final weight distribution - RAG: {final_weights[0]:.2f}, "
            f"CAG: {final_weights[1]:.2f}, Graph: {final_weights[2]:.2f}"
        )
        if len(final_weights) > 3:
            logger.info(f"Web: {final_weights[3]:.2f}")


def main() -> None:
    """
    Main training entry point.

    Parses command-line arguments, loads configuration and episodes,
    and initiates policy training.
    """
    parser = argparse.ArgumentParser(
        description="Train RL policy for multi-source fusion weight optimization"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes to run (default: 1000)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to synthetic episodes directory (default: data/synthetic_episodes)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    config = load_config()

    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = Path(__file__).parent.parent / "data" / "synthetic_episodes"

    policy_path = Path(__file__).parent.parent / "rl_policy.zip"
    log_dir = Path(__file__).parent / "logs"

    logger.info("=" * 60)
    logger.info("RL FUSION TRAINING")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Policy path: {policy_path}")
    logger.info(f"Log dir: {log_dir}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("=" * 60)

    set_random_seeds(args.seed)

    episodes = load_episodes(data_path)

    if not episodes:
        logger.error("No valid episodes found. Cannot train.")
        return

    train_policy(episodes, config, args.episodes, policy_path, log_dir)

    logger.info("Training session complete")


if __name__ == "__main__":
    main()
