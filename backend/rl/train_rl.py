# Author: Bradley R. Kinnard
# backend/rl/train_rl.py
# Offline RL training for fusion weight optimization

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from backend.rl.fusion_env import FusionEnv
from backend.core.utils import embed_text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Load config.yaml from project root."""
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_episodes(data_path: Path) -> List[Dict[str, Any]]:
    """Load all JSON episodes from data directory."""
    episodes = []

    if not data_path.exists():
        logger.warning("Data path does not exist: %s", data_path)
        return episodes

    json_files = list(data_path.glob("*.json"))
    logger.info("Found %d episode files", len(json_files))

    for fpath in json_files:
        try:
            with open(fpath, "r") as f:
                episode = json.load(f)

            required = ["query", "fused_context", "response", "reward"]
            if all(k in episode for k in required):
                episodes.append(episode)
            else:
                logger.warning("Episode missing required keys: %s", fpath.name)

        except Exception as e:
            logger.error("Failed to load episode %s: %s", fpath.name, e)

    logger.info("Loaded %d valid episodes", len(episodes))
    return episodes


def set_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seeded all RNGs with seed=%d", seed)


def train_policy(
    episodes: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    total_episodes: int
):
    """Train PPO policy on synthetic episodes."""

    # init env
    env = FusionEnv()
    vec_env = DummyVecEnv([lambda: env])

    policy_path = Path(__file__).parent.parent.parent / "rl_policy.zip"

    # check for existing policy
    if policy_path.exists():
        logger.info("Loading existing policy from %s", policy_path)
        model = PPO.load(str(policy_path), env=vec_env)
    else:
        logger.info("Initializing new PPO policy")
        model = PPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs={"net_arch": [256, 256, 128]},
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gae_lambda=0.95,
            gamma=0.99,
            clip_range=0.2,
            verbose=0
        )

    # tracking for logging
    episode_rewards = []
    episode_weights = []

    for ep_idx in range(total_episodes):
        if not episodes:
            logger.warning("No episodes available, cannot train")
            break

        # sample an episode
        episode = random.choice(episodes)

        query = episode["query"]
        reward = episode["reward"]

        # embed query as observation
        obs = embed_text(query)
        obs = obs.reshape(1, -1)

        # predict action (fusion weights)
        action, _ = model.predict(obs, deterministic=False)

        # extract weights
        if isinstance(action, np.ndarray):
            weights = action.flatten()
        else:
            weights = np.array(action)

        # softmax to ensure they sum to 1
        exp_w = np.exp(weights)
        normalized = exp_w / np.sum(exp_w)

        episode_rewards.append(reward)
        episode_weights.append(normalized)

        # manually step through env for replay buffer
        vec_env.reset()
        vec_env.step(action)

        # log every 100 episodes
        if (ep_idx + 1) % 100 == 0:
            recent_rewards = episode_rewards[-100:]
            recent_weights = np.array(episode_weights[-100:])

            mean_reward = np.mean(recent_rewards)
            mean_rag = np.mean(recent_weights[:, 0])
            mean_cag = np.mean(recent_weights[:, 1])
            mean_graph = np.mean(recent_weights[:, 2])

            logger.info(
                "Episode %d/%d | Mean reward: %.3f | Weights RAG:%.2f CAG:%.2f Graph:%.2f",
                ep_idx + 1, total_episodes, mean_reward, mean_rag, mean_cag, mean_graph
            )

    # train model on collected experience
    logger.info("Running PPO optimization on collected trajectories")
    model.learn(total_timesteps=total_episodes * 10, reset_num_timesteps=False)

    # save final policy
    model.save(str(policy_path))
    logger.info("Saved trained policy to %s", policy_path)

    # final stats
    if episode_rewards:
        final_mean = np.mean(episode_rewards)
        final_std = np.std(episode_rewards)
        logger.info("Training complete. Final mean reward: %.3f (std: %.3f)", final_mean, final_std)

    if episode_weights:
        final_weights = np.mean(episode_weights, axis=0)
        logger.info(
            "Final weight distribution - RAG: %.2f, CAG: %.2f, Graph: %.2f",
            final_weights[0], final_weights[1], final_weights[2]
        )


def main():
    parser = argparse.ArgumentParser(description="Train RLFO fusion policy")
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to synthetic episodes directory"
    )

    args = parser.parse_args()

    cfg = load_config()

    # determine data path
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = Path(__file__).parent.parent.parent / "data" / "synthetic_episodes"

    logger.info("Starting RLFO training with %d episodes", args.episodes)
    logger.info("Data path: %s", data_path)

    set_seeds(42)

    episodes = load_episodes(data_path)

    if not episodes:
        logger.error("No valid episodes found. Cannot train.")
        return

    train_policy(episodes, cfg, args.episodes)

    logger.info("Training session complete")


if __name__ == "__main__":
    main()
