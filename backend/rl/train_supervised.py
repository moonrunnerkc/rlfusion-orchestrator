# Author: Bradley R. Kinnard
# Supervised training on 100k synthetic episodes with ground truth optimal weights

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from backend.rl.fusion_env import FusionEnv
from backend.core.utils import embed_text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


def load_synthetic_episodes(jsonl_path: Path) -> list:
    """Load synthetic training episodes from JSONL"""
    episodes = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            episodes.append(json.loads(line))
    logger.info(f"Loaded {len(episodes):,} synthetic episodes")
    return episodes


def train_supervised(episodes: list, timesteps: int = 500000):
    """Train policy with supervised learning from synthetic episodes"""

    logger.info("🎯 Starting supervised training on 100k synthetic dataset")
    logger.info(f"📊 Training for {timesteps:,} timesteps")

    # Create wrapper environment that cycles through synthetic episodes
    class SyntheticFusionEnv(FusionEnv):
        def __init__(self, episodes_data):
            super().__init__()
            self.episodes_data = episodes_data
            self.episode_idx = 0

        def reset(self, seed=None, options=None):
            # Get next episode from synthetic data (NO EXPENSIVE RETRIEVAL)
            episode = self.episodes_data[self.episode_idx % len(self.episodes_data)]
            self.episode_idx += 1

            # Store query and optimal weights
            self.current_query = episode['query']
            self.optimal_weights = np.array([
                episode['optimal_weights']['rag'],
                episode['optimal_weights']['cag'],
                episode['optimal_weights']['graph'],
                episode['optimal_weights']['web']
            ])

            # Return observation without calling parent (skip retrieval)
            # FusionEnv expects concatenated embeddings: [context, doc, query]
            # We only have query, so pad with dummy embeddings
            obs = np.concatenate([
                embed_text(""),  # Dummy context
                embed_text(""),  # Dummy doc
                embed_text(self.current_query)  # Actual query
            ])
            info = {"query": self.current_query}
            return obs, info

        def step(self, action):
            # SKIP EXPENSIVE RETRIEVAL - We already have optimal weights
            # Just compute supervised reward directly without calling parent step()

            # Normalize action to weights [0, 1]
            weights = np.clip(action, -1.0, 1.0)
            weights = (weights + 1.0) / 2.0
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(4) / 4.0

            # Supervised loss: how close to optimal weights
            mse = np.mean((weights - self.optimal_weights) ** 2)
            supervised_reward = 1.0 - mse  # Higher reward = closer to optimal

            # Return dummy observation with proper shape (concatenated embeddings)
            obs = np.concatenate([
                embed_text(""),  # Dummy context
                embed_text(""),  # Dummy doc
                embed_text(self.current_query)  # Actual query
            ])
            done = True  # One step per episode
            truncated = False
            info = {"mse": mse, "optimal_match": 1.0 - mse}

            return obs, supervised_reward, done, truncated, info

    # Create environment with synthetic data
    env = SyntheticFusionEnv(episodes)
    vec_env = DummyVecEnv([lambda: env])

    # Create PPO policy with better hyperparameters for supervised learning
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-3,  # Higher LR for supervised
        n_steps=2048,
        batch_size=256,  # Larger batches
        n_epochs=20,  # More epochs per update
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cpu"  # CPU for Blackwell compatibility
    )

    logger.info("Training policy on synthetic episodes...")
    model.learn(total_timesteps=timesteps)

    # Save trained policy
    policy_path = Path(__file__).parents[2] / "backend" / "rl_policy.zip"
    model.save(str(policy_path))
    logger.info(f"✅ Policy saved to {policy_path}")

    # Evaluate on sample queries
    logger.info("\n📊 Testing trained policy on sample queries:")
    test_queries = [
        "What is reinforcement learning?",  # Should favor RAG
        "How does the system architecture work?",  # Should favor Graph
        "Look up https://github.com",  # Should favor Web
        "What is the capital of France?"  # Should favor CAG
    ]

    for query in test_queries:
        obs = np.concatenate([embed_text(""), embed_text(""), embed_text(query)])
        obs = obs.reshape(1, -1)
        action, _ = model.predict(obs, deterministic=True)
        weights = action.flatten()

        # Normalize
        exp_w = np.exp(weights)
        rl_weights = exp_w / np.sum(exp_w)

        logger.info(f"\nQuery: {query}")
        logger.info(f"  RAG={rl_weights[0]:.3f} CAG={rl_weights[1]:.3f} Graph={rl_weights[2]:.3f} Web={rl_weights[3]:.3f}")

    # Save the trained model
    save_path = Path(__file__).parents[2] / "backend" / "rl_policy.zip"
    model.save(str(save_path))
    logger.info(f"\n💾 Model saved to: {save_path}")
    logger.info("✅ Training complete!")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/synthetic_episodes/training_100k.jsonl")
    parser.add_argument("--timesteps", type=int, default=500000)
    args = parser.parse_args()

    # Load episodes
    data_path = Path(__file__).parents[2] / args.data
    episodes = load_synthetic_episodes(data_path)

    # Train
    train_supervised(episodes, args.timesteps)
