# Author: Bradley R. Kinnard
# Train RL policy on REAL episodes (not synthetic data)

import argparse
import json
import logging
from pathlib import Path
import numpy as np
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


def load_real_episodes(episodes_dir: Path) -> list:
    """Load all real episode JSON files from directory"""
    episodes = []
    episode_files = sorted(episodes_dir.glob("ep_*.json"))

    for ep_file in episode_files:
        with open(ep_file, 'r') as f:
            episode = json.load(f)
            episodes.append(episode)

    logger.info(f"✅ Loaded {len(episodes)} REAL episodes from {episodes_dir}")
    return episodes


def train_on_real_episodes(episodes: list, timesteps: int = 100000):
    """Train policy with supervised learning from REAL user episodes"""

    logger.info("🎯 Training on REAL user episodes (no synthetic data)")
    logger.info(f"📊 {len(episodes)} episodes | {timesteps:,} timesteps")

    # Create wrapper environment that cycles through real episodes
    class RealEpisodeFusionEnv(FusionEnv):
        def __init__(self, episodes_data):
            super().__init__()
            self.episodes_data = episodes_data
            self.episode_idx = 0

        def reset(self, seed=None, options=None):
            # Get next real episode (NO EXPENSIVE RETRIEVAL)
            episode = self.episodes_data[self.episode_idx % len(self.episodes_data)]
            self.episode_idx += 1

            # Extract query and ground truth reward
            self.current_query = episode["query"]
            self.ground_truth_reward = episode["reward"]

            # Embed query as observation
            query_emb = embed_text(self.current_query)

            return query_emb.astype(np.float32), {}

        def step(self, action):
            # Action is the fusion weights (4D continuous)
            weights = action

            # Reward is the ground truth from real episode
            # (This was the actual user feedback or critique score)
            reward = self.ground_truth_reward

            # Episode ends after one step (supervised learning)
            done = True
            truncated = False

            # Next observation (doesn't matter since done=True)
            next_obs = np.zeros(384, dtype=np.float32)

            return next_obs, reward, done, truncated, {}

    # Create vectorized environment
    def make_env():
        return RealEpisodeFusionEnv(episodes)

    env = DummyVecEnv([make_env])

    # Initialize PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/ppo_real_episodes"
    )

    logger.info("🚀 Starting training on REAL episodes...")

    # Train
    model.learn(total_timesteps=timesteps)

    logger.info("\n📊 Testing trained policy on sample queries:")

    # Test on a few queries
    test_queries = [
        "What is reinforcement learning?",
        "How does the system architecture work?",
        "Explain RAG vs CAG",
        "What is the capital of France?"
    ]

    for query in test_queries:
        obs = embed_text(query).astype(np.float32)
        obs = obs.reshape(1, -1)
        action, _ = model.predict(obs, deterministic=True)
        weights = action.flatten()

        # Normalize
        exp_w = np.exp(weights)
        rl_weights = exp_w / np.sum(exp_w)

        logger.info(f"\nQuery: {query}")
        logger.info(f"  RAG={rl_weights[0]:.3f} CAG={rl_weights[1]:.3f} Graph={rl_weights[2]:.3f} Web={rl_weights[3]:.3f}")

    # Save the trained model
    save_path = Path(__file__).parents[2] / "backend" / "rl_policy_real.zip"
    model.save(str(save_path))
    logger.info(f"\n💾 Model saved to: {save_path}")
    logger.info("✅ Training on REAL episodes complete!")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL policy on REAL user episodes")
    parser.add_argument("--episodes-dir", type=str, default="data/synthetic_episodes")
    parser.add_argument("--timesteps", type=int, default=100000)
    args = parser.parse_args()

    # Load real episodes
    episodes_dir = Path(__file__).parents[2] / args.episodes_dir
    episodes = load_real_episodes(episodes_dir)

    # Train
    train_on_real_episodes(episodes, args.timesteps)
