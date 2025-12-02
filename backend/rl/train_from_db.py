# Author: Bradley R. Kinnard
# Train RL policy from real episodes stored in SQLite replay buffer

import argparse
import sqlite3
import logging
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))

from backend.rl.fusion_env import FusionEnv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


def load_episodes_from_db(db_path: Path) -> list:
    """Load real episodes from SQLite episodes table"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if episodes table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='episodes'")
    if not cursor.fetchone():
        logger.error("❌ No 'episodes' table found in database!")
        conn.close()
        return []

    # Load all episodes
    cursor.execute("""
        SELECT id, query, reward, rag_weight, cag_weight, graph_weight, fused_context
        FROM episodes
        ORDER BY id
    """)
    rows = cursor.fetchall()
    conn.close()

    episodes = []
    for ep_id, query, reward, rag_w, cag_w, graph_w, fused_context in rows:
        # Reconstruct episode format
        # Note: We need to compute web weight (should be 1 - sum of others, but let's derive it)
        web_w = max(0.0, 1.0 - (rag_w + cag_w + graph_w))

        episodes.append({
            "episode_id": ep_id,
            "query": query,
            "reward": reward,
            "optimal_weights": {
                "rag": rag_w,
                "cag": cag_w,
                "graph": graph_w,
                "web": web_w
            },
            "fused_context": fused_context if fused_context else ""
        })

    logger.info(f"✅ Loaded {len(episodes)} real episodes from database")
    return episodes


def load_episodes_from_db_old(db_path: Path) -> list:
    """OLD: Load real episodes from SQLite replay buffer (DEPRECATED)"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if replay table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='replay'")
    if not cursor.fetchone():
        logger.error("❌ No 'replay' table found in database!")
        conn.close()
        return []

    # Load all episodes
    cursor.execute("SELECT episode_id, state, action, reward FROM replay ORDER BY episode_id")
    rows = cursor.fetchall()
    conn.close()

    episodes = []
    for episode_id, state_blob, action_blob, reward in rows:
        # Deserialize blobs (assuming they're numpy arrays stored as bytes)
        state = np.frombuffer(state_blob, dtype=np.float32)
        action = np.frombuffer(action_blob, dtype=np.float32)

        episodes.append({
            "episode_id": episode_id,
            "state": state.tolist(),
            "action": action.tolist(),
            "reward": float(reward)
        })

    logger.info(f"📊 Loaded {len(episodes)} real episodes from replay buffer")
    return episodes


def train_from_db_episodes(episodes: list, timesteps: int = 100000):
    """Train policy from real database episodes"""

    if not episodes:
        logger.error("❌ No episodes to train on!")
        return None

    logger.info(f"🎯 Starting training on {len(episodes)} real episodes")
    logger.info(f"📊 Training for {timesteps:,} timesteps")

    # Calculate average reward for insight
    avg_reward = np.mean([ep["reward"] for ep in episodes])
    logger.info(f"📈 Average episode reward: {avg_reward:.3f}")

    # Create wrapper environment that replays real episodes (training only - no external calls)
    class DBReplayEnv(FusionEnv):
        def __init__(self, episodes_data):
            super().__init__()
            self.episodes_data = episodes_data
            self.episode_idx = 0
            self._optimal_action = None
            self._optimal_reward = 0.0
            self._current_query = ""

        def reset(self, seed=None, options=None):
            # Get next episode from database (cycle through)
            episode = self.episodes_data[self.episode_idx % len(self.episodes_data)]
            self.episode_idx += 1

            # Set current query without calling retrieve (no external calls during training)
            self._current_query = episode["query"]
            self.current_query = episode["query"]

            # Add to conversation history and keep last 3
            self.conversation_history.append(episode["query"])
            if len(self.conversation_history) > 3:
                self.conversation_history = self.conversation_history[-3:]

            # Build observation from query embeddings (no retrieval needed)
            from backend.core.utils import embed_text
            recent_queries = self.conversation_history[-3:]
            padded = recent_queries + [""] * (3 - len(recent_queries))
            obs = np.concatenate([embed_text(q) for q in padded]).astype(np.float32)

            # Store optimal action and reward for this episode
            # Convert optimal_weights dict to action array [rag, cag, graph, web]
            # Note: Convert [0,1] weights back to [-1,1] action space
            optimal_weights = episode["optimal_weights"]
            normalized_weights = np.array([
                optimal_weights["rag"],
                optimal_weights["cag"],
                optimal_weights["graph"],
                optimal_weights["web"]
            ], dtype=np.float32)

            # Convert from [0,1] to [-1,1] action space
            self._optimal_action = (normalized_weights * 2.0) - 1.0
            self._optimal_reward = episode["reward"]

            return obs, {}

        def step(self, action):
            # Pure imitation learning - no external calls
            # Reward agent for matching optimal weights from real episodes
            action_diff = np.linalg.norm(action - self._optimal_action)

            # Reward inversely proportional to distance from optimal
            # Max diff is ~2.83 (sqrt(4*2) for 4-dim unit vectors)
            imitation_reward = 1.0 - min(action_diff / 2.83, 1.0)

            # Scale by the stored reward from the real episode
            combined_reward = imitation_reward * self._optimal_reward

            # Build next observation (doesn't matter since terminated=True)
            from backend.core.utils import embed_text
            recent_queries = self.conversation_history[-3:]
            padded = recent_queries + [""] * (3 - len(recent_queries))
            obs = np.concatenate([embed_text(q) for q in padded]).astype(np.float32)

            terminated = True  # Single-step episodes
            truncated = False

            info = {
                "imitation_reward": float(imitation_reward),
                "optimal_reward": float(self._optimal_reward),
                "combined_reward": float(combined_reward),
                "optimal_weights": self._optimal_action.tolist(),
                "query": self._current_query
            }

            return obs, combined_reward, terminated, truncated, info

    # Create vectorized environment
    env = DummyVecEnv([lambda: DBReplayEnv(episodes)])

    # Initialize PPO
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=min(2048, len(episodes) * 10),
        batch_size=min(64, len(episodes)),
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        device="cpu"  # Use CPU to avoid GPU issues
    )

    logger.info("🚀 Training PPO policy on real episodes...")
    try:
        model.learn(total_timesteps=timesteps)
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Test the trained policy
    logger.info("\n🧪 Testing trained policy on sample episodes:")

    test_env = DBReplayEnv(episodes)
    for i in range(min(5, len(episodes))):
        obs, _ = test_env.reset()
        action, _ = model.predict(obs, deterministic=True)

        # Convert action [-1,1] to weights [0,1]
        weights = np.clip(action, -1.0, 1.0)
        weights = (weights + 1.0) / 2.0
        total = weights.sum()
        rl_weights = weights / total if total > 0 else np.full(4, 0.25)

        episode = episodes[i]
        optimal = episode["optimal_weights"]

        logger.info(f"\n📊 Episode {episode['episode_id']} | Query: {episode['query'][:60]}...")
        logger.info(f"   Ground truth reward: {episode['reward']:.3f}")
        logger.info(f"   Optimal weights: RAG={optimal['rag']:.3f} CAG={optimal['cag']:.3f} Graph={optimal['graph']:.3f} Web={optimal['web']:.3f}")
        logger.info(f"   Learned weights: RAG={rl_weights[0]:.3f} CAG={rl_weights[1]:.3f} Graph={rl_weights[2]:.3f} Web={rl_weights[3]:.3f}")

    # Save the trained model
    save_path = Path(__file__).parents[2] / "backend" / "rl_policy.zip"
    model.save(str(save_path))
    logger.info(f"\n💾 Model saved to: {save_path}")
    logger.info("✅ Training complete!")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="db/rlfo_cache.db")
    parser.add_argument("--timesteps", type=int, default=100000)
    args = parser.parse_args()

    # Load episodes from database
    db_path = Path(__file__).parents[2] / args.db

    if not db_path.exists():
        logger.error(f"❌ Database not found: {db_path}")
        sys.exit(1)

    episodes = load_episodes_from_db(db_path)

    if episodes:
        # Train on real episodes
        train_from_db_episodes(episodes, args.timesteps)
    else:
        logger.warning("⚠️  No episodes found in database. Use the system first to generate real episodes!")
