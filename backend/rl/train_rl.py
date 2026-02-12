# Author: Bradley R. Kinnard
# train_rl.py - Offline RL training (CQL) + PPO fine-tuning for fusion weights
# Originally built for personal offline use, now open-sourced for public benefit.

import argparse
import json
import logging
import os
import random
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from d3rlpy.algos import CQLConfig
from d3rlpy.dataset import MDPDataset
from d3rlpy.preprocessing import MinMaxActionScaler
from d3rlpy.models.encoders import DefaultEncoderFactory
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from backend.rl.fusion_env import FusionEnv
from backend.core.utils import embed_text

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(name)s | %(message)s')
logger = logging.getLogger(__name__)

_cfg_path = PROJECT_ROOT / "backend" / "config.yaml"
cfg = yaml.safe_load(open(_cfg_path))
env = FusionEnv()

# Device selection - respects RLFUSION_DEVICE env var or config
_device_cfg = os.environ.get("RLFUSION_DEVICE", cfg.get('embedding', {}).get('device', 'cpu'))
_use_gpu = _device_cfg == 'cuda' and torch.cuda.is_available()
cql = CQLConfig(
    actor_learning_rate=3e-4, critic_learning_rate=3e-4, conservative_weight=5.0,
    batch_size=256, actor_encoder_factory=DefaultEncoderFactory(),
    critic_encoder_factory=DefaultEncoderFactory(), action_scaler=MinMaxActionScaler(),
).create(device="cuda" if _use_gpu else "cpu")


def load_config() -> Dict[str, Any]:
    with open(PROJECT_ROOT / "backend" / "config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_episodes(data_path: Path) -> List[Dict[str, Any]]:
    if not data_path.exists():
        logger.warning("Data path missing: %s", data_path)
        return []
    episodes = []
    required = ["query", "fused_context", "response", "reward"]
    for fpath in data_path.glob("*.json"):
        try:
            ep = json.load(open(fpath))
            if all(k in ep for k in required):
                episodes.append(ep)
        except Exception as e:
            logger.error("Failed loading %s: %s", fpath.name, e)
    logger.info("Loaded %d episodes", len(episodes))
    return episodes


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_replay_buffer():
    """Load training episodes from the database."""
    db_path = PROJECT_ROOT / "db" / "rlfo_cache.db"
    if not db_path.exists():
        return None

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]

    if 'replay' in tables:
        cursor.execute("SELECT state, action, reward, next_state, terminal FROM replay")
    elif 'episodes' in tables:
        cursor.execute("SELECT query, response, reward, rag_weight, cag_weight, graph_weight FROM episodes")
    else:
        conn.close()
        return None

    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return None

    observations, actions, rewards, terminals = [], [], [], []
    for row in rows:
        if len(row) == 5:
            state, action, reward, next_state, terminal = row
            obs = np.frombuffer(state, dtype=np.float32) if isinstance(state, bytes) else np.array(json.loads(state), dtype=np.float32)
            act = np.frombuffer(action, dtype=np.float32) if isinstance(action, bytes) else np.array(json.loads(action), dtype=np.float32)
            term = float(terminal)
        else:
            query, response, reward, rag_w, cag_w, graph_w = row
            obs = embed_text(query)
            act = np.array([rag_w, cag_w, graph_w, 0.0], dtype=np.float32)
            term = 1.0
        observations.append(obs)
        actions.append(act)
        rewards.append(float(reward))
        terminals.append(term)

    return MDPDataset(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=np.float32),
    )


def train_cql_offline():
    dataset = load_replay_buffer()
    if dataset is None:
        logger.info("No dataset, skipping CQL")
        return

    logger.info("CQL training on %d transitions", dataset.size())
    best_reward, patience, wait = -float("inf"), 3, 0

    for epoch in range(50):
        cql.fit(dataset, n_steps=1000, show_progress=True)
        total_reward = 0
        obs, _ = env.reset()
        for _ in range(10):
            action = cql.predict(np.array([obs[:384]]))[0]
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            if done or truncated:
                obs, _ = env.reset()
        val_reward = total_reward / 10
        logger.info("Epoch %d | val_reward: %.3f", epoch + 1, val_reward)

        if val_reward > best_reward + 0.01:
            best_reward, wait = val_reward, 0
            cql.save_model(str(Path(__file__).parent.parent.parent / "models" / "rl_policy_cql.d3"))
        else:
            wait += 1
            if wait >= patience:
                break

    logger.info("CQL complete | best: %.3f", best_reward)


def train_policy(episodes: List[Dict[str, Any]], cfg: Dict[str, Any], total_episodes: int):
    vec_env = DummyVecEnv([lambda: FusionEnv()])
    policy_path = Path(__file__).parent.parent.parent / "rl_policy.zip"

    if policy_path.exists():
        model = PPO.load(str(policy_path), env=vec_env)
    else:
        model = PPO("MlpPolicy", vec_env, policy_kwargs={"net_arch": [256, 256, 128]},
                    learning_rate=3e-4, n_steps=2048, batch_size=64, verbose=0)

    from stable_baselines3.common.logger import configure
    log_dir = Path(__file__).parent.parent.parent / "training" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))

    episode_rewards, episode_weights = [], []

    for ep_idx in range(total_episodes):
        if not episodes:
            break
        episode = random.choice(episodes)
        query = episode.get("query", "")
        reward = episode.get("reward", 0.0)
        if not query.strip():
            continue

        embed = embed_text(query)
        q_lower = query.lower()
        query_type = [0.0] * 5
        if any(w in q_lower for w in ["what is", "who is", "when", "where"]):
            query_type[0] = 1.0
        elif any(w in q_lower for w in ["how to", "how do", "steps"]):
            query_type[1] = 1.0
        elif any(w in q_lower for w in ["why", "explain", "concept"]):
            query_type[2] = 1.0
        elif any(w in q_lower for w in ["compare", "difference", "vs"]):
            query_type[3] = 1.0
        else:
            query_type[4] = 1.0

        features = np.array([0.5, 0.4, 0.3, 0.5, 0.0, 0.3, len(query.split()) / 50.0] + query_type, dtype=np.float32)
        obs = np.concatenate([embed, features]).reshape(1, -1)
        action, _ = model.predict(obs, deterministic=False)
        weights = np.exp(action.flatten())
        normalized = weights / weights.sum()

        episode_rewards.append(reward)
        episode_weights.append(normalized)

        if (ep_idx + 1) % 100 == 0:
            recent = np.array(episode_weights[-100:])
            logger.info("Ep %d/%d | reward: %.3f | RAG:%.2f CAG:%.2f Graph:%.2f",
                        ep_idx + 1, total_episodes, np.mean(episode_rewards[-100:]),
                        recent[:, 0].mean(), recent[:, 1].mean(), recent[:, 2].mean())

    model.save(str(policy_path))
    if episode_rewards:
        logger.info("Done | mean: %.3f std: %.3f", np.mean(episode_rewards), np.std(episode_rewards))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--data-path", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    data_path = Path(args.data_path) if args.data_path else Path(__file__).parent.parent.parent / "data" / "synthetic_episodes"

    set_seeds(42)
    episodes = load_episodes(data_path)
    if not episodes:
        logger.error("No episodes found")
        return

    train_policy(episodes, cfg, args.episodes)


if __name__ == "__main__":
    main()
