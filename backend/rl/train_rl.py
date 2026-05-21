# Author: Bradley R. Kinnard
# train_rl.py - Offline CQL training on the 2-path episodes table.
#
# Reads the live `episodes` table (cag_weight, graph_weight, reward),
# turns each row into an (obs, action, reward, terminal) tuple, and
# trains a Conservative Q-Learning policy via d3rlpy.
#
# Reward is whatever the live critique() call produced at chat time, so
# training reward and serving reward come from the same scorer. This
# replaces the v1 path where the env produced "context[:400] + boilerplate"
# and critiqued that, which made the policy chase a scorer it never saw
# at inference.

import argparse
import logging
import os
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from d3rlpy.algos import CQLConfig
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.preprocessing import MinMaxActionScaler

from backend.rl.fusion_env import FusionEnv
from backend.core.utils import embed_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _device() -> str:
    pref = os.environ.get("RLFUSION_DEVICE", "cpu")
    return "cuda" if pref == "cuda" and torch.cuda.is_available() else "cpu"


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_episodes_two_path(db_path: Path) -> MDPDataset | None:
    """Load (obs, action, reward) tuples from the 2-path episodes table."""
    if not db_path.exists():
        logger.warning("DB missing at %s", db_path)
        return None

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cols = [r[1] for r in cur.execute("PRAGMA table_info(episodes)").fetchall()]
    has_rag = "rag_weight" in cols

    if has_rag:
        # legacy schema present; project to 2-path
        rows = cur.execute(
            "SELECT query, cag_weight, graph_weight, reward "
            "FROM episodes WHERE query != '' AND reward IS NOT NULL"
        ).fetchall()
    else:
        rows = cur.execute(
            "SELECT query, cag_weight, graph_weight, reward "
            "FROM episodes WHERE query != '' AND reward IS NOT NULL"
        ).fetchall()
    conn.close()

    if not rows:
        logger.warning("No episodes in %s", db_path)
        return None

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminals: list[float] = []

    for query, cag_w, graph_w, reward in rows:
        if cag_w is None and graph_w is None:
            continue
        obs = embed_text(query)
        act = np.array(
            [float(cag_w or 0.0), float(graph_w or 0.0)],
            dtype=np.float32,
        )
        observations.append(obs)
        actions.append(act)
        rewards.append(float(reward))
        terminals.append(1.0)

    if not observations:
        return None

    logger.info("Loaded %d 2-path episodes", len(observations))
    return MDPDataset(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=np.float32),
    )


def train_cql_offline(
    epochs: int = 50,
    steps_per_epoch: int = 1000,
    patience: int = 3,
    eval_episodes: int = 10,
) -> Dict[str, Any]:
    """Train CQL on 2-path episodes. Returns summary dict."""
    db_path = PROJECT_ROOT / "db" / "rlfo_cache.db"
    dataset = _load_episodes_two_path(db_path)
    if dataset is None:
        logger.error("No dataset; aborting CQL training")
        return {"status": "no_data"}

    device = _device()
    logger.info("CQL training on %d transitions, device=%s", dataset.size(), device)

    cql = CQLConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        conservative_weight=5.0,
        batch_size=256,
        actor_encoder_factory=DefaultEncoderFactory(),
        critic_encoder_factory=DefaultEncoderFactory(),
        action_scaler=MinMaxActionScaler(),
    ).create(device=device)

    out_path = PROJECT_ROOT / "models" / "rl_policy_cql.d3"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = FusionEnv()
    best_reward = -float("inf")
    wait = 0

    for epoch in range(epochs):
        cql.fit(dataset, n_steps=steps_per_epoch, show_progress=False)

        # online eval against the env (real generate + real critique)
        ep_rewards: list[float] = []
        for _ in range(eval_episodes):
            obs, _ = env.reset()
            action = cql.predict(np.array([obs[:384]]))[0]
            _, reward, _, _, _ = env.step(action)
            ep_rewards.append(float(reward))
        val_reward = float(np.mean(ep_rewards))
        logger.info(
            "Epoch %d | val_reward=%.3f (mean of %d evals)",
            epoch + 1, val_reward, eval_episodes,
        )

        if val_reward > best_reward + 0.01:
            best_reward = val_reward
            wait = 0
            cql.save_model(str(out_path))
            logger.info("Saved improved policy to %s", out_path)
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stop at epoch %d (no improvement for %d epochs)",
                            epoch + 1, patience)
                break

    return {
        "status": "ok",
        "best_reward": best_reward,
        "transitions": dataset.size(),
        "checkpoint": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline CQL training (2-path).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seeds(args.seed)
    result = train_cql_offline(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        patience=args.patience,
        eval_episodes=args.eval_episodes,
    )
    logger.info("Training summary: %s", result)


if __name__ == "__main__":
    main()
