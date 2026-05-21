# Author: Bradley R. Kinnard
# train_rl.py - Offline CQL training on the 2-path episodes table.
#
# Reads the live `episodes` table (cag_weight, graph_weight, reward) plus
# the post-2026-05-21 columns (obs_features, policy_weights, policy_action),
# turns each row into an (obs, action, reward, terminal) tuple, and trains
# a Conservative Q-Learning policy via d3rlpy.
#
# Reward is whatever the live critique() call produced at chat time, so
# training reward and serving reward come from the same scorer.

import argparse
import json
import logging
import os
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from d3rlpy.algos import CQLConfig
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.preprocessing import MinMaxActionScaler

from backend.core.utils import embed_text
from backend.rl.fusion_env import FusionEnv
from backend.rl.obs_builder import OBS_DIM, build_observation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Fixed eval query set so two `--seed N` runs are byte-identical when nothing
# else changes. Add new entries here when the obs space changes.
EVAL_QUERIES: tuple[str, ...] = (
    "What is RLFusion Orchestrator?",
    "How does the CSWR retrieval rerank work?",
    "Define conservative Q-learning in offline RL.",
    "Explain the fusion weight simplex projection.",
    "How are episodes logged to the replay buffer?",
)


def _device() -> str:
    pref = os.environ.get("RLFUSION_DEVICE", "cpu")
    return "cuda" if pref == "cuda" and torch.cuda.is_available() else "cpu"


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # d3rlpy >= 2.6 exposes a top-level seed helper; older versions ignore it.
    try:
        import d3rlpy as _d3

        if hasattr(_d3, "seed"):
            _d3.seed(seed)
    except Exception:  # pragma: no cover
        pass


def _row_to_obs_action(row: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Materialize (obs, action) for a single episode row.

    Prefers the F1.7 / F1.12 columns when present:
      - obs_features (JSON, length 10): concatenated with the query embedding
      - policy_action (JSON, length 2): raw pre-softmax action
      - policy_weights (JSON, length 2): used if policy_action missing

    Falls back to (embed_text(query), [cag_weight, graph_weight]) for legacy
    rows.
    """
    query = (row.get("query") or "").strip()
    if not query:
        return None

    embed = embed_text(query)

    feats_raw = row.get("obs_features")
    if feats_raw:
        try:
            feats = json.loads(feats_raw)
            obs = (
                build_observation(
                    query,
                    embed,
                    retrieval_results=None,  # unused when feats are supplied
                )
                if not isinstance(feats, list)
                else np.concatenate(
                    [embed.astype(np.float32), np.asarray(feats, dtype=np.float32)]
                )
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            obs = np.concatenate(
                [embed.astype(np.float32), np.zeros(10, dtype=np.float32)]
            )
    else:
        # legacy row: zero-pad to 394 so the trainer dataset is rectangular.
        obs = np.concatenate([embed.astype(np.float32), np.zeros(10, dtype=np.float32)])

    if obs.shape[0] != OBS_DIM:
        return None

    action_raw = row.get("policy_action") or row.get("policy_weights")
    if action_raw:
        try:
            action_list = json.loads(action_raw)
            action = np.asarray(action_list[:2], dtype=np.float32)
        except (json.JSONDecodeError, ValueError, TypeError):
            action = None
    else:
        action = None

    if action is None:
        cag_w = float(row.get("cag_weight") or 0.0)
        graph_w = float(row.get("graph_weight") or 0.0)
        action = np.asarray([cag_w, graph_w], dtype=np.float32)

    return obs.astype(np.float32), action


def _load_episodes_two_path(db_path: Path) -> MDPDataset | None:
    """Load (obs, action, reward) tuples from the 2-path episodes table."""
    if not db_path.exists():
        logger.warning("DB missing at %s", db_path)
        return None

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cols = {r[1] for r in cur.execute("PRAGMA table_info(episodes)").fetchall()}

    select_cols = ["query", "cag_weight", "graph_weight", "reward"]
    for opt in (
        "obs_features",
        "policy_weights",
        "policy_action",
        "from_cache",
        "had_empty_path",
        "schema_version",
    ):
        if opt in cols:
            select_cols.append(opt)

    where = ["query IS NOT NULL", "query != ''", "reward IS NOT NULL"]
    if "from_cache" in cols:
        # F1.6: CAG fast-path rows are not policy decisions; exclude them.
        where.append("(from_cache IS NULL OR from_cache = 0)")
    if "schema_version" in cols:
        where.append("(schema_version IS NULL OR schema_version >= 2)")
    sql = f"SELECT {', '.join(select_cols)} FROM episodes WHERE {' AND '.join(where)}"

    rows = [dict(r) for r in cur.execute(sql).fetchall()]
    conn.close()

    if not rows:
        logger.warning("No usable episodes in %s", db_path)
        return None

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    terminals: list[float] = []

    for row in rows:
        materialized = _row_to_obs_action(row)
        if materialized is None:
            continue
        obs, act = materialized
        observations.append(obs)
        actions.append(act)
        rewards.append(float(row["reward"]))
        terminals.append(1.0)

    if not observations:
        logger.warning("No materializable episodes after filtering")
        return None

    logger.info("Loaded %d 2-path episodes (obs_dim=%d)", len(observations), OBS_DIM)
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

    # F1.12: bound the action scaler so a re-run with a different episode
    # count cannot re-fit the scale. Actions are pre-softmax logits in
    # the [-1, 1] range produced by the policy mu head.
    cql = CQLConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        conservative_weight=5.0,
        batch_size=256,
        actor_encoder_factory=DefaultEncoderFactory(),
        critic_encoder_factory=DefaultEncoderFactory(),
        action_scaler=MinMaxActionScaler(
            minimum=np.array([-1.0, -1.0], dtype=np.float32),
            maximum=np.array([1.0, 1.0], dtype=np.float32),
        ),
    ).create(device=device)

    out_path = PROJECT_ROOT / "models" / "rl_policy_cql.d3"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = FusionEnv()
    best_reward = -float("inf")
    wait = 0

    for epoch in range(epochs):
        cql.fit(dataset, n_steps=steps_per_epoch, show_progress=False)

        ep_rewards: list[float] = []
        for i in range(eval_episodes):
            query = EVAL_QUERIES[i % len(EVAL_QUERIES)]
            obs, _ = env.reset(options={"query": query})
            # F1.1: 394-d obs feeds straight in. Predict expects a batch.
            action = cql.predict(np.array([obs]))[0]
            _, reward, _, _, _ = env.step(action)
            ep_rewards.append(float(reward))
        val_reward = float(np.mean(ep_rewards))
        logger.info(
            "Epoch %d | val_reward=%.3f (mean of %d evals)",
            epoch + 1,
            val_reward,
            eval_episodes,
        )

        if val_reward > best_reward + 0.01:
            best_reward = val_reward
            wait = 0
            cql.save_model(str(out_path))
            logger.info("Saved improved policy to %s", out_path)
        else:
            wait += 1
            if wait >= patience:
                logger.info(
                    "Early stop at epoch %d (no improvement for %d epochs)",
                    epoch + 1,
                    patience,
                )
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
