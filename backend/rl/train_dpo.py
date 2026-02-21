# Author: Bradley R. Kinnard
"""Direct Preference Optimization for fusion weight policy refinement.

Collects preference pairs from the replay buffer (high-reward vs low-reward
episodes), then trains an MLP policy using a DPO contrastive loss adapted
for continuous action spaces. This bypasses explicit reward modeling and
lets the policy learn directly from preference rankings.

Key pieces:
    load_preference_pairs()  - build (preferred, rejected) pairs from DB
    DPOFusionPolicy          - PyTorch MLP that outputs 4D action logits
    dpo_loss()               - contrastive loss for continuous actions
    train_dpo()              - full training loop with logging and checkpoints
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import cfg
from backend.core.utils import embed_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# config defaults
_rl_cfg = cfg.get("rl", {})
_dpo_cfg = _rl_cfg.get("dpo", {})
_DEFAULT_BETA = float(_dpo_cfg.get("beta", 0.1))
_DEFAULT_MARGIN = float(_dpo_cfg.get("reward_margin", 0.1))
_DEFAULT_MIN_PAIRS = int(_dpo_cfg.get("min_preference_pairs", 500))
_DEFAULT_DPO_LR = float(_dpo_cfg.get("learning_rate", 1e-4))


# ── Preference pair construction ─────────────────────────────────────────

def load_episodes_from_db(db_path: Path) -> list[dict[str, object]]:
    """Load episodes with query, reward, and weight data from SQLite."""
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
        "SELECT query, reward, rag_weight, cag_weight, graph_weight, response "
        "FROM episodes ORDER BY id"
    )
    rows = cursor.fetchall()
    conn.close()

    episodes: list[dict[str, object]] = []
    for query, reward, rag_w, cag_w, graph_w, response in rows:
        web_w = max(0.0, 1.0 - (rag_w + cag_w + graph_w))
        episodes.append({
            "query": str(query),
            "reward": float(reward),
            "weights": np.array([rag_w, cag_w, graph_w, web_w], dtype=np.float32),
            "response": str(response or ""),
        })

    logger.info("Loaded %d episodes from %s", len(episodes), db_path)
    return episodes


def build_preference_pairs(
    episodes: list[dict[str, object]],
    margin: float = _DEFAULT_MARGIN,
) -> list[tuple[dict[str, object], dict[str, object]]]:
    """Build (preferred, rejected) pairs where reward gap exceeds margin.

    Pairs episodes with overlapping query domains by comparing every
    episode against every other. Preferred episode has strictly higher
    reward by at least `margin`.
    """
    if len(episodes) < 2:
        return []

    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    for i, ep_a in enumerate(episodes):
        for j, ep_b in enumerate(episodes):
            if i == j:
                continue
            r_a = float(ep_a["reward"])  # type: ignore[arg-type]
            r_b = float(ep_b["reward"])  # type: ignore[arg-type]
            if r_a - r_b >= margin:
                pairs.append((ep_a, ep_b))

    logger.info(
        "Built %d preference pairs from %d episodes (margin=%.2f)",
        len(pairs), len(episodes), margin,
    )
    return pairs


# ── DPO policy network ──────────────────────────────────────────────────

class DPOFusionPolicy(nn.Module):
    """MLP policy for DPO training on continuous 4D fusion weights.

    Architecture matches the CQL actor for weight compatibility:
    Linear(384, 256) -> ReLU -> Linear(256, 256) -> ReLU -> Linear(256, 4)
    Outputs 4D action logits clamped to [-1, 1].
    """

    def __init__(self, obs_dim: int = 384, action_dim: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean_action, log_std) for the Gaussian policy."""
        h = self.encoder(obs)
        mean = torch.clamp(self.mu(h), -1.0, 1.0)
        return mean, self.log_std.expand_as(mean)

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of action under the Gaussian policy."""
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        # Gaussian log prob: -0.5 * ((a - mu) / sigma)^2 - log(sigma) - 0.5*log(2*pi)
        var = std ** 2
        log_p = -0.5 * ((action - mean) ** 2 / var + 2 * log_std + np.log(2 * np.pi))
        return log_p.sum(dim=-1)  # sum over action dims

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic prediction for inference (RLPolicy protocol)."""
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32)
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)
            mean, _ = self.forward(obs_t)
            return mean.cpu().numpy()

    @classmethod
    def from_cql_weights(cls, cql_path: Path) -> DPOFusionPolicy:
        """Initialize from CQL checkpoint for warm-start."""
        policy = cls()
        if not cql_path.exists():
            logger.warning("CQL path %s not found, using random init", cql_path)
            return policy

        device = "cpu"
        state_dict = torch.load(str(cql_path), weights_only=False, map_location=device)
        pw = state_dict.get("policy", state_dict)

        try:
            policy.encoder[0].weight.data = pw["_encoder._layers.0.weight"]
            policy.encoder[0].bias.data = pw["_encoder._layers.0.bias"]
            policy.encoder[2].weight.data = pw["_encoder._layers.2.weight"]
            policy.encoder[2].bias.data = pw["_encoder._layers.2.bias"]
            policy.mu.weight.data = pw["_mu.weight"]
            policy.mu.bias.data = pw["_mu.bias"]
            logger.info("DPO policy warm-started from CQL weights")
        except KeyError as exc:
            logger.warning("CQL key mismatch (%s), using random init", exc)

        return policy


# ── DPO loss ─────────────────────────────────────────────────────────────

def dpo_loss(
    policy: DPOFusionPolicy,
    ref_policy: DPOFusionPolicy,
    obs: torch.Tensor,
    preferred_action: torch.Tensor,
    rejected_action: torch.Tensor,
    beta: float = _DEFAULT_BETA,
) -> torch.Tensor:
    """Compute DPO loss for continuous action spaces.

    L = -log(sigmoid(beta * (log pi(a_w|s)/pi_ref(a_w|s) - log pi(a_l|s)/pi_ref(a_l|s))))

    Where a_w is the preferred (winning) action and a_l is rejected (losing).
    """
    # log ratios for preferred and rejected actions
    log_ratio_w = policy.log_prob(obs, preferred_action) - ref_policy.log_prob(obs, preferred_action)
    log_ratio_l = policy.log_prob(obs, rejected_action) - ref_policy.log_prob(obs, rejected_action)

    # DPO objective: maximize margin between preferred and rejected
    logits = beta * (log_ratio_w - log_ratio_l)
    loss = -F.logsigmoid(logits).mean()
    return loss


# ── Training loop ────────────────────────────────────────────────────────

def prepare_dpo_tensors(
    pairs: list[tuple[dict[str, object], dict[str, object]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert preference pairs to tensors: (observations, preferred_actions, rejected_actions).

    Embeds queries as 384-dim observations. Uses stored weights as actions.
    """
    obs_list: list[np.ndarray] = []
    pref_list: list[np.ndarray] = []
    rej_list: list[np.ndarray] = []

    for preferred, rejected in pairs:
        # use the preferred query's embedding as the observation
        query = str(preferred["query"])
        emb = embed_text(query)
        obs_list.append(emb)
        pref_list.append(np.array(preferred["weights"], dtype=np.float32))
        rej_list.append(np.array(rejected["weights"], dtype=np.float32))

    return (
        torch.tensor(np.array(obs_list), dtype=torch.float32),
        torch.tensor(np.array(pref_list), dtype=torch.float32),
        torch.tensor(np.array(rej_list), dtype=torch.float32),
    )


def train_dpo(
    pairs: list[tuple[dict[str, object], dict[str, object]]],
    beta: float = _DEFAULT_BETA,
    epochs: int = 50,
    learning_rate: float = _DEFAULT_DPO_LR,
    batch_size: int = 32,
    warm_start_path: Path | None = None,
    save_path: Path | None = None,
) -> Path:
    """Train a DPO fusion policy from preference pairs.

    Creates a policy network (optionally warm-started from CQL), a frozen
    reference copy, and optimizes via the DPO contrastive loss.
    """
    if len(pairs) < 2:
        raise ValueError(f"Need at least 2 preference pairs, got {len(pairs)}")

    # initialize policy (warm-start from CQL if available)
    cql_path = warm_start_path or (PROJECT_ROOT / "models" / "rl_policy_cql.d3")
    policy = DPOFusionPolicy.from_cql_weights(cql_path)

    # frozen reference policy (no gradient updates)
    ref_policy = DPOFusionPolicy.from_cql_weights(cql_path)
    for param in ref_policy.parameters():
        param.requires_grad = False
    ref_policy.eval()

    # prepare training data
    obs, pref_actions, rej_actions = prepare_dpo_tensors(pairs)
    n_samples = obs.shape[0]
    logger.info("DPO training: %d pairs, beta=%.2f, epochs=%d, lr=%.1e", n_samples, beta, epochs, learning_rate)

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    best_loss = float("inf")
    patience_counter = 0
    patience_limit = 10

    for epoch in range(epochs):
        # shuffle
        perm = torch.randperm(n_samples)
        obs_shuf = obs[perm]
        pref_shuf = pref_actions[perm]
        rej_shuf = rej_actions[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_obs = obs_shuf[start:end]
            batch_pref = pref_shuf[start:end]
            batch_rej = rej_shuf[start:end]

            loss = dpo_loss(policy, ref_policy, batch_obs, batch_pref, batch_rej, beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("DPO epoch %d/%d | loss=%.4f", epoch + 1, epochs, avg_loss)

        # early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.info("DPO early stop at epoch %d (patience=%d)", epoch + 1, patience_limit)
                break

    # save the trained policy
    if save_path is None:
        save_path = PROJECT_ROOT / "models" / "rl_policy_dpo.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), str(save_path))
    logger.info("DPO policy saved to %s (final loss=%.4f)", save_path, best_loss)
    return save_path


def load_dpo_policy(path: Path) -> DPOFusionPolicy:
    """Load a trained DPO policy from disk."""
    policy = DPOFusionPolicy()
    state = torch.load(str(path), weights_only=True, map_location="cpu")
    policy.load_state_dict(state)
    policy.eval()
    return policy


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DPO training for fusion weight policy")
    parser.add_argument("--db", type=str, default="db/rlfo_cache.db", help="Episode database path")
    parser.add_argument("--beta", type=float, default=_DEFAULT_BETA, help="DPO beta parameter")
    parser.add_argument("--margin", type=float, default=_DEFAULT_MARGIN, help="Reward margin for pairs")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=_DEFAULT_DPO_LR, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--save", type=str, default=None, help="Model save path")
    args = parser.parse_args()

    db_path = PROJECT_ROOT / args.db
    episodes = load_episodes_from_db(db_path)

    if not episodes:
        logger.error("No episodes in %s. Populate the replay buffer first.", db_path)
        sys.exit(1)

    pairs = build_preference_pairs(episodes, margin=args.margin)
    min_pairs = _DEFAULT_MIN_PAIRS

    if len(pairs) < min_pairs:
        logger.warning(
            "Only %d preference pairs (need %d for stable DPO training). "
            "Proceeding anyway, but results may be noisy.",
            len(pairs), min_pairs,
        )

    if not pairs:
        logger.error("Zero preference pairs. Need varied reward scores in the replay buffer.")
        sys.exit(1)

    save_path = Path(args.save) if args.save else None
    train_dpo(
        pairs,
        beta=args.beta,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
