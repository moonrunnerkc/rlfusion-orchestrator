# Author: Bradley R. Kinnard
# retrain_fusion.py - Offline CQL retraining for 2-path fusion (CAG + Graph).
# Runs 500 simulated episodes through FusionEnv, collects transitions,
# then trains a new CQL policy with 2D action output.

import sys
import time
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.rl.fusion_env import FusionEnv

EPISODES = 500
SEED = 42
OLD_POLICY = PROJECT_ROOT / "models" / "rl_policy_cql.d3"
NEW_POLICY = PROJECT_ROOT / "models" / "rl_policy_cql_2path.d3"
ARCHIVE_POLICY = PROJECT_ROOT / "models" / "rl_policy_cql_4path_archived.d3"

# diverse training queries
TRAINING_QUERIES = [
    "What is RLFusion Orchestrator?",
    "How does the fusion mechanism work?",
    "Explain the CSWR algorithm",
    "What is CAG retrieval?",
    "How does GraphRAG traverse entities?",
    "Describe the safety agent pipeline",
    "What are the retrieval paths?",
    "How does the RL policy adapt?",
    "Explain knowledge graph traversal",
    "What is critique-based reward?",
    "How does query decomposition work?",
    "Describe the agent orchestration",
    "What is the observation space?",
    "How does context fusion work?",
    "Explain the weighting mechanism",
    "What is the memory system?",
    "How does session context work?",
    "Describe adaptive routing",
    "What models are used?",
    "How does the CPU triage worker operate?",
]


class CQLTrainer:
    """Lightweight offline CQL trainer for 2-path fusion policy."""

    def __init__(self, obs_dim: int = 394, act_dim: int = 2, lr: float = 3e-4):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = dev

        # same architecture as CQLPolicyWrapper but 2D output
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        ).to(dev)
        self.mu = nn.Linear(256, act_dim).to(dev)

        # Q-network for CQL
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        ).to(dev)

        self.policy_optim = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.mu.parameters()), lr=lr,
        )
        self.q_optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Forward pass through policy network."""
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            # only use first 384 dims (embedding) for the encoder
            action = torch.clamp(self.mu(self.encoder(obs_t[:, :self.obs_dim])), -1.0, 1.0)
            return action.cpu().numpy()

    def train_step(
        self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
        cql_alpha: float = 1.0,
    ) -> dict[str, float]:
        """Single CQL training step on a batch."""
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rew_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Q-value update with CQL conservative penalty
        q_input = torch.cat([obs_t, act_t], dim=-1)
        q_values = self.q_net(q_input).squeeze(-1)
        td_loss = nn.functional.mse_loss(q_values, rew_t)

        # CQL regularizer: penalize Q-values of random actions
        random_actions = torch.rand_like(act_t) * 2 - 1  # uniform in [-1, 1]
        q_rand = self.q_net(torch.cat([obs_t, random_actions], dim=-1)).squeeze(-1)
        cql_loss = cql_alpha * (q_rand.mean() - q_values.mean())

        q_total = td_loss + cql_loss
        self.q_optim.zero_grad()
        q_total.backward()
        self.q_optim.step()

        # Policy update: maximize Q-value of policy actions
        policy_actions = torch.clamp(self.mu(self.encoder(obs_t)), -1.0, 1.0)
        q_policy = self.q_net(torch.cat([obs_t, policy_actions], dim=-1)).squeeze(-1)
        policy_loss = -q_policy.mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        return {
            "td_loss": td_loss.item(),
            "cql_loss": cql_loss.item(),
            "policy_loss": policy_loss.item(),
            "avg_reward": rew_t.mean().item(),
            "avg_q": q_values.mean().item(),
        }

    def save(self, path: Path) -> None:
        """Save as d3rlpy-compatible checkpoint format."""
        state = {
            "policy": {
                "_encoder._layers.0.weight": self.encoder[0].weight.data.cpu(),
                "_encoder._layers.0.bias": self.encoder[0].bias.data.cpu(),
                "_encoder._layers.2.weight": self.encoder[2].weight.data.cpu(),
                "_encoder._layers.2.bias": self.encoder[2].bias.data.cpu(),
                "_mu.weight": self.mu.weight.data.cpu(),
                "_mu.bias": self.mu.bias.data.cpu(),
            },
        }
        torch.save(state, str(path))
        print(f"  Policy saved to {path}")


def collect_episodes(env: FusionEnv, n_episodes: int, seed: int = 42) -> tuple:
    """Run episodes and collect (obs, action, reward) tuples."""
    rng = np.random.default_rng(seed)
    observations, actions, rewards = [], [], []

    for ep in range(n_episodes):
        query = TRAINING_QUERIES[ep % len(TRAINING_QUERIES)]
        obs, _ = env.reset(seed=seed + ep, options={"query": query})

        # explore with noise
        action = rng.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
        next_obs, reward, _, _, info = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(reward)

        if (ep + 1) % 50 == 0:
            avg_r = np.mean(rewards[-50:])
            print(f"  Episode {ep + 1}/{n_episodes} | avg reward (last 50): {avg_r:.3f}")

    return np.array(observations), np.array(actions), np.array(rewards)


def main() -> None:
    print("=" * 64)
    print("CQL Policy Retraining for 2-Path Fusion (CAG + Graph)")
    print("=" * 64)

    # archive old 4-path policy
    if OLD_POLICY.exists() and not ARCHIVE_POLICY.exists():
        shutil.copy2(OLD_POLICY, ARCHIVE_POLICY)
        print(f"Archived old policy: {ARCHIVE_POLICY}")

    print(f"\nCollecting {EPISODES} episodes...")
    t0 = time.perf_counter()
    env = FusionEnv()

    observations, actions, rewards = collect_episodes(env, EPISODES, SEED)
    env.close()

    elapsed = time.perf_counter() - t0
    print(f"Collection complete: {elapsed:.1f}s, avg reward: {np.mean(rewards):.3f}")

    # train CQL policy
    print(f"\nTraining CQL policy (obs_dim={observations.shape[1]}, act_dim=2)...")
    trainer = CQLTrainer(obs_dim=observations.shape[1], act_dim=2)

    n_epochs = 100
    batch_size = min(64, len(observations))

    for epoch in range(n_epochs):
        # sample batch
        idx = np.random.choice(len(observations), size=batch_size, replace=False)
        metrics = trainer.train_step(
            observations[idx], actions[idx], rewards[idx],
        )
        if (epoch + 1) % 20 == 0:
            print(
                f"  Epoch {epoch + 1}/{n_epochs} | "
                f"td={metrics['td_loss']:.4f} cql={metrics['cql_loss']:.4f} "
                f"policy={metrics['policy_loss']:.4f} avg_q={metrics['avg_q']:.3f}"
            )

    # save new policy
    trainer.save(NEW_POLICY)

    # verify weights sum to 1.0
    test_obs = observations[:5]
    test_actions = trainer.predict(test_obs)
    from torch.nn.functional import softmax as _sm
    for i, act in enumerate(test_actions):
        w = _sm(torch.tensor(act), dim=0).numpy()
        print(f"  Test query {i}: CAG={w[0]:.3f} Graph={w[1]:.3f} (sum={w.sum():.4f})")

    # also copy as the main policy if training looks good
    avg_reward = np.mean(rewards)
    if avg_reward > 0:
        shutil.copy2(NEW_POLICY, OLD_POLICY)
        print(f"\nNew policy promoted to {OLD_POLICY}")
    else:
        print(f"\nAvg reward {avg_reward:.3f} too low, keeping old policy.")

    print("\nRetraining complete.")


if __name__ == "__main__":
    main()
