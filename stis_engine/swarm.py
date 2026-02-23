# Author: Bradley R. Kinnard
"""Core STIS consensus engine: multi-agent convergence in continuous latent space.

Forces N agents to reach mathematical agreement on hidden state vectors before
any discrete token is sampled. Uses pairwise cosine similarity and centroid
blending until mean similarity exceeds the configured threshold.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from stis_engine.config import ModelConfig, SwarmConfig

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceStep:
    """Snapshot of a single consensus iteration for audit trails."""
    iteration: int
    mean_similarity: float
    max_deviation: float
    centroid_norm: float


@dataclass
class SwarmResult:
    """Complete output from a swarm consensus generation run."""
    text: str
    total_tokens: int
    convergence_log: list[list[ConvergenceStep]] = field(default_factory=list)
    final_similarity: float = 0.0
    total_iterations: int = 0
    wall_time_secs: float = 0.0

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "total_tokens": self.total_tokens,
            "convergence_log": [
                [{"iteration": s.iteration, "mean_similarity": s.mean_similarity,
                  "max_deviation": s.max_deviation, "centroid_norm": s.centroid_norm}
                 for s in steps]
                for steps in self.convergence_log
            ],
            "final_similarity": self.final_similarity,
            "total_iterations": self.total_iterations,
            "wall_time_secs": self.wall_time_secs,
        }


class SwarmEngine:
    """Multi-agent consensus engine operating on transformer hidden states.

    Each agent maintains an independent copy of the input sequence. At every
    generation step, the final hidden state vectors are extracted before the
    lm_head projection. The engine computes pairwise cosine similarity across
    agents. If mean similarity falls below the threshold, a centroid blend
    pulls the agents toward agreement. Once similarity exceeds the threshold,
    the unified centroid is projected through lm_head to produce the next token.
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                 model_cfg: ModelConfig, swarm_cfg: SwarmConfig) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._model_cfg = model_cfg
        self._swarm_cfg = swarm_cfg
        self._device = next(model.parameters()).device
        self._hidden_dim = model.config.hidden_size
        self._lm_head = model.lm_head

        logger.info("SwarmEngine initialized: %d agents, threshold=%.3f, alpha=%.2f, hidden_dim=%d",
                     swarm_cfg.num_agents, swarm_cfg.similarity_threshold, swarm_cfg.alpha,
                     self._hidden_dim)

    # minimum tokens before we honor EOS, prevents empty output
    _MIN_TOKENS_BEFORE_EOS = 20

    # noise scale for agent diversity (fraction of hidden state norm)
    _DIVERSITY_SCALE = 0.008

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int | None = None) -> SwarmResult:
        """Run swarm consensus generation on the given prompt.

        Each token step runs the convergence loop until agents agree,
        then samples from the unified centroid's logit distribution.
        Agents are initialized with small random perturbations to prevent
        trivial convergence on the first iteration.
        """
        t_start = time.perf_counter()
        max_tokens = max_new_tokens or self._model_cfg.max_new_tokens
        num_agents = self._swarm_cfg.num_agents
        threshold = self._swarm_cfg.similarity_threshold
        alpha = self._swarm_cfg.alpha
        max_iters = self._swarm_cfg.max_iterations

        # tokenize input and replicate across agents
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt").to(self._device)
        agent_sequences = [input_ids.clone() for _ in range(num_agents)]

        generated_tokens: list[int] = []
        all_convergence_logs: list[list[ConvergenceStep]] = []
        cumulative_iters = 0
        last_similarity = 0.0

        eos_id = self._tokenizer.eos_token_id

        for token_step in range(max_tokens):
            # collect final hidden states from each agent
            agent_states = self._extract_hidden_states(agent_sequences)

            # inject diversity: perturb each agent's hidden state so they
            # don't trivially converge on iteration 0 from identical inputs
            if num_agents > 1:
                state_norm = agent_states.norm(dim=-1, keepdim=True).mean()
                noise_scale = self._DIVERSITY_SCALE * state_norm
                noise = torch.randn_like(agent_states) * noise_scale
                # keep agent 0 as anchor, perturb the rest
                noise[0] = 0.0
                agent_states = agent_states + noise

            # run convergence loop
            step_log, converged_centroid, iters_used, sim = self._converge(
                agent_states, threshold, alpha, max_iters
            )
            all_convergence_logs.append(step_log)
            cumulative_iters += iters_used
            last_similarity = sim

            # project the converged centroid through lm_head to get logits
            next_token_id = self._sample_from_centroid(converged_centroid)

            # enforce minimum output length before honoring EOS
            if next_token_id == eos_id and len(generated_tokens) < self._MIN_TOKENS_BEFORE_EOS:
                # suppress EOS, sample next best token instead
                next_token_id = self._sample_from_centroid(
                    converged_centroid, suppress_eos=True
                )

            generated_tokens.append(next_token_id)

            if next_token_id == eos_id:
                break

            # append the chosen token to all agent sequences
            token_tensor = torch.tensor([[next_token_id]], device=self._device)
            agent_sequences = [
                torch.cat([seq, token_tensor], dim=-1) for seq in agent_sequences
            ]

        output_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        wall_time = time.perf_counter() - t_start

        return SwarmResult(
            text=output_text,
            total_tokens=len(generated_tokens),
            convergence_log=all_convergence_logs,
            final_similarity=last_similarity,
            total_iterations=cumulative_iters,
            wall_time_secs=round(wall_time, 4),
        )

    def _extract_hidden_states(self, agent_sequences: list[torch.Tensor]) -> torch.Tensor:
        """Run each agent's sequence through the model, extract final hidden state.

        Returns tensor of shape (num_agents, hidden_dim) containing the last-token
        hidden state for each agent before lm_head projection.
        """
        states = []
        for seq in agent_sequences:
            outputs = self._model(input_ids=seq, output_hidden_states=True)
            # last layer, last token position
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            states.append(last_hidden.squeeze(0))
        return torch.stack(states)  # (num_agents, hidden_dim)

    def _converge(
        self,
        agent_states: torch.Tensor,
        threshold: float,
        alpha: float,
        max_iters: int,
    ) -> tuple[list[ConvergenceStep], torch.Tensor, int, float]:
        """Run the consensus blending loop until similarity > threshold.

        At each iteration:
        1. Compute pairwise cosine similarity across agents.
        2. If mean similarity >= threshold, return the centroid.
        3. Otherwise blend: state_new = (1 - alpha) * state + alpha * centroid.

        Returns (step_log, converged_centroid, iterations_used, final_similarity).
        """
        states = agent_states.clone()
        step_log: list[ConvergenceStep] = []

        for i in range(max_iters):
            # normalize for cosine similarity
            normed = F.normalize(states, dim=-1)
            # pairwise cosine sim matrix: (N, N)
            sim_matrix = normed @ normed.T
            num_agents = states.shape[0]

            # extract upper triangle (all unique pairs, excluding self-sim)
            mask = torch.triu(torch.ones(num_agents, num_agents, device=states.device), diagonal=1).bool()
            pairwise_sims = sim_matrix[mask]
            mean_sim = float(pairwise_sims.mean())
            max_dev = float((1.0 - pairwise_sims).max())

            centroid = states.mean(dim=0)
            centroid_norm = float(centroid.norm())

            step_log.append(ConvergenceStep(
                iteration=i,
                mean_similarity=round(mean_sim, 6),
                max_deviation=round(max_dev, 6),
                centroid_norm=round(centroid_norm, 4),
            ))

            if mean_sim >= threshold:
                return step_log, centroid, i + 1, mean_sim

            # blend each agent toward the centroid
            states = (1.0 - alpha) * states + alpha * centroid.unsqueeze(0)

        # hit iteration cap: return best-effort centroid
        centroid = states.mean(dim=0)
        final_sim = float(pairwise_sims.mean()) if len(pairwise_sims) > 0 else 0.0
        logger.warning("Convergence cap reached after %d iterations (sim=%.4f, threshold=%.4f)",
                        max_iters, final_sim, threshold)
        return step_log, centroid, max_iters, final_sim

    def _sample_from_centroid(self, centroid: torch.Tensor,
                               suppress_eos: bool = False) -> int:
        """Project the converged centroid through lm_head and sample a token.

        Uses temperature + top-p (nucleus) sampling for controlled stochasticity.
        When suppress_eos is True, forces EOS logit to -inf so the model
        picks a real content token instead.
        """
        logits = self._lm_head(centroid.unsqueeze(0))  # (1, vocab_size)
        logits = logits.squeeze(0)  # (vocab_size,)

        # suppress EOS when below minimum token threshold
        if suppress_eos and self._tokenizer.eos_token_id is not None:
            logits[self._tokenizer.eos_token_id] = float("-inf")

        temperature = self._model_cfg.temperature
        if temperature > 0:
            logits = logits / temperature

        # top-p (nucleus) filtering
        top_p = self._model_cfg.top_p
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # remove tokens outside the nucleus
        removal_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[removal_mask] = float("-inf")

        # sample from the filtered distribution
        probs = F.softmax(sorted_logits, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1)
        token_id = int(sorted_indices[sampled_idx])

        return token_id

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def num_agents(self) -> int:
        return self._swarm_cfg.num_agents

    @property
    def similarity_threshold(self) -> float:
        return self._swarm_cfg.similarity_threshold
