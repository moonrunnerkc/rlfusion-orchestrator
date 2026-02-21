# Author: Bradley R. Kinnard
"""Agent protocols and shared type definitions for the orchestration pipeline.

Every agent follows plan() -> act() -> reflect(). The orchestrator wires
agents into a LangGraph DAG and classifies query complexity to pick the
right pipeline shape.
"""
from __future__ import annotations

import logging
from typing import Literal, Protocol, TypedDict, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# Complexity buckets drive pipeline routing
QueryComplexity = Literal["simple", "complex", "adversarial"]


class RetrieverItem(TypedDict, total=False):
    """Single chunk/result from any retriever path."""
    text: str
    score: float
    source: str
    id: str
    retriever: str
    url: str
    title: str
    csw_score: float
    local_stability: float
    question_fit: float
    drift_penalty: float


class RetrievalResults(TypedDict):
    """Aggregated results from all four retrieval paths."""
    rag: list[RetrieverItem]
    cag: list[RetrieverItem]
    graph: list[RetrieverItem]
    web: list[RetrieverItem]
    web_status: str


class PipelineState(TypedDict, total=False):
    """Shared mutable state passed through the agent pipeline.

    total=False because nodes progressively fill fields as they execute.
    LangGraph merges each node's return dict into this accumulated state.
    """
    # ----- Input -----
    query: str
    mode: str
    session_id: str

    # ----- Safety gate -----
    is_safe: bool
    safety_reason: str
    blocked: bool

    # ----- Memory / expansion -----
    expanded_query: str
    query_expanded: bool
    is_memory_request: bool
    memory_content: str

    # ----- Retrieval -----
    retrieval_results: RetrievalResults

    # ----- Fusion -----
    rl_weights: list[float]
    fused_context: str
    actual_weights: list[float]
    web_status: str
    context_parts: list[str]

    # ----- Profile / conversation -----
    profile_text: str
    conversation_context: str

    # ----- Prompt construction -----
    system_prompt: str
    user_prompt: str

    # ----- LLM output -----
    llm_response: str

    # ----- Critique -----
    reward: float
    clean_response: str
    proactive_suggestions: list[str]
    critique_reason: str
    faithfulness_checked: bool
    faithfulness_score: float
    sensitivity_level: float

    # ----- Meta -----
    complexity: QueryComplexity


class OrchestrationResult(TypedDict):
    """Final output consumed by endpoint handlers. Matches frozen API contract."""
    response: str
    fusion_weights: dict[str, float]
    reward: float
    proactive_suggestions: list[str]
    blocked: bool
    safety_reason: str
    web_status: str


class PreparedContext(TypedDict):
    """Pre-generation context for /ws streaming. Everything needed to call the LLM."""
    system_prompt: str
    user_prompt: str
    actual_weights: list[float]
    fused_context: str
    web_status: str
    expanded_query: str
    query_expanded: bool
    is_safe: bool
    safety_reason: str
    blocked: bool
    is_memory_request: bool
    memory_content: str


@runtime_checkable
class RLPolicy(Protocol):
    """Protocol for RL policies (CQL, PPO) that predict fusion weights."""
    def predict(self, obs: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class BaseAgent(Protocol):
    """Protocol for all agents in the multi-agent orchestration pipeline.

    Three-phase cycle per invocation:
        plan()    - analyze input, pick strategy
        act()     - execute strategy, produce state updates
        reflect() - evaluate output quality, log observations
    """

    @property
    def name(self) -> str:
        """Short identifier for logging and metrics."""
        ...

    def plan(self, state: PipelineState) -> PipelineState:
        """Analyze input state, determine execution strategy."""
        ...

    def act(self, state: PipelineState) -> PipelineState:
        """Execute strategy and return state updates."""
        ...

    def reflect(self, state: PipelineState) -> PipelineState:
        """Self-evaluate output quality. Returns reflection metadata."""
        ...
