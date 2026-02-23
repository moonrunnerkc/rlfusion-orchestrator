# Author: Bradley R. Kinnard
"""Orchestrator: LangGraph-powered agent pipeline with complexity-based routing.

Classifies queries as simple/complex/adversarial, constructs the appropriate
agent DAG, and runs it. Provides both full-pipeline (for /chat) and
prepare+finalize (for /ws streaming) interfaces.
"""
from __future__ import annotations

import logging
import re
from typing import ClassVar

from langgraph.graph import END, StateGraph

from backend.agents.base import (
    OrchestrationResult,
    PipelineState,
    PreparedContext,
    QueryComplexity,
    RLPolicy,
)
from backend.agents.critique_agent import CritiqueAgent
from backend.agents.fusion_agent import FusionAgent
from backend.agents.retrieval_agent import RetrievalAgent
from backend.agents.safety_agent import SafetyAgent
from backend.config import cfg
from backend.core.critique import get_critique_instruction, strip_critique_block
from backend.core.memory import (
    clear_memory,
    expand_query_with_context,
    get_context_for_prompt,
    record_turn,
)
from backend.core.profile import detect_and_save_memory, get_user_profile

logger = logging.getLogger(__name__)

# Prompt constants (moved from main.py for orchestrator ownership)
IDENTITY_BLOCK = """IDENTITY RULES:
- "I/me/my" (when YOU speak) = RLFusion AI assistant
- "you/your" (when USER addresses you) = RLFusion AI
- "I/me/my" (when USER speaks) = The human user
- Personal details in context (name, pets, job) = THE USER's info, not yours
- You are software, you don't have pets/vehicles/personal life

TERMINOLOGY (do NOT invent other expansions):
- CSWR = Chunk Stability Weighted Retrieval (filters RAG chunks by neighbor similarity, question fit, drift)
- CQL = Conservative Q-Learning (offline RL policy that routes retrieval weights)
- OOD = Out-of-Distribution detection (Mahalanobis distance)

USER PROFILE:
- If the context contains a USER PROFILE section, those are facts the user previously asked you to remember.
- When asked "have you remembered" or "do you know about me", check USER PROFILE first.
- When no profile data matches, say so honestly.
"""

WEB_INSTRUCTION = """
WEB RESULTS: [WEB:...] entries are live internet search. Prioritize them for current events, businesses, prices.
Cite source URLs. Ignore unrelated [RAG:...] results.
"""

# Keywords that indicate a personal/memory query
_PERSONAL_KEYWORDS = [
    "my ", "me ", "i ", "brad", "preference", "like", "favorite",
    "remember", "told you", "you know", "recall", "profile",
]

# Adversarial indicators for complexity classification
_ADVERSARIAL_PATTERNS = [
    re.compile(r"ignore\s+(previous|above|all)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:DAN|evil|unrestricted|jailbroken)", re.IGNORECASE),
    re.compile(r"(?:system|admin)\s*:\s*override", re.IGNORECASE),
]


def classify_complexity(query: str) -> QueryComplexity:
    """Classify query into simple/complex/adversarial for pipeline routing.

    Adversarial: known attack patterns detected.
    Simple: short factual lookups, single entity, low ambiguity.
    Complex: multi-hop reasoning, high sensitivity, lengthy queries.
    """
    # adversarial: known injection/attack patterns
    for pattern in _ADVERSARIAL_PATTERNS:
        if pattern.search(query):
            logger.info("Query classified as adversarial: matched %s", pattern.pattern[:40])
            return "adversarial"

    word_count = len(query.split())
    question_marks = query.count("?")

    # simple: short factual queries with a single question
    if word_count <= 8 and question_marks <= 1:
        return "simple"

    # complex: multi-part queries, long queries, or multiple questions
    if word_count > 25 or question_marks > 1:
        return "complex"

    # default to complex for safety
    return "complex"


def generate_system_prompt(mode: str, context_parts: list[str]) -> str:
    """Build the system prompt based on mode and context content."""
    has_cag_hit = any(p.startswith("[CAG:") for p in context_parts)
    cag_only = has_cag_hit and len([p for p in context_parts if p.startswith("[CAG:")]) == len(context_parts)
    has_web = any("[WEB:" in p for p in context_parts)
    critique_suffix = get_critique_instruction()

    if cag_only:
        return "Return the exact text after [CAG:] with no changes."

    if mode == "build":
        return f"""{IDENTITY_BLOCK}
You are an elite AI architect. Be INNOVATIVE, SPECIFIC, CUTTING-EDGE.
Never output [RAG:...], [CAG:...], [GRAPH:...], [WEB:...] tags.
{critique_suffix}"""

    web_block = WEB_INSTRUCTION if has_web else ""
    return f"""{IDENTITY_BLOCK}{web_block}
You are a knowledgeable assistant with access to retrieved context.
Never output source tags. Reference sources naturally.
RULES:
1. Read the retrieved context carefully for relevant information
2. Use your own knowledge to supplement when context is incomplete or off-topic
3. If context covers the topic well, ground your answer in it and cite specifics
4. If context is unrelated to the question, rely on your own knowledge and say so
5. Never fabricate sources or pretend context says something it does not
6. YOU are RLFusion (AI), USER is the human
7. Ignore context sections that are clearly from a different topic
{critique_suffix}"""


def generate_user_prompt(mode: str, query: str, fused_context: str, context_parts: list[str]) -> str:
    """Build the user prompt based on mode and context."""
    has_cag_hit = any(p.startswith("[CAG:") for p in context_parts)
    cag_only = has_cag_hit and len([p for p in context_parts if p.startswith("[CAG:")]) == len(context_parts)

    if cag_only:
        return f"Sources:\n{fused_context}\n\nReturn ONLY the text after [CAG:] - exact copy."

    if mode == "build":
        return f"""Knowledge sources:\n{fused_context}\n\nRequest: {query}\n\nSynthesize innovative concept:"""

    return f"CONTEXT:\n{fused_context}\n\nQUESTION: {query}\n\nANSWER:"


def apply_markdown_formatting(text: str) -> str:
    """Clean up LLM output: strip source tags, fix heading spacing."""
    text = re.sub(r'\[RAG:[^\]]+\]\s*', '', text)
    text = re.sub(r'\[CAG:[^\]]+\]\s*', '', text)
    text = re.sub(r'\[GRAPH:[^\]]+\]\s*', '', text)
    text = re.sub(r'\[WEB:[^\]]+\]\s*', '', text)
    text = re.sub(r'([^\n])(#{1,3}\s)', r'\1\n\n\2', text)
    text = re.sub(r'([.!?:])\s*(-\s)', r'\1\n\n\2', text)
    text = re.sub(r'([.!?])\s+([A-Z#])', r'\1\n\n\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


class Orchestrator:
    """Routes queries through the agent pipeline based on complexity.

    Pipeline shapes:
        Simple:      safety -> retrieve -> fuse
        Complex:     safety -> retrieve -> fuse (deeper retrieval)
        Adversarial: safety (strict) -> retrieve -> fuse

    Provides three interfaces:
        run()             - full pipeline including LLM generation (for /chat)
        prepare()         - single-call pre-generation assembly (for /chat)
        step_*() methods  - granular per-agent calls (for /ws with live status)
        finalize()        - post-generation critique and cleanup (for /ws streaming)
    """
    _NAME: ClassVar[str] = "orchestrator"

    def __init__(self, rl_policy: RLPolicy | None = None) -> None:
        self._safety = SafetyAgent()
        self._retrieval = RetrievalAgent()
        self._fusion = FusionAgent(rl_policy=rl_policy)
        self._critique = CritiqueAgent()
        self._graph = self._build_graph()

    @property
    def rl_policy(self) -> RLPolicy | None:
        return self._fusion.rl_policy

    @rl_policy.setter
    def rl_policy(self, policy: RLPolicy | None) -> None:
        self._fusion.rl_policy = policy

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph DAG for the agent pipeline."""
        builder = StateGraph(PipelineState)

        builder.add_node("safety", self._safety)
        builder.add_node("retrieve", self._retrieval)
        builder.add_node("fuse", self._fusion)

        builder.set_entry_point("safety")

        # safety gate: blocked queries short-circuit to END
        builder.add_conditional_edges(
            "safety",
            _safety_router,
            {"blocked": END, "retrieve": "retrieve"},
        )

        builder.add_edge("retrieve", "fuse")
        builder.add_edge("fuse", END)

        return builder.compile()

    # ---- Per-agent step methods for granular /ws pipeline status ----

    def step_preprocess(
        self,
        query: str,
        session_id: str = "",
    ) -> dict[str, object]:
        """Memory expansion + complexity classification. Fast, no LLM calls."""
        expanded_query, expansion_meta = expand_query_with_context(session_id, query)
        is_memory_request, memory_content = detect_and_save_memory(query)
        complexity = classify_complexity(query)
        return {
            "expanded_query": expanded_query,
            "query_expanded": expansion_meta["expanded"],
            "is_memory_request": is_memory_request,
            "memory_content": memory_content or "",
            "complexity": complexity,
        }

    def step_safety(self, query: str, complexity: QueryComplexity) -> PipelineState:
        """Run safety agent only. Returns safety gate result."""
        state: PipelineState = {"query": query, "complexity": complexity}
        return self._safety(state)

    def step_retrieval(
        self,
        query: str,
        expanded_query: str,
        query_expanded: bool,
        complexity: QueryComplexity,
    ) -> PipelineState:
        """Run retrieval agent only. Returns retrieval results + web_status."""
        state: PipelineState = {
            "query": query,
            "expanded_query": expanded_query,
            "query_expanded": query_expanded,
            "complexity": complexity,
        }
        return self._retrieval(state)

    def step_fusion(
        self,
        query: str,
        expanded_query: str,
        retrieval_results: dict[str, list[dict[str, float | str]]],
    ) -> PipelineState:
        """Run fusion agent only. Returns fused context + weights."""
        state: PipelineState = {
            "query": query,
            "expanded_query": expanded_query,
            "retrieval_results": retrieval_results,
        }
        return self._fusion(state)

    def step_stis_check(
        self,
        retrieval_results: dict[str, list[dict[str, float | str]]],
        query: str = "",
    ) -> dict[str, object]:
        """Evaluate whether to route to STIS engine for contradiction resolution.

        Checks RAG content against ontology facts via LLM claim comparison.
        Returns the full routing decision dict from should_route_to_stis().
        """
        from backend.core.critique import should_route_to_stis

        if not cfg.get("stis", {}).get("enabled", False):
            return {
                "route_to_stis": False,
                "reason": "STIS disabled in config",
                "contradiction": {"contradicted": False, "similarity": 1.0,
                                  "rag_claim": "", "graph_claim": ""},
                "best_cswr": 1.0,
            }

        rag = retrieval_results.get("rag", [])
        graph = retrieval_results.get("graph", [])
        return should_route_to_stis(rag, graph, query=query)

    def build_prompts(
        self,
        query: str,
        mode: str,
        session_id: str,
        fused_context: str,
        actual_weights: list[float],
        web_status: str,
        expanded_query: str,
        query_expanded: bool,
    ) -> PreparedContext:
        """Assemble system/user prompts from pipeline results. No agent execution."""
        # inject user profile for personal queries
        if any(w in query.lower() for w in _PERSONAL_KEYWORDS):
            profile = get_user_profile()
            if profile:
                fused_context = profile + "\n" + fused_context

        # inject conversation context
        conv_context = get_context_for_prompt(session_id)
        if conv_context:
            fused_context = conv_context + "\n\n" + fused_context

        context_parts = fused_context.split("\n\n")
        system_prompt = generate_system_prompt(mode, context_parts)
        user_prompt = generate_user_prompt(mode, query, fused_context, context_parts)

        return PreparedContext(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            actual_weights=actual_weights,
            fused_context=fused_context,
            web_status=web_status,
            expanded_query=expanded_query,
            query_expanded=query_expanded,
            is_safe=True,
            safety_reason="Safe",
            blocked=False,
            is_memory_request=False,
            memory_content="",
            retrieval_results={"rag": [], "cag": [], "graph": [], "web": [], "web_status": "disabled"},
        )

    # ---- Original single-call interface (kept for /chat endpoint) ----

    def prepare(
        self,
        query: str,
        mode: str = "chat",
        session_id: str = "",
    ) -> PreparedContext:
        """Pre-generation pipeline: safety, memory, retrieval, fusion, prompts.

        Called by /ws before streaming begins. Returns everything the endpoint
        needs to construct the LLM call and stream tokens.
        """
        # memory expansion
        expanded_query, expansion_meta = expand_query_with_context(session_id, query)

        # memory storage check
        is_memory_request, memory_content = detect_and_save_memory(query)
        if is_memory_request:
            return PreparedContext(
                system_prompt="",
                user_prompt="",
                actual_weights=[0.25, 0.25, 0.25, 0.25],
                fused_context="",
                web_status="disabled",
                expanded_query=expanded_query,
                query_expanded=expansion_meta["expanded"],
                is_safe=True,
                safety_reason="Safe",
                blocked=False,
                is_memory_request=True,
                memory_content=memory_content or "",
                retrieval_results={"rag": [], "cag": [], "graph": [], "web": [], "web_status": "disabled"},
            )

        # classify complexity
        complexity = classify_complexity(query)

        # run the LangGraph pipeline (safety -> retrieve -> fuse)
        initial_state: PipelineState = {
            "query": query,
            "mode": mode,
            "session_id": session_id,
            "expanded_query": expanded_query,
            "query_expanded": expansion_meta["expanded"],
            "complexity": complexity,
        }

        result = self._graph.invoke(initial_state)

        # check if blocked by safety
        if result.get("blocked", False):
            return PreparedContext(
                system_prompt="",
                user_prompt="",
                actual_weights=[0.0, 0.0, 0.0, 0.0],
                fused_context="",
                web_status=result.get("web_status", "disabled"),
                expanded_query=expanded_query,
                query_expanded=expansion_meta["expanded"],
                is_safe=False,
                safety_reason=result.get("safety_reason", "Query blocked"),
                blocked=True,
                is_memory_request=False,
                memory_content="",
                retrieval_results={"rag": [], "cag": [], "graph": [], "web": [], "web_status": "disabled"},
            )

        fused_context = result.get("fused_context", "")
        actual_weights = result.get("actual_weights", [0.25, 0.25, 0.25, 0.25])
        web_status = result.get("web_status", "disabled")
        retrieval_results = result.get("retrieval_results", {"rag": [], "cag": [], "graph": [], "web": [], "web_status": "disabled"})

        # inject user profile for personal queries
        if any(w in query.lower() for w in _PERSONAL_KEYWORDS):
            profile = get_user_profile()
            if profile:
                fused_context = profile + "\n" + fused_context

        # inject conversation context
        conv_context = get_context_for_prompt(session_id)
        if conv_context:
            fused_context = conv_context + "\n\n" + fused_context

        context_parts = fused_context.split("\n\n")
        system_prompt = generate_system_prompt(mode, context_parts)
        user_prompt = generate_user_prompt(mode, query, fused_context, context_parts)

        return PreparedContext(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            actual_weights=actual_weights,
            fused_context=fused_context,
            web_status=web_status,
            expanded_query=expanded_query,
            query_expanded=expansion_meta["expanded"],
            is_safe=True,
            safety_reason="Safe",
            blocked=False,
            is_memory_request=False,
            memory_content="",
            retrieval_results=retrieval_results,
        )

    def finalize(
        self,
        query: str,
        llm_response: str,
        fused_context: str,
        actual_weights: list[float],
        web_status: str,
    ) -> OrchestrationResult:
        """Post-generation pipeline: critique, cleanup, episode logging.

        Called by /ws after streaming completes. Runs the CritiqueAgent on the
        full response and returns the final result.
        """
        formatted_response = apply_markdown_formatting(llm_response)

        state: PipelineState = {
            "query": query,
            "llm_response": formatted_response,
            "fused_context": fused_context,
            "actual_weights": actual_weights,
        }
        critique_updates = self._critique(state)

        clean_response = critique_updates.get("clean_response", formatted_response)

        # append web search notice if API key is missing
        if web_status == "no_api_key":
            web_notice = (
                "\n\n---\n\u26a0\ufe0f **Web search is enabled but no API key is configured.** "
                "To enable web search, set `TAVILY_API_KEY` in your `.env` file. "
                "Get a free key at [tavily.com](https://tavily.com)."
            )
            clean_response += web_notice

        return OrchestrationResult(
            response=clean_response,
            fusion_weights={
                "rag": actual_weights[0] if len(actual_weights) > 0 else 0.0,
                "cag": actual_weights[1] if len(actual_weights) > 1 else 0.0,
                "graph": actual_weights[2] if len(actual_weights) > 2 else 0.0,
            },
            reward=critique_updates.get("reward", 0.0),
            proactive_suggestions=critique_updates.get("proactive_suggestions", []),
            blocked=False,
            safety_reason="Safe",
            web_status=web_status,
        )

    def run(
        self,
        query: str,
        mode: str = "chat",
        session_id: str = "",
    ) -> OrchestrationResult:
        """Full pipeline for /chat: safety -> retrieve -> fuse -> generate -> critique.

        Handles LLM generation internally (non-streaming) and returns the
        complete orchestration result matching the frozen API contract.
        """
        from ollama import Client

        prepared = self.prepare(query, mode, session_id)

        # early exit: memory request
        if prepared["is_memory_request"]:
            preview = prepared["memory_content"][:100]
            if len(prepared["memory_content"]) > 100:
                preview += "..."
            return OrchestrationResult(
                response=f"\u2705 **Remembered:**\n\n> {preview}",
                fusion_weights={"rag": 0.0, "cag": 0.0, "graph": 0.0},
                reward=1.0,
                proactive_suggestions=[],
                blocked=False,
                safety_reason="Safe",
                web_status="disabled",
            )

        # early exit: blocked by safety
        if prepared["blocked"]:
            return OrchestrationResult(
                response="I can't help with that request. " + prepared["safety_reason"],
                fusion_weights={"rag": 0.0, "cag": 0.0, "graph": 0.0},
                reward=0.0,
                proactive_suggestions=[],
                blocked=True,
                safety_reason=prepared["safety_reason"],
                web_status=prepared["web_status"],
            )

        # STIS contradiction check before LLM generation
        stis_decision = self.step_stis_check(prepared["retrieval_results"])

        if stis_decision["route_to_stis"]:
            from backend.core.stis_client import request_stis_consensus, log_stis_resolution

            contradiction = stis_decision["contradiction"]
            stis_result = request_stis_consensus(
                query,
                str(contradiction["rag_claim"]),
                str(contradiction["graph_claim"]),
            )
            log_stis_resolution(
                query,
                str(contradiction["rag_claim"]),
                str(contradiction["graph_claim"]),
                float(contradiction["similarity"]),
                float(stis_decision["best_cswr"]),
                stis_result,
            )
            if stis_result["resolved"]:
                logger.info("STIS resolved contradiction for /chat query")
                llm_response = stis_result["resolution"]["text"]
                return self.finalize(
                    query=query,
                    llm_response=llm_response,
                    fused_context=prepared["fused_context"],
                    actual_weights=prepared["actual_weights"],
                    web_status=prepared["web_status"],
                )
            logger.warning("STIS failed (%s), falling back to Ollama",
                           stis_result["error"])

        # LLM generation via Ollama (non-streaming for /chat)
        client = Client(host=cfg["llm"]["host"])
        response_obj = client.chat(
            model=cfg["llm"]["model"],
            messages=[
                {"role": "system", "content": prepared["system_prompt"]},
                {"role": "user", "content": prepared["user_prompt"]},
            ],
            options={"temperature": 0.3},
        )
        llm_response = response_obj["message"]["content"]

        return self.finalize(
            query=query,
            llm_response=llm_response,
            fused_context=prepared["fused_context"],
            actual_weights=prepared["actual_weights"],
            web_status=prepared["web_status"],
        )


def _safety_router(state: PipelineState) -> str:
    """LangGraph conditional edge: route based on safety gate result."""
    if state.get("blocked", False):
        return "blocked"
    return "retrieve"
