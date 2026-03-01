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

# Prompt constants (single source of truth for all endpoints)
IDENTITY_BLOCK = """You are RLFusion AI, an expert retrieval-augmented assistant.
Personal details in context belong to the USER, not you.
If context contains a USER PROFILE section, use those facts to personalize answers.
"""

WEB_INSTRUCTION = """
WEB RESULTS: Web search is currently disabled.
"""

# Token budgets per mode (generation cap)
NUM_PREDICT = {"chat": 400, "build": 500}

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
        web_block = WEB_INSTRUCTION if has_web else ""
        return f"""{IDENTITY_BLOCK}{web_block}
You are an expert AI architect and systems designer.
Do not output raw source tags like [RAG:...] or [GRAPH:...].
INSTRUCTIONS:
1. Answer ONLY the design request. Nothing else.
2. If reference material below is relevant, integrate its patterns into your design.
3. If reference material is NOT relevant, pretend it does not exist. Never mention, acknowledge, or refer to it.
4. Name specific technologies, protocols, data structures, and algorithms.
5. Be concrete and production-grade. Include components, data flow, and failure handling.
6. Never produce generic textbook answers. Be opinionated.
{critique_suffix}"""

    web_block = WEB_INSTRUCTION if has_web else ""
    return f"""{IDENTITY_BLOCK}{web_block}
You have retrieved context from knowledge sources below.
Do not output raw source tags. Cite information naturally.
RULES:
1. Answer ONLY the question asked. Nothing else.
2. If context is relevant, ground your answer in it. Cite specific facts, numbers, and details.
3. If context is partially relevant, use relevant parts and supplement with your own knowledge.
4. If context is unrelated, pretend it does not exist. Never mention, acknowledge, or refer to it. Answer from your own knowledge.
5. Never fabricate sources or claim context says something it does not.
{critique_suffix}"""


def generate_user_prompt(mode: str, query: str, fused_context: str, context_parts: list[str]) -> str:
    """Build the user prompt based on mode and context."""
    has_cag_hit = any(p.startswith("[CAG:") for p in context_parts)
    cag_only = has_cag_hit and len([p for p in context_parts if p.startswith("[CAG:")]) == len(context_parts)
    has_context = bool(fused_context.strip())

    if cag_only:
        return f"Sources:\n{fused_context}\n\nReturn ONLY the text after [CAG:] - exact copy."

    if mode == "build":
        if has_context:
            return (f"REFERENCE MATERIAL:\n{fused_context}\n\n"
                    f"DESIGN REQUEST: {query}\n\n"
                    f"Produce a concrete, production-grade architecture. "
                    f"Integrate relevant reference material where it applies:")
        return f"DESIGN REQUEST: {query}\n\nProduce a concrete, production-grade architecture:"

    if has_context:
        return (f"RETRIEVED CONTEXT:\n{fused_context}\n\n"
                f"QUESTION: {query}\n\n"
                f"Answer the question. Use the context above if relevant, "
                f"otherwise answer from your own knowledge:")
    return f"QUESTION: {query}\n\nAnswer from your knowledge:"


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

        # enforce total context budget (~5000 chars, ~1250 tokens)
        if len(fused_context) > 5000:
            fused_context = fused_context[:5000]

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
                actual_weights=[0.5, 0.5],
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
                actual_weights=[0.0, 0.0],
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
        actual_weights = result.get("actual_weights", [0.5, 0.5])
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
                "cag": actual_weights[0] if len(actual_weights) > 0 else 0.0,
                "graph": actual_weights[1] if len(actual_weights) > 1 else 0.0,
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
        prepared = self.prepare(query, mode, session_id)

        # early exit: memory request
        if prepared["is_memory_request"]:
            preview = prepared["memory_content"][:100]
            if len(prepared["memory_content"]) > 100:
                preview += "..."
            return OrchestrationResult(
                response=f"\u2705 **Remembered:**\n\n> {preview}",
                fusion_weights={"cag": 0.0, "graph": 0.0},
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
                fusion_weights={"cag": 0.0, "graph": 0.0},
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

        # LLM generation via inference engine (non-streaming for /chat)
        from backend.core.model_router import get_engine
        engine = get_engine()
        _num_predict = NUM_PREDICT.get(mode, 800)
        llm_response = engine.generate(
            messages=[
                {"role": "system", "content": prepared["system_prompt"]},
                {"role": "user", "content": prepared["user_prompt"]},
            ],
            temperature=0.3, num_ctx=4096, num_predict=_num_predict,
        )

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
