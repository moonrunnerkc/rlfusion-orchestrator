"""
RLFusion Orchestrator: FastAPI backend for multi-source retrieval with reinforcement learning.

This module implements a unified API for retrieval-augmented generation (RAG),
context-augmented generation (CAG), and knowledge graph fusion, with dynamic
source weighting learned via reinforcement learning.

Author: Bradley R. Kinnard
License: MIT
"""

import json
import logging
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ollama import Client
from stable_baselines3 import PPO

from backend.config import cfg
from backend.core.critique import critique, log_episode_to_replay_buffer
from backend.core.fusion import fuse_context
from backend.core.profile import detect_and_save_memory, get_user_profile
from backend.core.retrievers import get_rag_index, retrieve
from backend.core.utils import embed_text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

if not hasattr(np, '_core'):
    _core_module = types.ModuleType('numpy._core')
    _core_module.numeric = np
    sys.modules['numpy._core'] = _core_module
    sys.modules['numpy._core.numeric'] = np
    logger.info("Applied numpy._core compatibility shim for stable-baselines3")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA-capable GPU required for RLFusion Orchestrator")

device = torch.device("cuda")
logger.info(f"Initialized on GPU: {torch.cuda.get_device_name(0)}")
logger.info(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

app = FastAPI(
    title="RLFusion Orchestrator",
    version="0.1.0",
    description="Multi-source retrieval system with RL-optimized fusion"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_rl_policy: Optional[PPO] = None


def _apply_markdown_formatting(text: str) -> str:
    """
    Apply consistent markdown formatting to generated text.

    Ensures proper spacing around headers, bullet points, and paragraphs
    while preserving semantic content.

    Args:
        text: Raw generated text from LLM

    Returns:
        Formatted text with proper markdown structure
    """
    text = re.sub(r'([^\n])(#{1,3}\s)', r'\1\n\n\2', text)
    text = re.sub(r'([.!?:])\s*(-\s)', r'\1\n\n\2', text)
    text = re.sub(r'([.!?])\s+([A-Z#])', r'\1\n\n\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _compute_rl_fusion_weights(query: str, policy: Optional[PPO]) -> np.ndarray:
    """
    Compute source fusion weights using RL policy or heuristics.

    This function generates a 4D weight vector [RAG, CAG, Graph, Web] by either:
    1. Using the trained RL policy to predict optimal weights based on query context
    2. Falling back to query-based heuristics if policy is unavailable or outputs are uniform

    Args:
        query: User query string
        policy: Trained PPO policy (may be None)

    Returns:
        Normalized 4D numpy array of weights summing to 1.0
    """
    web_enabled = cfg.get("web", {}).get("enabled", False)

    if policy is not None:
        try:
            prev_queries = ["", ""]
            context_queries = prev_queries + [query]
            obs = np.concatenate([embed_text(q) for q in context_queries])
            obs = obs.reshape(1, -1)

            action, _ = policy.predict(obs, deterministic=True)
            weights = action.flatten()

            if len(weights) == 3:
                weights = np.append(weights, 0.0)

            exp_w = np.exp(weights)
            rl_weights = exp_w / np.sum(exp_w)

            if max(rl_weights) - min(rl_weights) < 0.05:
                logger.info("Policy outputs uniform, applying query heuristics")
                query_lower = query.lower()

                if any(kw in query_lower for kw in ['http://', 'https://', 'website', '.com', 'look up']):
                    rl_weights = np.array([0.2, 0.1, 0.2, 0.5])
                elif any(kw in query_lower for kw in ['how does', 'architecture', 'design', 'workflow', 'system']):
                    rl_weights = np.array([0.2, 0.1, 0.6, 0.1])
                elif any(kw in query_lower for kw in ['what is', 'explain', 'describe', 'document']):
                    rl_weights = np.array([0.6, 0.2, 0.1, 0.1])
                else:
                    rl_weights = np.array([0.4, 0.2, 0.3, 0.1])

            if not web_enabled:
                rl_weights[3] = 0.0
                if np.sum(rl_weights[:3]) > 0:
                    rl_weights[:3] = rl_weights[:3] / np.sum(rl_weights[:3])

            logger.info(f"RL weights: RAG={rl_weights[0]:.2f} CAG={rl_weights[1]:.2f} "
                       f"Graph={rl_weights[2]:.2f} Web={rl_weights[3]:.2f}")
            return rl_weights

        except Exception as e:
            logger.warning(f"Policy prediction failed: {e}, using defaults")

    default_weights = np.array([0.25, 0.25, 0.25, 0.25])
    if not web_enabled:
        default_weights = np.array([0.33, 0.33, 0.34, 0.0])

    return default_weights


def _build_fusion_context(
    retrieval_results: Dict[str, List[Dict[str, Any]]],
    weights: np.ndarray
) -> str:
    """
    Construct weighted fusion context from multi-source retrieval results.

    Applies score thresholds and weight-based selection to determine which
    retrieved chunks are included in the final context sent to the LLM.

    Args:
        retrieval_results: Dict with keys 'rag', 'cag', 'graph', 'web'
        weights: 4D weight array from RL policy

    Returns:
        Formatted fusion context string with source annotations
    """
    rag_results = retrieval_results["rag"]
    cag_results = retrieval_results["cag"]
    graph_results = retrieval_results["graph"]
    web_results = retrieval_results.get("web", [])

    total_items = 15
    rag_take = max(2, int(weights[0] * total_items))
    cag_take = max(1, int(weights[1] * total_items))
    graph_take = max(1, int(weights[2] * total_items))
    web_take = max(1, int(weights[3] * total_items)) if len(weights) > 3 else 0

    rag_items = [r for r in rag_results if r["score"] >= 0.40][:rag_take]
    cag_items = [c for c in cag_results if c["score"] >= 0.85][:cag_take]
    graph_items = [g for g in graph_results if g["score"] >= 0.50][:graph_take]
    web_items = [w for w in web_results if w["score"] >= 0.60][:web_take] if web_take > 0 else []

    context_parts = []
    for r in rag_items:
        context_parts.append(f"[RAG:{r['score']:.2f}|w={weights[0]:.2f}] {r['text']}")
    for c in cag_items:
        context_parts.append(f"[CAG:{c['score']:.2f}|w={weights[1]:.2f}] {c['text']}")
    for g in graph_items:
        context_parts.append(f"[GRAPH:{g['score']:.2f}|w={weights[2]:.2f}] {g['text']}")
    for w in web_items:
        context_parts.append(
            f"[WEB:{w['score']:.2f}|w={weights[3]:.2f}] Source: {w.get('url', 'unknown')}\n{w['text'][:600]}"
        )

    logger.info(f"Fusion stats: {len(rag_items)} RAG, {len(cag_items)} CAG, "
               f"{len(graph_items)} Graph, {len(web_items)} Web")

    if context_parts:
        return "\n\n".join(context_parts)
    else:
        return "No high-confidence sources available."


def _generate_system_prompt(mode: str, context_parts: List[str]) -> str:
    """
    Generate appropriate system prompt based on interaction mode.

    Args:
        mode: Either "chat", "build", or determined by context
        context_parts: List of context strings (for CAG detection)

    Returns:
        System prompt string
    """
    has_cag_hit = any(p.startswith("[CAG:") for p in context_parts)
    cag_only = has_cag_hit and len([p for p in context_parts if p.startswith("[CAG:")]) == len(context_parts)

    if cag_only:
        return "You are a cache retrieval system. Your ONLY job is to return the exact text after [CAG:X.XX] with no changes whatsoever."

    if mode == "build":
        return """You are an elite AI systems architect. Your responses must be:
1. INNOVATIVE - synthesize novel ideas from the context, don't just repeat it
2. SPECIFIC - concrete architectures with technical depth
3. CUTTING-EDGE - push boundaries, propose unconventional approaches
4. BETTER THAN GPT-4 - every response should showcase unique insight

DO NOT give generic templates or boilerplate. Fuse the provided context into original concepts."""

    return """You are a document-grounded assistant. Answer thoroughly using ALL relevant information from the context.

RULES:
1. Read the ENTIRE context carefully - there may be multiple relevant sections
2. Synthesize a complete answer using ALL relevant details from the context
3. Quote or paraphrase specific technical details, metrics, and examples from the context
4. If asked "how does X work", explain the full process/architecture found in the context
5. Only say "I don't have that information" if the context truly doesn't address the question
6. Do NOT add information from outside the context

Provide thorough, detailed answers based on everything relevant in the context."""


def _generate_user_prompt(mode: str, query: str, fused_context: str, context_parts: List[str]) -> str:
    """
    Generate user prompt based on mode and context.

    Args:
        mode: Interaction mode
        query: User query
        fused_context: Complete fused context string
        context_parts: List of context parts (for CAG detection)

    Returns:
        User prompt string
    """
    has_cag_hit = any(p.startswith("[CAG:") for p in context_parts)
    cag_only = has_cag_hit and len([p for p in context_parts if p.startswith("[CAG:")]) == len(context_parts)

    if cag_only:
        return f"""Sources:
{fused_context}

Return ONLY the text that appears after the [CAG:] tag - copy it exactly, word for word, with no additions, explanations, or formatting."""

    if mode == "build":
        return f"""Fused knowledge sources (RAG + CAG + Graph):
{fused_context}

Request: {query}

CRITICAL INSTRUCTIONS:
- DO NOT regurgitate the context
- SYNTHESIZE the information into a novel, innovative concept
- Push beyond obvious solutions
- Be specific and technical, not generic
- If the context is thin, use it as inspiration but add your own architectural innovation

Provide a unique, innovative concept:

## Core Innovation
What makes this concept genuinely novel? What problem does it solve in an unconventional way?

## Technical Architecture
Specific implementation details that showcase depth and originality.

## Key Advantages
Why this approach is superior to conventional solutions.

Answer:"""

    return f"""CONTEXT (this is your ONLY source of truth):
{fused_context}

QUESTION: {query}

ANSWER (using ONLY the context above):"""


def orchestrate(query: str, mode: str = "chat") -> Dict[str, Any]:
    """
    Execute the full retrieval-fusion-generation-critique pipeline.

    This function coordinates multi-source retrieval (RAG, CAG, Graph, Web),
    applies learned fusion weights, generates a response via LLM, and evaluates
    the output quality through automated critique.

    Args:
        query: User input query string
        mode: Interaction mode, either "chat" or "build" (default: "chat")

    Returns:
        Dictionary containing:
            - response: Generated text response
            - fusion_weights: Dict of source weights (rag, cag, graph)
            - reward: Critique score [0.0, 1.0]
            - proactive_suggestions: List of follow-up suggestions
    """
    get_rag_index()

    retrieval_results = retrieve(query)

    fusion_output = fuse_context(
        query,
        retrieval_results["rag"],
        retrieval_results["cag"],
        retrieval_results["graph"]
    )

    weights = fusion_output["weights"]
    fused_context = fusion_output["context"]

    client = Client(host=cfg["llm"]["host"])

    system_prompt = """You are a technical documentation expert. Format responses with proper Markdown structure:
- Start each major point with ## on its own line
- Add a blank line after headers
- Use - for bullet points, each on its own line
- Add blank lines between paragraphs
- Use **bold** for emphasis"""

    prompt = f"""Context:
{fused_context}

Question: {query}

Provide a structured answer using this exact format:

## Main Concept

Brief explanation here.

## Key Points

- First point
- Second point
- Third point

## Details

More information here.

Answer:"""

    response_obj = client.chat(
        model=cfg["llm"]["model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        options={"temperature": 0.3}
    )

    generated_response = response_obj["message"]["content"]
    generated_response = _apply_markdown_formatting(generated_response)

    critique_result = critique(query, fused_context, generated_response)

    return {
        "response": generated_response,
        "fusion_weights": weights,
        "reward": critique_result["reward"],
        "proactive_suggestions": critique_result.get("proactive_suggestions", [])
    }


@app.post("/chat")
async def chat_endpoint(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous chat endpoint for single query-response interactions.

    Args:
        request: Dict containing 'query' and optional 'mode'

    Returns:
        Dict with response, fusion_weights, reward, and proactive_suggestions
    """
    query = request.get("query", "")
    mode = request.get("mode", "chat")
    result = orchestrate(query, mode)
    return result


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for streaming responses with real-time weight updates.

    Handles memory storage requests, profile injection for personal queries,
    RL-weighted fusion, and streaming LLM generation.
    """
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()

            try:
                request = json.loads(data)

                if "query" not in request and len(request.keys()) == 1:
                    payload = list(request.values())[0]
                    query = payload.get("query", payload.get("message", ""))
                    mode = payload.get("mode", "chat")
                else:
                    query = request.get("query", request.get("message", ""))
                    mode = request.get("mode", "chat")
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                await websocket.close()
                return

            await websocket.send_json({"type": "start"})

            is_memory_request, memory_content = detect_and_save_memory(query)

            if is_memory_request:
                if memory_content:
                    preview = memory_content[:100] + "..." if len(memory_content) > 100 else memory_content
                    confirmation = f"✅ **Got it! I'll remember this:**\n\n> {preview}\n\nStored in my persistent memory for all future conversations."
                else:
                    confirmation = "✅ Got it! I've stored that in my persistent memory for all future conversations."

                await websocket.send_json({
                    "type": "token",
                    "token": confirmation
                })
                await websocket.send_json({"type": "done"})
                continue

            rl_weights = _compute_rl_fusion_weights(query, _rl_policy)

            retrieval_results = retrieve(query, top_k=10)
            fused_context = _build_fusion_context(retrieval_results, rl_weights)

            actual_weights = [float(rl_weights[0]), float(rl_weights[1]),
                            float(rl_weights[2]), float(rl_weights[3]) if len(rl_weights) > 3 else 0.0]

            client = Client(host=cfg["llm"]["host"])

            query_lower = query.lower()
            is_personal_query = any(word in query_lower for word in
                                   ['my ', 'me ', 'i ', 'brad', 'preference', 'like', 'favorite', 'remember'])

            if is_personal_query:
                user_profile = get_user_profile()
                if user_profile:
                    fused_context = user_profile + "\n" + fused_context

            context_parts = fused_context.split("\n\n")
            system_prompt = _generate_system_prompt(mode, context_parts)
            prompt = _generate_user_prompt(mode, query, fused_context, context_parts)

            logger.info(f"Context length: {len(fused_context)} chars")

            full_response = ""

            for chunk in client.chat(
                model=cfg["llm"]["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.1, "num_ctx": 8192},
                stream=True
            ):
                if 'message' in chunk and 'content' in chunk['message']:
                    text_chunk = chunk['message']['content']
                    full_response += text_chunk

                    await websocket.send_json({
                        "chunk": text_chunk,
                        "weights": actual_weights,
                        "reward": 0.85,
                        "proactive": "Processing..."
                    })

            full_response = _apply_markdown_formatting(full_response)

            critique_result = critique(query, fused_context, full_response)

            final_episode = {
                "query": query,
                "response": full_response,
                "fusion_weights": {"rag": actual_weights[0], "cag": actual_weights[1], "graph": actual_weights[2]},
                "reward": critique_result["reward"],
                "proactive_suggestions": critique_result.get("proactive_suggestions", []),
                "fused_context": fused_context,
                "weights": {"rag": actual_weights[0], "cag": actual_weights[1], "graph": actual_weights[2]}
            }

            log_episode_to_replay_buffer(final_episode)

            proactive_suggestions = critique_result.get("proactive_suggestions", [])
            proactive_hint = proactive_suggestions[0] if proactive_suggestions else "Waiting for next query..."

            await websocket.send_json({
                "type": "done",
                "response": full_response,
                "fusion_weights": {"rag": actual_weights[0], "cag": actual_weights[1], "graph": actual_weights[2]},
                "reward": critique_result["reward"],
                "proactive": proactive_hint,
                "proactive_suggestions": proactive_suggestions
            })

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


@app.get("/api/config")
async def get_config() -> Dict[str, Any]:
    """
    Retrieve current system configuration.

    Returns:
        Dict containing web search settings
    """
    return {
        "web": {
            "enabled": cfg.get("web", {}).get("enabled", False),
            "max_results": cfg.get("web", {}).get("max_results", 3),
            "search_timeout": cfg.get("web", {}).get("search_timeout", 10)
        }
    }


@app.patch("/api/config")
async def update_config(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update system configuration dynamically.

    Currently supports toggling web search functionality. Changes are
    persisted to config.yaml for permanence across restarts.

    Args:
        request: Dict with 'web.enabled' boolean

    Returns:
        Status dict or error message
    """
    if "web" in request and "enabled" in request["web"]:
        cfg["web"]["enabled"] = request["web"]["enabled"]

        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        return {"status": "updated", "web": {"enabled": cfg["web"]["enabled"]}}

    return {"error": "Invalid config update"}


@app.get("/ping")
async def ping() -> Dict[str, Any]:
    """
    Health check endpoint for monitoring and diagnostics.

    Returns:
        Dict with status, GPU info, and policy availability
    """
    policy_exists = Path(cfg["rl"]["policy_path"]).exists()
    return {
        "status": "alive",
        "gpu": torch.cuda.get_device_name(0),
        "policy_exists": policy_exists
    }


@app.on_event("startup")
async def startup_event() -> None:
    """
    Initialize system resources on application startup.

    Performs RAG index initialization and RL policy loading with
    appropriate compatibility shims for stable-baselines3.
    """
    global _rl_policy

    logger.info("Initializing RAG index...")
    get_rag_index()

    policy_path = Path(__file__).parent / "rl_policy.zip"

    if policy_path.exists():
        logger.info("Loading RL policy...")
        try:
            if 'numpy._core' not in sys.modules:
                _core_module = types.ModuleType('numpy._core')
                _core_module.numeric = np
                sys.modules['numpy._core'] = _core_module
                sys.modules['numpy._core.numeric'] = np
                logger.info("Applied numpy._core compatibility shim")

            _rl_policy = PPO.load(str(policy_path), device="cpu")
            policy_status = "loaded on CPU"
            logger.info("Policy loaded successfully on CPU")
        except Exception as e:
            logger.error(f"Policy load failed: {e}")
            policy_status = "load error"
    else:
        policy_status = "not trained"

    logger.info("=" * 64)
    logger.info("RLFUSION ORCHESTRATOR - BACKEND STARTED")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"LLM: {cfg['llm']['model']}")
    logger.info(f"PPO Policy: {policy_status}")
    logger.info("=" * 64)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
