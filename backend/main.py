# backend/main.py
# Author: Bradley R. Kinnard
# The main event - FastAPI server that orchestrates RAG/CAG/graph fusion

import os
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from backend.config import cfg
from backend.core.retrievers import retrieve
from backend.core.fusion import fuse_context
from backend.core.critique import critique
from ollama import Client
import json

# No GPU? No service. This thing needs CUDA or it's not gonna work.
assert torch.cuda.is_available(), "ERROR: CUDA not detected. GPU is required for RLFO."
device = torch.device("cuda")

print(f"RLFO running on GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

# TODO: break config into separate modules when this inevitably grows
# TODO: add proper request logging (not just print statements)

app = FastAPI(title="RLFusion Orchestrator", version="0.1.0")

# CORS is completely open right now - lock it down before going live
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # yeah I know, dangerous
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def orchestrate(query: str, mode: str = "chat") -> dict:
    """
    The main pipeline - retrieves from three sources, fuses them together,
    Runs the full retrieve -> fuse -> generate -> critique pipeline.

    Returns dict with response, weights, reward, and suggestions.
    """
    from backend.core.retrievers import get_rag_index
    get_rag_index()  # Force index build on first query

    # === FINAL WORKING FUSION THAT CANNOT BE IGNORED ===
    from backend.core import retrievers

    # Force all three retrievals — if any of these lines are missing, that pillar is dead
    rag   = retrievers.retrieve_rag(query)
    cag   = retrievers.retrieve_cag(query)
    graph = retrievers.retrieve_graph(query)

    # Debug print so you can see it in your terminal
    print(f"\n=== RETRIEVAL DEBUG ===")
    print(f"RAG hits : {len(rag)}")
    print(f"CAG hits : {len(cag)} → {cag}")
    print(f"Graph hits: {len(graph)}")
    print(f"======================\n")

    # Strict production thresholds
    context_parts = []
    for r in rag:
        if r["score"] >= 0.65: context_parts.append(f"[RAG:{r['score']:.2f}] {r['text']}")
    for c in cag:
        if c["score"] >= 0.85: context_parts.append(f"[CAG:{c['score']:.2f}] {c['text']}")
    for g in graph:
        if g["score"] >= 0.70: context_parts.append(f"[GRAPH:{g['score']:.2f}] {g['text']}")

    final_context = "\n\n".join(context_parts) if context_parts else "No high-confidence sources."
    fused_context = final_context
    # =================================================

    # Step 1: retrieve with default weights
    retrieval_results = retrieve(query)

    # Step 2: fuse the three result lists
    fusion_output = fuse_context(
        query,
        retrieval_results["rag"],
        retrieval_results["cag"],
        retrieval_results["graph"]
    )

    weights = fusion_output["weights"]

    # Step 3: generate response with Ollama
    client = Client(host=cfg["llm"]["host"])

    # Force the LLM to structure its response properly
    system_prompt = """You are a technical documentation expert. You MUST format responses with proper Markdown structure:
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
        options={"temperature": 0.3}  # Lower temp for more structured output
    )

    generated_response = response_obj["message"]["content"]

    # Simple, safe Markdown formatting fix
    import re

    # Add line breaks before headers
    generated_response = generated_response.replace('## ', '\n\n## ')
    generated_response = generated_response.replace('# ', '\n\n# ')

    # Add line breaks before bullet points (various patterns)
    generated_response = re.sub(r'([.!?])\s*-\s', r'\1\n\n- ', generated_response)
    generated_response = re.sub(r':\s*-\s', r':\n\n- ', generated_response)

    # Break up long sentences
    generated_response = re.sub(r'\.\s+([A-Z])', r'.\n\n\1', generated_response)

    # Clean up multiple newlines
    generated_response = re.sub(r'\n{3,}', '\n\n', generated_response)
    generated_response = generated_response.strip()    # Step 4: critique the output
    critique_result = critique(query, fused_context, generated_response)

    # Return exact schema from rlfo.pdf
    return {
        "response": generated_response,
        "fusion_weights": weights,
        "reward": critique_result["reward"],
        "proactive_suggestions": critique_result.get("proactive_suggestions", [])
    }

@app.post("/chat")
async def chat_endpoint(request: dict):
    query = request.get("query", "")
    mode = request.get("mode", "chat")
    result = orchestrate(query, mode)
    return result

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                    request = json.loads(data)
                    # Accept both the old raw format and the future proper format
                    if "query" not in request and len(request.keys()) == 1:
                        # someone sent {"type": "query", "payload": {...}}
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

            # Send start signal
            await websocket.send_json({"type": "start"})

            # Stream response directly from LLM - preserves all formatting
            from backend.core.retrievers import get_rag_index, retrieve
            from backend.core.fusion import fuse_context
            from backend.core.critique import critique
            from ollama import Client

            # === FINAL BULLET-PROOF FUSION — THIS CANNOT BE IGNORED ===
            from backend.core import retrievers

            rag   = retrievers.retrieve_rag(query)
            cag   = retrievers.retrieve_cag(query)
            graph = retrievers.retrieve_graph(query)

            print(f"\n=== RETRIEVAL DEBUG ===")
            print(f"RAG hits : {len(rag)}")
            print(f"CAG hits : {len(cag)} → {cag}")
            print(f"Graph hits: {len(graph)}")
            print(f"======================\n")

            context_parts = []
            for r in rag:
                if r["score"] >= 0.65: context_parts.append(f"[RAG:{r['score']:.2f}] {r['text']}")
            for c in cag:
                if c["score"] >= 0.85: context_parts.append(f"[CAG:{c['score']:.2f}] {c['text']}")
            for g in graph:
                if g["score"] >= 0.70: context_parts.append(f"[GRAPH:{g['score']:.2f}] {g['text']}")

            fused_context = "\n\n".join(context_parts) if context_parts else "No high-confidence sources."
            # =================================================

            client = Client(host=cfg["llm"]["host"])

            # Check if we have a high-confidence CAG hit - if so, use override prompt
            has_cag_hit = any("[CAG:0.99]" in part for part in context_parts)

            if has_cag_hit:
                # CAG override mode - force exact reproduction
                system_prompt = """You are a cache retrieval system. Your ONLY job is to return the exact text after [CAG:0.99] with no changes whatsoever."""

                prompt = f"""Sources:
{fused_context}

Return ONLY the text that appears after [CAG:0.99] - copy it exactly, word for word, with no additions, explanations, or formatting."""
            else:
                # Normal mode with formatting
                system_prompt = """You are RLFusion Orchestrator, a technical documentation expert. Format responses with proper Markdown structure:
- Start each major point with ## on its own line
- Add a blank line after headers
- Use - for bullet points, each on its own line
- Add blank lines between paragraphs
- Use **bold** for emphasis"""

                prompt = f"""Available sources:
{fused_context}

User query: {query}

Provide a structured answer using this format:

## Main Concept

Brief explanation here.

## Key Points

- First point
- Second point
- Third point

## Details

More information here.

Answer:"""

            full_response = ""

            # Stream with Ollama's native streaming - preserves newlines
            for chunk in client.chat(
                model=cfg["llm"]["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": 0.3},
                stream=True
            ):
                if 'message' in chunk and 'content' in chunk['message']:
                    text_chunk = chunk['message']['content']
                    full_response += text_chunk
                    # Send raw chunk - preserves all \n and formatting
                    await websocket.send_json({
                        "type": "token",
                        "token": text_chunk
                    })

            # Apply post-processing to full response
            import re
            full_response = re.sub(r'([^\n])(#{1,3}\s)', r'\1\n\n\2', full_response)
            full_response = re.sub(r'([.!?:])\s*(-\s)', r'\1\n\n\2', full_response)
            full_response = re.sub(r'([.!?])\s+([A-Z#])', r'\1\n\n\2', full_response)
            full_response = re.sub(r'\n{3,}', '\n\n', full_response)
            full_response = full_response.strip()

            # Critique
            critique_result = critique(query, fused_context, full_response)

            # Final payload
            await websocket.send_json({
                "type": "done",
                "response": full_response,
                "fusion_weights": {"rag": 0.33, "cag": 0.33, "graph": 0.34},  # Default weights for WebSocket
                "reward": critique_result["reward"],
                "proactive_suggestions": critique_result.get("proactive_suggestions", [])
            })
    except WebSocketDisconnect:
        pass

@app.get("/ping")
async def ping():
    """Health check endpoint"""
    policy_exists = Path(cfg["rl"]["policy_path"]).exists()
    return {
        "status": "alive",
        "gpu": torch.cuda.get_device_name(0),
        "policy_exists": policy_exists
    }

@app.on_event("startup")
async def startup_event():
    """Print startup banner with system info"""
    # Force RAG index initialization on startup
    from backend.core.retrievers import get_rag_index
    print("Initializing RAG index...")
    get_rag_index()

    # Check if policy exists in project root
    policy_path = Path("rl_policy.zip")
    policy_status = "loaded" if policy_path.exists() else "not trained"
    print("\n" + "="*64)
    print("RLFUSION ORCHESTRATOR - BACKEND STARTED")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"LLM: {cfg['llm']['model']}")
    print(f"PPO Policy: {policy_status}")
    print(f"orchestrate() API live - exact spec from rlfo.pdf page 5")
    print("="*64 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
