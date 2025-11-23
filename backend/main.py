# backend/main.py
# Author: Bradley R. Kinnard
# RLFO backend - FastAPI server for the fusion orchestrator

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

# GPU required - if you don't have CUDA this won't run
assert torch.cuda.is_available(), "ERROR: CUDA not detected. GPU is required for RLFO."
device = torch.device("cuda")

print(f"RLFO running on GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

# TODO: split cfg into separate modules when this gets bigger
# TODO: add request logging middleware

app = FastAPI(title="RLFusion Orchestrator", version="0.1.0")

# CORS wide open for now - tighten this before production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def orchestrate(query: str, mode: str = "chat") -> dict:
    """
    Main orchestration loop from rlfo.pdf page 5.
    Runs the full retrieve -> fuse -> generate -> critique pipeline.

    Returns dict with response, weights, reward, and suggestions.
    """
    # Step 1: retrieve with default weights
    retrieval_results = retrieve(query)

    # Step 2: fuse the three result lists
    fusion_output = fuse_context(
        query,
        retrieval_results["rag"],
        retrieval_results["cag"],
        retrieval_results["graph"]
    )

    fused_context = fusion_output["fused_context"]
    weights = fusion_output["weights"]

    # Step 3: generate response with Ollama
    client = Client(host=cfg["llm"]["host"])
    prompt = f"Using only this context:\n\n{fused_context}\n\nAnswer: {query}"

    response_obj = client.chat(
        model=cfg["llm"]["model"],
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.72}
    )

    generated_response = response_obj["message"]["content"]

    # Step 4: critique the output
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
            request = json.loads(data)
            query = request.get("query", "")
            mode = request.get("mode", "chat")

            # Send start signal
            await websocket.send_json({"type": "start"})

            result = orchestrate(query, mode)

            # Stream the response token-by-token (simulate for now)
            for token in result["response"].split():
                await websocket.send_json({
                    "type": "token",
                    "token": token + " "
                })

            # Final payload
            await websocket.send_json({
                "type": "done",
                "response": result["response"],
                "fusion_weights": result["fusion_weights"],
                "reward": result["reward"],
                "proactive_suggestions": result["proactive_suggestions"]
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
    policy_status = "loaded" if Path(cfg["rl"]["policy_path"]).exists() else "not trained"
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
