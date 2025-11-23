# backend/main.py
# Author: Bradley R. Kinnard
# RLFO backend - FastAPI server for the fusion orchestrator

import os
import torch
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# GPU required - if you don't have CUDA this won't run
assert torch.cuda.is_available(), "ERROR: CUDA not detected. GPU is required for RLFO."
device = torch.device("cuda")

print(f"RLFO running on GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

# config loading
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

cfg = config

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

@app.get("/ping")
async def ping():
    """Health check endpoint"""
    policy_exists = Path(cfg["rl"]["policy_path"]).exists()
    return {
        "status": "alive",
        "gpu": torch.cuda.get_device_name(0),
        "policy_exists": policy_exists
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main orchestration WebSocket - query handling goes here eventually"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # TODO: wire up actual fusion orchestration
            await websocket.send_text(f"Echo: {data} (orchestration coming soon)")
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket")

@app.on_event("startup")
async def startup_event():
    """Print startup banner with system info"""
    policy_status = "loaded" if Path(cfg["rl"]["policy_path"]).exists() else "not trained"
    print("\n" + "="*64)
    print("RLFUSION ORCHESTRATOR - BACKEND STARTED")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"LLM: {cfg['llm']['model']}")
    print(f"PPO Policy: {policy_status}")
    print("="*64 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
