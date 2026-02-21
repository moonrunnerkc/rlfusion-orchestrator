# Author: Bradley R. Kinnard
"""Supervised fine-tuning pipeline with LoRA for domain-adapted LLM training.

Phase 6: extracts high-reward episodes from the replay buffer, formats them
as instruction-following examples, and trains a LoRA adapter via Hugging Face
SFTTrainer. Trained models can be exported to GGUF for Ollama deployment.

Key pieces:
    load_training_episodes()  - pull high-reward data from replay DB
    prepare_sft_dataset()     - format episodes into train/val splits
    run_sft()                 - full LoRA training loop with checkpointing
    export_gguf()             - convert adapter to GGUF for local inference
    SFTJobConfig / SFTJobResult - typed config and result structs
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import TypedDict

from backend.config import cfg, PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config defaults from config.yaml ─────────────────────────────────────

_ft_cfg = cfg.get("fine_tuning", {})
_DEFAULT_BASE_MODEL = str(_ft_cfg.get("base_model", "meta-llama/Llama-3.1-8B"))
_DEFAULT_LORA_RANK = int(_ft_cfg.get("lora_rank", 16))
_DEFAULT_LORA_ALPHA = int(_ft_cfg.get("lora_alpha", 32))
_DEFAULT_LORA_DROPOUT = float(_ft_cfg.get("lora_dropout", 0.05))
_DEFAULT_LR = float(_ft_cfg.get("learning_rate", 2e-4))
_DEFAULT_EPOCHS = int(_ft_cfg.get("num_epochs", 3))
_DEFAULT_BATCH_SIZE = int(_ft_cfg.get("batch_size", 4))
_DEFAULT_MAX_SEQ_LEN = int(_ft_cfg.get("max_seq_length", 2048))
_DEFAULT_MIN_REWARD = float(_ft_cfg.get("min_reward_threshold", 0.8))
_DEFAULT_MAX_EPISODES = int(_ft_cfg.get("max_training_episodes", 5000))
_DEFAULT_VAL_SPLIT = float(_ft_cfg.get("val_split", 0.1))
_DEFAULT_OUTPUT_DIR = str(_ft_cfg.get("output_dir", "models/fine_tuned"))


# ── Typed structures ─────────────────────────────────────────────────────

class TrainingEpisode(TypedDict):
    """Single episode extracted from the replay buffer."""
    query: str
    response: str
    reward: float
    rag_weight: float
    cag_weight: float
    graph_weight: float


class SFTJobConfig(TypedDict):
    """Configuration for a fine-tuning job."""
    base_model: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    learning_rate: float
    num_epochs: int
    batch_size: int
    max_seq_length: int
    min_reward: float
    max_episodes: int
    val_split: float
    output_dir: str


class SFTJobResult(TypedDict):
    """Result of a completed fine-tuning job."""
    status: str
    model_path: str
    episodes_used: int
    train_size: int
    val_size: int
    elapsed_secs: float
    lora_rank: int
    lora_alpha: int
    base_model: str
    error: str


def default_config() -> SFTJobConfig:
    """Build a config from yaml defaults. Useful for CLI and endpoint."""
    return SFTJobConfig(
        base_model=_DEFAULT_BASE_MODEL,
        lora_rank=_DEFAULT_LORA_RANK,
        lora_alpha=_DEFAULT_LORA_ALPHA,
        lora_dropout=_DEFAULT_LORA_DROPOUT,
        learning_rate=_DEFAULT_LR,
        num_epochs=_DEFAULT_EPOCHS,
        batch_size=_DEFAULT_BATCH_SIZE,
        max_seq_length=_DEFAULT_MAX_SEQ_LEN,
        min_reward=_DEFAULT_MIN_REWARD,
        max_episodes=_DEFAULT_MAX_EPISODES,
        val_split=_DEFAULT_VAL_SPLIT,
        output_dir=_DEFAULT_OUTPUT_DIR,
    )


# ── Data loading ─────────────────────────────────────────────────────────

def load_training_episodes(
    min_reward: float = _DEFAULT_MIN_REWARD,
    max_episodes: int = _DEFAULT_MAX_EPISODES,
    db_path: str | None = None,
) -> list[TrainingEpisode]:
    """Pull high-reward episodes from the replay buffer DB.

    Selects episodes with reward >= min_reward, ordered by reward descending.
    Returns empty list if DB or table doesn't exist.
    """
    resolved = Path(db_path) if db_path else PROJECT_ROOT / "db" / "rlfo_cache.db"
    if not resolved.exists():
        logger.warning("Replay DB not found at %s", resolved)
        return []

    conn = sqlite3.connect(str(resolved))
    cursor = conn.cursor()

    # verify episodes table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='episodes'"
    )
    if not cursor.fetchone():
        conn.close()
        logger.warning("No 'episodes' table in %s", resolved)
        return []

    cursor.execute(
        "SELECT query, response, reward, rag_weight, cag_weight, graph_weight "
        "FROM episodes WHERE reward >= ? ORDER BY reward DESC LIMIT ?",
        (min_reward, max_episodes),
    )
    rows = cursor.fetchall()
    conn.close()

    episodes = []
    for query, response, reward, rag_w, cag_w, graph_w in rows:
        if not query or not response:
            continue
        episodes.append(TrainingEpisode(
            query=str(query),
            response=str(response)[:4000],
            reward=float(reward or 0.0),
            rag_weight=float(rag_w or 0.0),
            cag_weight=float(cag_w or 0.0),
            graph_weight=float(graph_w or 0.0),
        ))

    logger.info(
        "Loaded %d episodes (reward >= %.2f) from %s",
        len(episodes), min_reward, resolved,
    )
    return episodes


def prepare_sft_dataset(
    episodes: list[TrainingEpisode],
    val_split: float = _DEFAULT_VAL_SPLIT,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Format episodes as instruction-following examples and split train/val.

    Each example has 'instruction' (the query) and 'output' (the response).
    The split is deterministic: last N% of episodes become validation.
    """
    if not episodes:
        return [], []

    if not 0.0 <= val_split < 1.0:
        raise ValueError(f"val_split must be in [0, 1), got {val_split}")

    formatted = []
    for ep in episodes:
        formatted.append({
            "instruction": ep["query"],
            "output": ep["response"],
        })

    # deterministic split: tail becomes validation
    val_count = max(1, int(len(formatted) * val_split)) if val_split > 0 else 0
    train_data = formatted[:len(formatted) - val_count] if val_count else formatted
    val_data = formatted[len(formatted) - val_count:] if val_count else []

    logger.info("Dataset split: %d train, %d val", len(train_data), len(val_data))
    return train_data, val_data


def _format_prompt(example: dict[str, str]) -> str:
    """Convert an instruction/output pair into a single prompt string."""
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']}"
    )


# ── Training ─────────────────────────────────────────────────────────────

def run_sft(job_config: SFTJobConfig | None = None) -> SFTJobResult:
    """Run LoRA fine-tuning on high-reward replay episodes.

    Uses Hugging Face SFTTrainer with PEFT LoRA adapters. The base model
    is loaded in 4-bit quantization for memory efficiency. Training data
    comes from the replay buffer filtered by reward threshold.
    """
    config = job_config or default_config()
    t0 = time.time()

    output_path = PROJECT_ROOT / config["output_dir"]
    output_path.mkdir(parents=True, exist_ok=True)

    # load training data
    episodes = load_training_episodes(
        min_reward=config["min_reward"],
        max_episodes=config["max_episodes"],
    )

    if len(episodes) < 10:
        return SFTJobResult(
            status="insufficient_data",
            model_path="",
            episodes_used=len(episodes),
            train_size=0,
            val_size=0,
            elapsed_secs=round(time.time() - t0, 2),
            lora_rank=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            base_model=config["base_model"],
            error=f"Need at least 10 episodes, found {len(episodes)}",
        )

    train_data, val_data = prepare_sft_dataset(episodes, config["val_split"])

    # lazy imports: these are heavy and only needed during training
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl.trainer.sft_trainer import SFTTrainer
    except ImportError as exc:
        return SFTJobResult(
            status="missing_dependency",
            model_path="",
            episodes_used=len(episodes),
            train_size=len(train_data),
            val_size=len(val_data),
            elapsed_secs=round(time.time() - t0, 2),
            lora_rank=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            base_model=config["base_model"],
            error=f"Missing package: {exc}. Install with: pip install peft trl transformers datasets",
        )

    try:
        logger.info(
            "Starting SFT: base=%s, lora_r=%d, alpha=%d, episodes=%d",
            config["base_model"], config["lora_rank"],
            config["lora_alpha"], len(episodes),
        )

        # 4-bit quantization config for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # LoRA adapter config
        lora_config = LoraConfig(
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "LoRA params: %d trainable / %d total (%.2f%%)",
            trainable, total, 100 * trainable / total,
        )

        # format datasets
        train_texts = [_format_prompt(ex) for ex in train_data]
        val_texts = [_format_prompt(ex) for ex in val_data] if val_data else None

        train_dataset = Dataset.from_dict({"text": train_texts})
        eval_dataset = Dataset.from_dict({"text": val_texts}) if val_texts else None

        # training arguments
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=config["num_epochs"],
            per_device_train_batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset else "no",
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=4,
            report_to="none",
        )

        from peft import PeftModel as _PeftModel
        trainer = SFTTrainer(
            model=model if isinstance(model, _PeftModel) else model,  # type: ignore[arg-type]
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )

        trainer.train()

        # save adapter and tokenizer
        adapter_path = output_path / "adapter"
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))

        elapsed = round(time.time() - t0, 2)
        logger.info("SFT complete: adapter saved to %s (%.1fs)", adapter_path, elapsed)

        return SFTJobResult(
            status="completed",
            model_path=str(adapter_path),
            episodes_used=len(episodes),
            train_size=len(train_data),
            val_size=len(val_data),
            elapsed_secs=elapsed,
            lora_rank=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            base_model=config["base_model"],
            error="",
        )

    except (RuntimeError, ValueError, OSError, KeyError, torch.cuda.CudaError) as exc:
        elapsed = round(time.time() - t0, 2)
        logger.error("SFT failed: %s", exc, exc_info=True)
        return SFTJobResult(
            status="failed",
            model_path="",
            episodes_used=len(episodes),
            train_size=len(train_data),
            val_size=len(val_data),
            elapsed_secs=elapsed,
            lora_rank=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            base_model=config["base_model"],
            error=str(exc),
        )


# ── GGUF export ──────────────────────────────────────────────────────────

def export_gguf(
    adapter_path: str,
    output_path: str | None = None,
    quantization: str = "Q4_K_M",
) -> Path:
    """Convert a LoRA adapter to GGUF format for Ollama deployment.

    Requires llama-cpp-python or llama.cpp convert tooling to be available.
    The adapter is first merged with the base model, then quantized.
    """
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_dir}")

    resolved_output = Path(output_path) if output_path else (
        adapter_dir.parent / f"model-{quantization.lower()}.gguf"
    )

    try:
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
        import torch

        logger.info("Merging LoRA adapter from %s", adapter_dir)
        model = AutoPeftModelForCausalLM.from_pretrained(
            str(adapter_dir),
            torch_dtype=torch.float16,
            device_map="auto",
        )
        merged = model.merge_and_unload()

        # save merged model for GGUF conversion
        merged_dir = adapter_dir.parent / "merged"
        merged.save_pretrained(str(merged_dir))
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(merged_dir))

        logger.info("Merged model saved to %s", merged_dir)
        logger.info(
            "To convert to GGUF, run: "
            "python llama.cpp/convert_hf_to_gguf.py %s --outtype %s --outfile %s",
            merged_dir, quantization.lower(), resolved_output,
        )

        return resolved_output

    except ImportError as exc:
        raise ImportError(
            f"GGUF export requires peft and transformers: {exc}"
        ) from exc


# ── CLI entry point ──────────────────────────────────────────────────────

def main() -> None:
    """CLI for running fine-tuning jobs."""
    import argparse
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))

    parser = argparse.ArgumentParser(
        description="Run LoRA fine-tuning on high-reward replay episodes"
    )
    parser.add_argument(
        "--base-model", default=_DEFAULT_BASE_MODEL,
        help="HuggingFace model ID for the base model",
    )
    parser.add_argument("--lora-rank", type=int, default=_DEFAULT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=_DEFAULT_LORA_ALPHA)
    parser.add_argument("--lr", type=float, default=_DEFAULT_LR)
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument("--min-reward", type=float, default=_DEFAULT_MIN_REWARD)
    parser.add_argument("--max-episodes", type=int, default=_DEFAULT_MAX_EPISODES)
    parser.add_argument(
        "--output", default=_DEFAULT_OUTPUT_DIR,
        help="Output directory for trained adapter",
    )
    parser.add_argument(
        "--export-gguf", action="store_true",
        help="Export merged model to GGUF after training",
    )
    args = parser.parse_args()

    job_config = SFTJobConfig(
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=_DEFAULT_LORA_DROPOUT,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_length=_DEFAULT_MAX_SEQ_LEN,
        min_reward=args.min_reward,
        max_episodes=args.max_episodes,
        val_split=_DEFAULT_VAL_SPLIT,
        output_dir=args.output,
    )

    result = run_sft(job_config)
    print(json.dumps(result, indent=2))

    if result["status"] == "completed" and args.export_gguf:
        try:
            gguf_path = export_gguf(result["model_path"])
            print(f"GGUF export target: {gguf_path}")
        except (ImportError, FileNotFoundError, RuntimeError, OSError) as exc:
            print(f"GGUF export failed: {exc}")


if __name__ == "__main__":
    main()
