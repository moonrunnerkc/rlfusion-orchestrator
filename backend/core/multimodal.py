# Author: Bradley R. Kinnard
"""Multi-modal embedding pipeline: CLIP-based image/text retrieval,
PDF image extraction via PyMuPDF, and optional Ollama vision captioning.

Phase 7 of the RLFusion upgrade plan. Maintains a separate FAISS index
for CLIP embeddings (512-dim) alongside the existing BGE text index (384-dim).
All heavy models load lazily on first use.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import TypedDict

import faiss
import numpy as np

from backend.config import PROJECT_ROOT, cfg
from backend.core.utils import ensure_path

logger = logging.getLogger(__name__)

# ── Optional: CLIP via transformers ──────────────────────────────────
_HAS_CLIP = False
try:
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    _HAS_CLIP = True
except ImportError:
    logger.info("transformers/Pillow not installed; multimodal features disabled")

# ── Optional: PyMuPDF for PDF image extraction ───────────────────────
_HAS_FITZ = False
try:
    import fitz  # PyMuPDF

    _HAS_FITZ = True
except ImportError:
    logger.info("PyMuPDF not installed; PDF image extraction unavailable")


# ── Config defaults ──────────────────────────────────────────────────

def _mm_cfg() -> dict[str, object]:
    """Pull multimodal config with safe defaults."""
    return cfg.get("multimodal", {})


def _is_enabled() -> bool:
    return bool(_mm_cfg().get("enabled", False))


def _clip_model_name() -> str:
    return str(_mm_cfg().get("clip_model", "openai/clip-vit-base-patch32"))


def _vision_model_name() -> str:
    return str(_mm_cfg().get("vision_model", "llava"))


def _image_store_path() -> Path:
    p = PROJECT_ROOT / "data" / "images"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _image_index_path() -> Path:
    p = PROJECT_ROOT / "indexes"
    p.mkdir(parents=True, exist_ok=True)
    return p / "image_index.faiss"


def _image_metadata_path() -> Path:
    p = PROJECT_ROOT / "indexes"
    p.mkdir(parents=True, exist_ok=True)
    return p / "image_metadata.json"


_SUPPORTED_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"})

# CLIP produces 512-dim vectors
_CLIP_DIM = 512


# ── TypedDicts ───────────────────────────────────────────────────────

class ImageChunk(TypedDict, total=False):
    """Metadata for an extracted or uploaded image."""
    image_id: str
    image_path: str
    source: str
    caption: str
    content_type: str
    width: int
    height: int
    page_number: int | None


class ImageSearchResult(TypedDict, total=False):
    """Single result from cross-modal image search."""
    text: str
    score: float
    source: str
    id: str
    image_path: str
    caption: str
    width: int
    height: int


# ── Lazy CLIP singleton ─────────────────────────────────────────────

_clip_cache: dict[str, object] = {"model": None, "processor": None, "loaded": False}


def _load_clip() -> tuple[object, object]:
    """Load CLIP model + processor on first call. Returns (model, processor)."""
    if _clip_cache["loaded"]:
        return _clip_cache["model"], _clip_cache["processor"]  # type: ignore[return-value]

    if not _HAS_CLIP:
        raise RuntimeError(
            "CLIP unavailable: install transformers and Pillow "
            "(pip install transformers Pillow)"
        )

    model_name = _clip_model_name()
    logger.info("Loading CLIP model: %s", model_name)
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    _clip_cache["model"] = model
    _clip_cache["processor"] = processor
    _clip_cache["loaded"] = True
    logger.info("CLIP model loaded (%s)", model_name)
    return model, processor


def _image_id(image_bytes: bytes) -> str:
    """Stable hash for deduplication."""
    return hashlib.shake_256(image_bytes).hexdigest(16)


# ── Core embedding functions ─────────────────────────────────────────

def embed_image(image_path: Path) -> np.ndarray:
    """Embed a single image via CLIP. Returns 512-dim L2-normalized vector."""
    model, processor = _load_clip()
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")  # type: ignore[operator]

    import torch
    with torch.no_grad():
        feats = model.get_image_features(**inputs)  # type: ignore[union-attr]
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze().cpu().numpy().astype(np.float32)


def embed_image_batch(image_paths: list[Path]) -> np.ndarray:
    """Embed multiple images via CLIP. Returns (N, 512) array."""
    if not image_paths:
        return np.empty((0, _CLIP_DIM), dtype=np.float32)

    model, processor = _load_clip()
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True)  # type: ignore[operator]

    import torch
    with torch.no_grad():
        feats = model.get_image_features(**inputs)  # type: ignore[union-attr]
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)


def embed_text_for_images(text: str) -> np.ndarray:
    """Embed text via CLIP text encoder for cross-modal search. Returns 512-dim vector."""
    model, processor = _load_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)  # type: ignore[operator]

    import torch
    with torch.no_grad():
        feats = model.get_text_features(**inputs)  # type: ignore[union-attr]
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze().cpu().numpy().astype(np.float32)


# ── PDF image extraction ─────────────────────────────────────────────

def extract_pdf_images(pdf_path: Path) -> list[ImageChunk]:
    """Pull embedded images from a PDF via PyMuPDF. Saves to data/images/."""
    if not _HAS_FITZ:
        logger.debug("PyMuPDF not available, skipping image extraction for %s", pdf_path)
        return []

    store = _image_store_path()
    chunks: list[ImageChunk] = []

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        logger.warning("Failed to open PDF %s for image extraction: %s", pdf_path, exc)
        return []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                img_bytes = base_image["image"]
                ext = base_image.get("ext", "png")
                content_type = f"image/{ext}"

                # skip tiny images (likely icons/bullets)
                width = base_image.get("width", 0)
                height = base_image.get("height", 0)
                if width < 50 or height < 50:
                    continue

                img_id = _image_id(img_bytes)
                fname = f"{img_id}.{ext}"
                dest = store / fname
                if not dest.exists():
                    dest.write_bytes(img_bytes)

                chunks.append(ImageChunk(
                    image_id=img_id,
                    image_path=str(dest.relative_to(PROJECT_ROOT)),
                    source=str(pdf_path),
                    caption=f"Image from {pdf_path.name}, page {page_num + 1}",
                    content_type=content_type,
                    width=width,
                    height=height,
                    page_number=page_num + 1,
                ))
            except Exception as exc:
                logger.debug("Skipped image xref=%d in %s: %s", xref, pdf_path, exc)

    doc.close()
    logger.info("Extracted %d images from %s", len(chunks), pdf_path.name)
    return chunks


def extract_markdown_images(md_path: Path) -> list[ImageChunk]:
    """Find image references in Markdown and resolve them to actual files."""
    import re

    if not md_path.exists():
        return []

    text = md_path.read_text(errors="replace")
    # match ![alt](path) and bare image paths
    pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
    chunks: list[ImageChunk] = []

    for alt_text, img_ref in pattern.findall(text):
        # resolve relative to markdown file's directory
        img_path = (md_path.parent / img_ref).resolve()
        if not img_path.exists():
            continue
        if img_path.suffix.lower() not in _SUPPORTED_IMAGE_EXTS:
            continue

        try:
            img = Image.open(img_path) if _HAS_CLIP else None
            width = img.width if img else 0
            height = img.height if img else 0
        except Exception:
            width, height = 0, 0

        img_bytes = img_path.read_bytes()
        img_id = _image_id(img_bytes)

        # copy to image store for consistent serving
        store = _image_store_path()
        dest = store / f"{img_id}{img_path.suffix.lower()}"
        if not dest.exists():
            dest.write_bytes(img_bytes)

        caption = alt_text if alt_text else f"Image from {md_path.name}"
        chunks.append(ImageChunk(
            image_id=img_id,
            image_path=str(dest.relative_to(PROJECT_ROOT)),
            source=str(md_path),
            caption=caption,
            content_type=f"image/{img_path.suffix.lower().lstrip('.')}",
            width=width,
            height=height,
            page_number=None,
        ))

    return chunks


def discover_standalone_images(docs_path: Path) -> list[ImageChunk]:
    """Find standalone image files in the docs directory tree."""
    if not docs_path.exists():
        return []

    chunks: list[ImageChunk] = []
    store = _image_store_path()

    for img_path in docs_path.rglob("*"):
        if img_path.suffix.lower() not in _SUPPORTED_IMAGE_EXTS:
            continue
        if not img_path.is_file():
            continue

        try:
            img_bytes = img_path.read_bytes()
            img_id = _image_id(img_bytes)

            if _HAS_CLIP:
                img = Image.open(img_path)
                width, height = img.width, img.height
            else:
                width, height = 0, 0

            # copy to image store
            dest = store / f"{img_id}{img_path.suffix.lower()}"
            if not dest.exists():
                dest.write_bytes(img_bytes)

            chunks.append(ImageChunk(
                image_id=img_id,
                image_path=str(dest.relative_to(PROJECT_ROOT)),
                source=str(img_path.relative_to(docs_path)),
                caption=img_path.stem.replace("_", " ").replace("-", " ").title(),
                content_type=f"image/{img_path.suffix.lower().lstrip('.')}",
                width=width,
                height=height,
                page_number=None,
            ))
        except Exception as exc:
            logger.debug("Skipped image %s: %s", img_path, exc)

    return chunks


# ── Image captioning via Ollama vision model ─────────────────────────

def caption_image(image_path: Path, fallback: str = "") -> str:
    """Generate a text caption for an image using Ollama vision model.

    Falls back to the provided default if the vision model is unavailable
    or captioning fails. Never blocks the pipeline on captioning errors.
    """
    if not image_path.exists():
        return fallback

    vision_model = _vision_model_name()
    try:
        from backend.core.model_router import get_engine
        import base64 as _b64mod

        engine = get_engine()

        # encode image as base64 for vision API
        img_bytes = image_path.read_bytes()
        img_b64 = _b64mod.b64encode(img_bytes).decode("utf-8")

        max_tokens = int(_mm_cfg().get("caption_max_tokens", 200))
        caption = engine.generate(
            messages=[{
                "role": "user",
                "content": "Describe this image concisely in 1-2 sentences.",
            }],
            model=vision_model,
            temperature=0.1, num_predict=max_tokens,
            images=[img_b64],
        ).strip()
        if caption:
            return caption
    except Exception as exc:
        logger.debug("Vision captioning unavailable (%s): %s", vision_model, exc)

    return fallback


def caption_images_batch(
    image_chunks: list[ImageChunk],
    skip_existing: bool = True,
) -> list[ImageChunk]:
    """Add captions to image chunks using the vision model. Modifies in-place + returns."""
    for chunk in image_chunks:
        current = chunk.get("caption", "")
        # skip if already has a real caption (not just a filename placeholder)
        if skip_existing and current and "Image from" not in current:
            continue

        img_path = PROJECT_ROOT / chunk["image_path"]
        chunk["caption"] = caption_image(img_path, fallback=current)

    return image_chunks


# ── FAISS image index ────────────────────────────────────────────────

def build_image_index(chunks: list[ImageChunk]) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from CLIP image embeddings.

    Uses inner-product (cosine similarity on normalized vectors) because
    CLIP embeddings are L2-normalized. Returns the index; also persists
    to disk alongside image metadata JSON.
    """
    index_path = _image_index_path()
    meta_path = _image_metadata_path()
    index = faiss.IndexFlatIP(_CLIP_DIM)

    if not chunks:
        ensure_path(str(index_path))
        faiss.write_index(index, str(index_path))
        meta_path.write_text("[]")
        return index

    # filter to chunks with valid image files
    valid: list[ImageChunk] = []
    paths: list[Path] = []
    for c in chunks:
        p = PROJECT_ROOT / c["image_path"]
        if p.exists():
            valid.append(c)
            paths.append(p)

    if not valid:
        ensure_path(str(index_path))
        faiss.write_index(index, str(index_path))
        meta_path.write_text("[]")
        return index

    embeddings = embed_image_batch(paths)
    index.add(embeddings)

    ensure_path(str(index_path))
    faiss.write_index(index, str(index_path))
    meta_path.write_text(json.dumps(
        [dict(c) for c in valid], indent=2,
    ))

    logger.info("Built image index: %d images, %d-dim CLIP embeddings", len(valid), _CLIP_DIM)
    return index


def get_image_index() -> faiss.IndexFlatIP:
    """Load or build the image FAISS index."""
    path = _image_index_path()
    if path.exists():
        logger.debug("Image index loaded from disk")
        return faiss.read_index(str(path))
    logger.info("No image index found; returning empty")
    return faiss.IndexFlatIP(_CLIP_DIM)


def retrieve_images(query: str, top_k: int = 5) -> list[ImageSearchResult]:
    """Cross-modal retrieval: embed text query via CLIP, search image index.

    Returns ranked image results with captions and scores. Gracefully returns
    empty if CLIP is unavailable or no images are indexed.
    """
    if not _is_enabled():
        return []
    if not _HAS_CLIP:
        return []

    index = get_image_index()
    if index.ntotal == 0:
        return []

    meta_path = _image_metadata_path()
    if not meta_path.exists():
        return []

    meta: list[dict[str, object]] = json.loads(meta_path.read_text())
    q_emb = embed_text_for_images(query).reshape(1, _CLIP_DIM)

    search_k = min(top_k, index.ntotal)
    scores, idxs = index.search(q_emb, search_k)

    results: list[ImageSearchResult] = []
    for i in range(len(idxs[0])):
        idx = int(idxs[0][i])
        score = float(scores[0][i])

        if idx < 0 or idx >= len(meta):
            continue
        # CLIP cosine sim threshold: skip weak matches
        if score < 0.15:
            continue

        m = meta[idx]
        results.append(ImageSearchResult(
            text=str(m.get("caption", "")),
            score=score,
            source=str(m.get("source", "")),
            id=str(m.get("image_id", "")),
            image_path=str(m.get("image_path", "")),
            caption=str(m.get("caption", "")),
            width=int(m.get("width", 0)),
            height=int(m.get("height", 0)),
        ))

    return results


# ── Full pipeline: process all documents for images ──────────────────

def process_documents_for_images(docs_path: Path) -> list[ImageChunk]:
    """Scan all documents in docs_path for images. Extracts from PDFs,
    resolves Markdown references, and discovers standalone image files.

    Returns all image chunks found, with basic filename-based captions.
    Call caption_images_batch() afterward for vision-model captions.
    """
    if not _is_enabled():
        logger.debug("Multimodal disabled in config")
        return []

    all_chunks: list[ImageChunk] = []

    # PDF image extraction
    if _HAS_FITZ:
        for pdf in docs_path.rglob("*.pdf"):
            all_chunks.extend(extract_pdf_images(pdf))

    # Markdown image references
    if _HAS_CLIP:
        for md in docs_path.rglob("*.md"):
            all_chunks.extend(extract_markdown_images(md))

    # standalone images
    all_chunks.extend(discover_standalone_images(docs_path))

    # deduplicate by image_id
    seen: set[str] = set()
    unique: list[ImageChunk] = []
    for c in all_chunks:
        if c["image_id"] not in seen:
            seen.add(c["image_id"])
            unique.append(c)

    logger.info(
        "Multimodal scan: %d images found (%d unique) from %s",
        len(all_chunks), len(unique), docs_path,
    )
    return unique


def build_multimodal_index(docs_path: Path | None = None) -> int:
    """Full pipeline: discover images, optionally caption, build FAISS index.

    Returns the number of images indexed.
    """
    if docs_path is None:
        docs_path = PROJECT_ROOT / "data" / "docs"

    chunks = process_documents_for_images(docs_path)
    if not chunks:
        # write empty index so get_image_index() doesn't keep rebuilding
        build_image_index([])
        return 0

    # try captioning (best-effort, won't block on failure)
    try:
        chunks = caption_images_batch(chunks)
    except Exception as exc:
        logger.debug("Batch captioning skipped: %s", exc)

    build_image_index(chunks)
    return len(chunks)


def image_to_base64(image_path: Path) -> str:
    """Read an image file and return as base64-encoded data URI."""
    if not image_path.exists():
        return ""

    suffix = image_path.suffix.lower().lstrip(".")
    mime_map = {"jpg": "jpeg", "svg": "svg+xml"}
    mime = mime_map.get(suffix, suffix)

    img_bytes = image_path.read_bytes()
    encoded = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/{mime};base64,{encoded}"
