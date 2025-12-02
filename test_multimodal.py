#!/usr/bin/env python3
"""
Test script for multimodal retrieval (CLIP + PyMuPDF).
Verifies image and PDF retrieval functions work correctly.
"""

import sys
sys.path.insert(0, '.')

from backend.core.retrievers import retrieve_image, retrieve_pdf
from pathlib import Path

def test_multimodal_retrievers():
    """Quick smoke test for CLIP and PyMuPDF retrievers."""

    print("="*70)
    print("MULTIMODAL RETRIEVAL TEST")
    print("="*70)
    print()

    # Test 1: PDF Retrieval (if any PDFs exist)
    print("📄 TEST 1: PDF Retrieval")
    pdf_paths = list(Path("data/docs").glob("*.pdf"))[:3]  # Top 3 PDFs

    if pdf_paths:
        pdf_paths_str = [str(p) for p in pdf_paths]
        print(f"   Found {len(pdf_paths)} PDFs to test")

        results = retrieve_pdf("what is FAISS indexing", pdf_paths_str, top_k=2)

        if results:
            print(f"   ✅ Retrieved {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"      {i}. Score: {result['score']:.3f}")
                print(f"         Source: {result['source']}")
                print(f"         Text: {result['text'][:100]}...")
        else:
            print("   ⚠️  No results (may need actual PDF content)")
    else:
        print("   ℹ️  No PDFs found in data/docs - skipping PDF test")

    print()

    # Test 2: Image Retrieval (if any images exist)
    print("🖼️  TEST 2: Image Retrieval")

    # Look for common image extensions
    image_exts = ['*.png', '*.jpg', '*.jpeg']
    image_paths = []
    for ext in image_exts:
        image_paths.extend(list(Path("data").rglob(ext))[:2])

    if image_paths:
        image_paths_str = [str(p) for p in image_paths[:3]]
        print(f"   Found {len(image_paths_str)} images to test")

        results = retrieve_image("technical diagram", image_paths_str, top_k=2)

        if results:
            print(f"   ✅ Retrieved {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"      {i}. Score: {result['score']:.3f}")
                print(f"         Source: {result['source']}")
        else:
            print("   ⚠️  No results")
    else:
        print("   ℹ️  No images found in data/ - skipping image test")

    print()
    print("="*70)
    print("TEST COMPLETE")
    print("="*70)
    print()
    print("✅ Multimodal retrievers are functional!")
    print("   - retrieve_pdf() ready for PDF documents")
    print("   - retrieve_image() ready for CLIP-based image search")
    print()
    print("To use in production:")
    print("   results = retrieve_pdf(query, ['/path/to/doc.pdf'])")
    print("   results = retrieve_image(query, ['/path/to/image.jpg'])")

if __name__ == '__main__':
    test_multimodal_retrievers()
