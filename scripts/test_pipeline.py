"""Script to test and profile the entire card recognition and resolution pipeline step-by-step.

Executes each pipeline stage individually for real Wallapop marketplace card offers:
1. Visual Isolation & Perspective Deskew (ONNX YOLO OBB + OpenCV)
2. Image Preprocessing & Multi-pass OCR (Tesseract)
3. Local LLM Entity Extraction & Normalization (Qwen 0.5B via llama-cpp)
4. Live TCGdex API Cascade Resolution & Cardmarket ID Retrieval

Always utilizes the internal library:
from delta_cards.recognition import CardRecognitionPipeline
"""

import argparse
import asyncio
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure src/ is in sys.path
repo_root = Path(__file__).resolve().parent.parent
src_path = str(repo_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Always import and use the library code inside src/delta_cards
from delta_cards.recognition import (
    CardRecognitionPipeline,
    CardNotDetectedError,
)


def download_image_if_needed(url: str, output_path: str) -> str:
    """Downloads an image from a URL to a local destination if not already cached."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        with open(output_path, "wb") as f:
            f.write(resp.read())
    return output_path


# Real Wallapop card offers from data/delta_cards.db with live CDN image URLs
WALLAPOP_OFFERS: List[Dict[str, Any]] = [
    {
        "id": "wp_pikachu_ex_057",
        "title": "Carta Pokémon Pikachu EX Teracristal 057/191",
        "description": "Carta Pokémon de Pikachu EX con arte Teracristal. Habilidad: Corazón Resuelto. Ataque: Rayo Top",
        "offer_url": "https://wallapop.com/item/carta-pokemon-pikachu-ex-teracristal-057-191-1300487932",
        "image_url": "https://cdn.wallapop.com/images/10420/li/9y/__/c10420p1300487932/i6722771485.jpg?pictureSize=W800",
        "price": 20.00,
        "expected_tcgdex_id": "sv08-057",
    },
    {
        "id": "wp_nidoking_base_11",
        "title": "Carta Pokémon Nidoking Holo Rare. Base Set 11/102. 1999, España. NM - Mint 8.0.",
        "description": "Carta Pokémon Nidoking Holo Rare. Base Set 11/102. 1999, España. NM - Mint 8.0.",
        "offer_url": "https://wallapop.com/item/carta-pokemon-nidoking-holo-rare-base-set-11-102-1999-espana-nm-mint-8-0-1300486713",
        "image_url": "https://cdn.wallapop.com/images/10420/li/9x/__/c10420p1300486713/i6722762419.jpg?pictureSize=W800",
        "price": 116.00,
        "expected_tcgdex_id": "base1-11",
    },
    {
        "id": "wp_zapdos_base_16",
        "title": "Carta Pokémon Zapdos Holo Rare. Base Set 16/102. 1999, España. NM 7.5.",
        "description": "Carta Pokémon Zapdos Holo Rare. Base Set 16/102. 1999, España. NM 7.5.",
        "offer_url": "https://wallapop.com/item/carta-pokemon-zapdos-holo-rare-base-set-16-102-1999-espana-nm-7-5-1300486653",
        "image_url": "https://cdn.wallapop.com/images/10420/li/9x/__/c10420p1300486653/i6722762127.jpg?pictureSize=W800",
        "price": 132.00,
        "expected_tcgdex_id": "base1-16",
    },
    {
        "id": "wp_shining_rayquaza_56",
        "title": "Carta Pokémon Shining Rayquaza 56/73",
        "description": "Carta Pokémon Shining Rayquaza 56/73 de la colección Leyendas Luminosas.",
        "offer_url": "https://wallapop.com/item/carta-pokemon-shining-rayquaza-56-73-1300436786",
        "image_url": "https://cdn.wallapop.com/images/10420/li/8v/__/c10420p1300436786/i6722549789.jpg?pictureSize=W800",
        "price": 100.00,
        "expected_tcgdex_id": "sm3.5-56",
    },
    {
        "id": "wp_wigglytuff_105",
        "title": "Carta Pokémon Wigglytuff 105/094",
        "description": "Carta Pokémon de Wigglytuff, edición 105/094. HP: 120. Ataques: Round, Seismic Toss. idioma : ingles",
        "offer_url": "https://wallapop.com/item/carta-pokemon-wigglytuff-105-094-1300488106",
        "image_url": "https://cdn.wallapop.com/images/10420/li/9z/__/c10420p1300488106/i6722772684.jpg?pictureSize=W800",
        "price": 3.00,
        "expected_tcgdex_id": "sv3.5-105",
    },
]


async def run_pipeline_step_by_step_profiling(limit: Optional[int] = None):
    """Executes the pipeline stage-by-stage for real Wallapop offers and prints detailed timing benchmarks."""
    active_offers = WALLAPOP_OFFERS[:limit] if (limit and limit > 0) else WALLAPOP_OFFERS

    print("=" * 95)
    print(" DELTA CARDS: Step-by-Step Pipeline Profiling across Real Wallapop Offers (CPU)")
    print("=" * 95)

    # 1. Initialize Library Pipeline
    t_init_start = time.perf_counter()
    print("[1/3] Initializing CardRecognitionPipeline from delta_cards.recognition...")
    pipeline = CardRecognitionPipeline()
    t_init = time.perf_counter() - t_init_start
    detector_name = (
        f"ONNX Runtime ({os.path.basename(pipeline.detector.model_path)})"
        if pipeline.detector.is_onnx
        else f"PyTorch ({pipeline.detector.model_path})"
    )
    llm_name = os.path.basename(pipeline.entity_parser.model_path)
    print(f"      ✓ Pipeline initialized in {t_init:.2f}s | Detector: {detector_name} | LLM: llama.cpp ({llm_name})\n")

    # 2. Local cache directory for real listing images
    temp_dir = tempfile.mkdtemp(prefix="delta_cards_real_offers_")
    print(f"[2/3] Downloading real Wallapop card images to local cache: {temp_dir}\n")

    results_summary = []

    # 3. Process each Wallapop offer step-by-step
    print(f"[3/3] Profiling pipeline stages individually for {len(active_offers)} Wallapop offer(s):\n")

    for idx, offer in enumerate(active_offers, start=1):
        print("═" * 95)
        print(f"[{idx}/{len(active_offers)}] WALLAPOP OFFER: '{offer['title']}'")
        print(f"  💶 Listing Price:         {offer['price']:.2f} EUR")
        print(f"  📝 Seller Description:    {offer['description']}")

        # Download image to local file so download latency is excluded from CV detection profiling
        img_path = os.path.join(temp_dir, f"{offer['id']}.jpg")
        await asyncio.to_thread(download_image_if_needed, offer["image_url"], img_path)

        t_offer_start = time.perf_counter()

        # -------------------------------------------------------------
        # STEP 1: Visual Isolation & Perspective Deskew (ONNX YOLO OBB + OpenCV)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        isolation_status = "isolated"
        card_image = None
        try:
            card_image = await asyncio.to_thread(pipeline.detector.isolate_card, img_path)
        except CardNotDetectedError:
            isolation_status = "fallback_full_frame"
            card_image = await asyncio.to_thread(pipeline.detector._load_image, img_path)
        except Exception:
            isolation_status = "failed"
            card_image = None
        t_step1 = time.perf_counter() - t0
        img_dims = f"{card_image.shape[1]}x{card_image.shape[0]}" if card_image is not None else "N/A"
        print(f"  ⏱️  [Step 1] Visual Isolation:       {t_step1:.3f}s | Status: {isolation_status} ({img_dims} px)")

        # -------------------------------------------------------------
        # STEP 2: Full-Card Image Preprocessing & Multi-pass OCR
        # -------------------------------------------------------------
        t1 = time.perf_counter()
        ocr_text = ""
        if card_image is not None:
            ocr_text = await asyncio.to_thread(pipeline.ocr_reader.extract_text, card_image)
        t_step2 = time.perf_counter() - t1
        ocr_snippet = ocr_text.replace("\n", " ").strip()[:65]
        print(f"  ⏱️  [Step 2] Preprocessing & OCR:     {t_step2:.3f}s | Chars: {len(ocr_text):<3} | Snippet: \"{ocr_snippet}...\"")

        # -------------------------------------------------------------
        # STEP 3: Hybrid Entity Extraction (Strict Deterministic -> LLM)
        # -------------------------------------------------------------
        t2 = time.perf_counter()
        strict_meta = pipeline.entity_parser.try_strict_deterministic_extraction(
            ocr_text=ocr_text,
            seller_title=offer["title"],
            seller_description=offer["description"],
        )

        metadata = None
        resolution = None
        method = "LLM Fallback"
        t_step3 = 0.0
        t_step4 = 0.0

        if strict_meta is not None:
            t3 = time.perf_counter()
            strict_resolution = await pipeline.resolver.resolve_card(
                strict_meta, strict_verification=True
            )
            t_step4 = time.perf_counter() - t3
            if strict_resolution.get("status") == "success":
                metadata = strict_meta
                resolution = strict_resolution
                method = "Fast-Path"
                t_step3 = max(0.001, time.perf_counter() - t2 - t_step4)
                print(
                    f"  ⚡ [Step 3] Entity Extraction:       {t_step3:.3f}s | Mode: STRICT DETERMINISTIC (Zero False Positives) | "
                    f"Name: '{metadata.card_name}' | Set: '{metadata.set_code}' | Num: '{metadata.collector_number}' | Lang: '{metadata.language.upper()}'"
                )

        if resolution is None:
            t_llm_start = time.perf_counter()
            metadata = await asyncio.to_thread(
                pipeline.entity_parser.parse_listing,
                ocr_text=ocr_text,
                seller_title=offer["title"],
                seller_description=offer["description"],
                use_fast_path=False,
            )
            t_step3 = time.perf_counter() - t_llm_start
            print(
                f"  ⏱️  [Step 3] Local LLM Extraction:    {t_step3:.3f}s | Mode: LLM FALLBACK ({llm_name}) | "
                f"Name: '{metadata.card_name}' | Set: '{metadata.set_code}' | Num: '{metadata.collector_number}' | Lang: '{metadata.language.upper()}'"
            )

            # -------------------------------------------------------------
            # STEP 4: TCGdex Live API Cascade Resolution & Cardmarket ID
            # -------------------------------------------------------------
            t3 = time.perf_counter()
            resolution = await pipeline.resolver.resolve_card(
                metadata, strict_verification=False
            )
            t_step4 = time.perf_counter() - t3

        t_total = time.perf_counter() - t_offer_start

        status = resolution.get("status", "unknown")
        strategy = resolution.get("strategy")
        card = resolution.get("verified_card") or {}
        cm = resolution.get("cardmarket") or {}
        card_lang = (metadata.language or "es").upper()

        print(
            f"  ⏱️  [Step 4] TCGdex Live Resolution:  {t_step4:.3f}s | "
            f"Strategy: {strategy} | TCGDex ID: {card.get('id')} ({method})"
        )
        print(f"  ⏱️  ─────────────────────────────────────────────────────────────")
        print(f"  ⏱️  TOTAL LATENCY:                  {t_total:.3f}s")
        print(f"  ✅ Verified Card:                  {card.get('name')} | Set: {card.get('set', {}).get('name')}")
        print(f"  🌐 Card Language (Physical):        {card_lang}")
        print(f"  🔗 Cardmarket ID (idProduct):       {cm.get('idProduct')}")
        print(f"  🌐 Cardmarket Redirect URL:         {cm.get('url')}")
        print(f"  🛍️  Wallapop Offer URL:             {offer.get('offer_url', 'N/A')}")
        print(f"  📊 Live Prices:                     {cm.get('prices')}")

        results_summary.append({
            "title": offer["title"][:38] + "..." if len(offer["title"]) > 40 else offer["title"],
            "tcgdex_id": card.get("id") or "N/A",
            "cardmarket_id": cm.get("idProduct") or "N/A",
            "lang": card_lang,
            "method": method,
            "t_iso": t_step1,
            "t_ocr": t_step2,
            "t_llm": t_step3,
            "t_tcg": t_step4,
            "t_tot": t_total,
            "status": status,
        })

    # Summary Benchmark Table
    print("\n" + "=" * 122)
    print(" PIPELINE STEP-BY-STEP PROFILING SUMMARY (HYBRID MODE)")
    print("=" * 122)
    header = (
        f"{'Offer Title':<38} | {'Lang':<4} | "
        f"{'TCGDex ID':<12} | {'CM ID':<8} | {'Method':<11} | {'Iso(s)':<6} | {'OCR(s)':<6} | {'Ext(s)':<6} | {'TCG(s)':<6} | {'Total':<6} | {'Status'}"
    )
    print(header)
    print("-" * len(header))
    for r in results_summary:
        print(
            f"{r['title']:<38} | {r['lang']:<4} | "
            f"{r['tcgdex_id']:<12} | {r['cardmarket_id']:<8} | {r['method']:<11} | "
            f"{r['t_iso']:<6.2f} | {r['t_ocr']:<6.2f} | {r['t_llm']:<6.2f} | {r['t_tcg']:<6.2f} | {r['t_tot']:<6.2f} | {r['status'].upper()}"
        )
    print("=" * 122)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test and profile the delta-cards recognition pipeline on real Wallapop offers."
    )
    parser.add_argument(
        "count",
        nargs="?",
        type=int,
        default=None,
        help="Number of Wallapop offers to test (e.g. 1, 2). Default: 2.",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Number of Wallapop offers to test (e.g. -n 1).",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Process all available Wallapop offers.",
    )
    args = parser.parse_args()

    if args.all:
        selected_limit = None
    elif args.limit is not None:
        selected_limit = args.limit if args.limit > 0 else None
    elif args.count is not None:
        selected_limit = args.count if args.count > 0 else None
    else:
        selected_limit = 2  # Default to 2 offers for fast, responsive profiling

    asyncio.run(run_pipeline_step_by_step_profiling(limit=selected_limit))
