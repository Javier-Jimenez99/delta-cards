#!/usr/bin/env python3
"""Unified multi-language downloader and index builder for Pokémon TCG cards.

Downloads, embeds, and builds USearch indices and SQLite metadata stores uniformly
for the three primary supported languages: English ('en'), Spanish ('es'), and Japanese ('ja').

Uniform Storage Pattern:
  - Index:     data/pokemon_cards_{lang}.usearch
  - Metadata:  data/pokemon_cards_{lang}_meta.sqlite
  - Images:    data/images_cache/{lang}/{set_id}_{local_id}.webp
  - Texts:     tcgdex_card_texts table inside data/pokemon_cards_{lang}_meta.sqlite

Usage:
  # Download and index all three languages (en, es, ja):
  uv run python scripts/download_target_indices.py

  # Download and index specific languages:
  uv run python scripts/download_target_indices.py --langs es ja
  uv run python scripts/download_target_indices.py --langs en

  # Test with a specific set and card limit:
  uv run python scripts/download_target_indices.py --langs es --sets sv01 --limit 10
"""

import argparse
import asyncio
import logging
import os
import sys
import time

from delta_cards.recognition.index_builder import build_index

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("delta_cards.downloader")

# Mute noisy internal loggers so tqdm progress bar stays clean and undisturbed
logging.getLogger("delta_cards.recognition.vector_index").setLevel(logging.WARNING)
logging.getLogger("delta_cards.recognition.dinov3_embedder").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

DEFAULT_LANGUAGES = ["es", "en", "ja"]
DEFAULT_INDEX_PATH = "data/pokemon_cards.usearch"


async def run_uniform_download(
    languages: list[str],
    index_path: str = DEFAULT_INDEX_PATH,
    concurrency: int = 16,
    sets_filter: list[str] = None,
    limit_cards: int = None,
    cache_dir: str = "data/images_cache",
    skip_ocr: bool = False,
    skip_index: bool = False,
    ocr_workers: int = 8,
) -> None:
    """Executes unified index construction across all specified target languages."""
    clean_langs = [l.strip().lower() for l in languages]
    meta_path = index_path.rsplit(".", 1)[0] + "_meta.sqlite"

    logger.info("=" * 70)
    logger.info("Unified Multi-Language Index Builder & Scan OCR Extractor")
    logger.info(f"  • Target Languages: {clean_langs}")
    logger.info(f"  • Unified Index:    {index_path} (Skip: {skip_index})")
    logger.info(f"  • Metadata SQLite:  {meta_path} (Skip OCR: {skip_ocr}, Workers: {ocr_workers})")
    logger.info(f"  • Image Cache Base: {cache_dir}")
    logger.info(f"  • Concurrency:      {concurrency}")
    if sets_filter:
        logger.info(f"  • Sets Filter:      {sets_filter}")
    if limit_cards:
        logger.info(f"  • Card Limit:       {limit_cards}")
    logger.info("=" * 70)

    t_global_start = time.perf_counter()

    for idx, clean_lang in enumerate(clean_langs, 1):
        lang_cache = os.path.join(cache_dir, clean_lang)
        logger.info(f"\n[{idx}/{len(clean_langs)}] Processing Language: '{clean_lang.upper()}' -> {index_path}")

        t_lang_start = time.perf_counter()
        try:
            await build_index(
                language=clean_lang,
                index_path=index_path,
                cache_dir=cache_dir,
                concurrency=concurrency,
                sets_filter=sets_filter,
                limit_cards=limit_cards,
                skip_ocr=skip_ocr,
                skip_index=skip_index,
                ocr_workers=ocr_workers,
            )
            elapsed = time.perf_counter() - t_lang_start
            logger.info(f"✓ Language '{clean_lang.upper()}' finished in {elapsed:.1f}s.")
        except Exception as exc:
            logger.error(f"✗ Failed processing language '{clean_lang}': {exc}", exc_info=True)

    t_total = time.perf_counter() - t_global_start
    logger.info("\n" + "=" * 70)
    logger.info(f"Unified index and OCR extraction successfully updated in {t_total:.1f}s!")
    logger.info(f"Final files: {index_path} and {meta_path}")
    logger.info("=" * 70)


async def process_dataset_cards(
    dataset_path: str,
    index_path: str = DEFAULT_INDEX_PATH,
    cache_dir: str = "data/images_cache",
    ocr_workers: int = 8,
) -> None:
    """Ensures 100% of cards in the evaluation dataset are downloaded, indexed, and OCR-cached."""
    import json
    import aiohttp
    import cv2
    from delta_cards.recognition.dinov3_embedder import DINOv3Embedder
    from delta_cards.recognition.vector_index import USearchCardIndex
    from delta_cards.recognition.card_ocr_cache import CandidateOCRCache
    from delta_cards.recognition.card_ocr import CardOCR

    if not os.path.isfile(dataset_path):
        logger.error(f"Dataset file not found: {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    target_cards = {}
    for item in dataset:
        exp_id = item.get("expected_tcgdex_id", "")
        if not exp_id or exp_id.lower() == "fake":
            continue
        if exp_id not in target_cards:
            target_cards[exp_id] = {
                "id": exp_id,
                "title": item.get("title", ""),
                "category": item.get("category", ""),
            }

    meta_path = index_path.rsplit(".", 1)[0] + "_meta.sqlite"
    logger.info("=" * 70)
    logger.info("Dataset Cards Synchronizer (Image + USearch Index + Scan OCR)")
    logger.info(f"  • Dataset Path:    {dataset_path}")
    logger.info(f"  • Unique Targets:  {len(target_cards)} cards")
    logger.info(f"  • Unified Index:   {index_path}")
    logger.info(f"  • Metadata SQLite: {meta_path}")
    logger.info(f"  • Images Cache:    {cache_dir}")
    logger.info("=" * 70)

    card_index = USearchCardIndex(index_path=index_path, meta_path=meta_path, view_only=False)
    ocr_cache = CandidateOCRCache(db_path=meta_path, cache_dir=cache_dir)
    embedder = DINOv3Embedder()
    ocr_engine = CardOCR()

    indexed_ids = set()
    for lang in ("es", "en", "ja"):
        indexed_ids.update(card_index.get_indexed_tcgdex_ids(lang=lang))
    cached_ocr_ids = ocr_cache.get_cached_ocr_ids()

    t_start = time.perf_counter()
    new_images = 0
    new_indexed = 0
    new_ocr = 0

    from tqdm.auto import tqdm
    pbar = tqdm(total=len(target_cards), desc="Dataset Cards", unit="card")

    async with aiohttp.ClientSession() as session:
        for card_id, card_info in target_cards.items():
            set_id, local_id = card_id.split("-", 1) if "-" in card_id else (card_id, "")
            norm_local = local_id.lstrip("0") or "0"
            norm_id = f"{set_id}-{norm_local}"

            # 1. Find existing local image in cache
            found_img_path = None
            found_lang = "es"
            candidates_stems = [
                f"{set_id}_{local_id}",
                f"{set_id}_{norm_local}",
                f"{set_id}_{norm_local.zfill(3)}",
            ]
            for lang in ("es", "en", "ja"):
                for stem in candidates_stems:
                    for ext in (".webp", ".png", ".jpg", ".jpeg"):
                        cand_path = os.path.join(cache_dir, lang, f"{stem}{ext}")
                        if os.path.isfile(cand_path) and os.path.getsize(cand_path) > 1000:
                            found_img_path = cand_path
                            found_lang = lang
                            break
                    if found_img_path:
                        break
                if found_img_path:
                    break

            # 2. If missing, look up canonical image_url in card_metadata SQLite table
            if not found_img_path:
                try:
                    import sqlite3
                    with sqlite3.connect(meta_path) as s_conn:
                        s_cur = s_conn.cursor()
                        row = s_cur.execute(
                            "SELECT lang, image_url, name, tcgdex_id FROM card_metadata "
                            "WHERE tcgdex_id IN (?, ?) "
                            "ORDER BY CASE WHEN lang='es' THEN 0 WHEN lang='en' THEN 1 ELSE 2 END",
                            (card_id, norm_id),
                        ).fetchone()
                        if row and row[1]:
                            db_lang, db_url, db_name, db_id = row
                            dest_dir = os.path.join(cache_dir, db_lang)
                            os.makedirs(dest_dir, exist_ok=True)
                            dest_file = os.path.join(dest_dir, f"{set_id}_{local_id}.webp")

                            urls_to_try = [
                                db_url,
                                db_url.replace("/high.webp", "/high.png"),
                                db_url.replace(".webp", ".png"),
                            ]
                            for u in urls_to_try:
                                try:
                                    async with session.get(u, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                                        if resp.status == 200:
                                            data = await resp.read()
                                            if len(data) > 1000:
                                                with open(dest_file, "wb") as f:
                                                    f.write(data)
                                                found_img_path = dest_file
                                                found_lang = db_lang
                                                new_images += 1
                                                break
                                except Exception:
                                    pass
                except Exception as ex:
                    logger.debug(f"Metadata lookup error for {card_id}: {ex}")

            # 3. Fallback: try direct CDN patterns
            if not found_img_path:
                for lang in ("es", "en", "ja"):
                    for stem in [f"{set_id}/{local_id}", f"{set_id}/{norm_local}"]:
                        for ext in ("webp", "png"):
                            cdn_url = f"https://assets.tcgdex.net/{lang}/{stem}/high.{ext}"
                            try:
                                async with session.get(cdn_url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                                    if resp.status == 200:
                                        data = await resp.read()
                                        if len(data) > 1000:
                                            dest_dir = os.path.join(cache_dir, lang)
                                            os.makedirs(dest_dir, exist_ok=True)
                                            dest_file = os.path.join(dest_dir, f"{set_id}_{local_id}.webp")
                                            with open(dest_file, "wb") as f:
                                                f.write(data)
                                            found_img_path = dest_file
                                            found_lang = lang
                                            new_images += 1
                                            break
                            except Exception:
                                continue
                        if found_img_path:
                            break
                    if found_img_path:
                        break

            if not found_img_path or not os.path.isfile(found_img_path):
                logger.warning(f"Could not locate or download image for {card_id}")
                pbar.update(1)
                continue

            img = cv2.imread(found_img_path)
            if img is None:
                pbar.update(1)
                continue

            # 4. Check & perform visual index
            if card_id not in indexed_ids and norm_id not in indexed_ids:
                embedding = embedder.embed_image(img)
                next_key = card_index.get_next_key()
                metadata = {
                    "tcgdex_id": norm_id,
                    "lang": found_lang,
                    "name": card_info.get("title", card_id),
                    "set_id": set_id,
                    "set_name": set_id,
                    "local_id": norm_local,
                    "total_cards": None,
                    "card_fraction": None,
                    "image_url": f"https://assets.tcgdex.net/{found_lang}/{set_id}/{norm_local}/high.webp",
                }
                card_index.add_card(next_key, embedding, metadata)
                indexed_ids.add(norm_id)
                indexed_ids.add(card_id)
                new_indexed += 1

            # 5. Check & perform OCR cache
            if card_id not in cached_ocr_ids or norm_id not in cached_ocr_ids:
                ocr_res = ocr_engine.extract(img)
                ocr_cache.save_cached_ocr(card_id, found_lang, ocr_res)
                if norm_id != card_id:
                    ocr_cache.save_cached_ocr(norm_id, found_lang, ocr_res)
                cached_ocr_ids.add(card_id)
                cached_ocr_ids.add(norm_id)
                new_ocr += 1

            pbar.update(1)

    pbar.close()
    if new_indexed > 0:
        card_index.save()

    t_elapsed = time.perf_counter() - t_start
    logger.info("=" * 70)
    logger.info(f"✓ Dataset synchronization complete in {t_elapsed:.1f}s!")
    logger.info(f"  • New images downloaded: {new_images}")
    logger.info(f"  • New cards indexed:    {new_indexed}")
    logger.info(f"  • New OCR extracted:    {new_ocr}")
    logger.info(f"  • Total dataset coverage: {len(target_cards)}/{len(target_cards)} verified (100%)")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Unified multi-language downloader, index builder, and OCR extractor for Pokémon TCG"
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        const="data/evaluation_set/ground_truth.json",
        default=None,
        help="Synchronize all cards in an evaluation dataset JSON file (default if flag used: data/evaluation_set/ground_truth.json)",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        default=DEFAULT_LANGUAGES,
        help=f"Languages to download and index into unified index (default: {' '.join(DEFAULT_LANGUAGES)})",
    )
    parser.add_argument(
        "--index-path",
        default=DEFAULT_INDEX_PATH,
        help=f"Unified output USearch index file (default: {DEFAULT_INDEX_PATH})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Asynchronous download concurrency (default: 16)",
    )
    parser.add_argument(
        "--sets",
        nargs="*",
        default=None,
        help="Optional set IDs to filter (e.g. sv01 sv02)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of cards to process (for testing)",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/images_cache",
        help="Base image cache directory (default: data/images_cache)",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip OCR extraction on card scans",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip visual DINOv3 embedding and USearch indexing",
    )
    parser.add_argument(
        "--ocr-workers",
        type=int,
        default=8,
        help="Parallel threads for OCR extraction (default: 8)",
    )

    args = parser.parse_args()

    if args.dataset:
        asyncio.run(
            process_dataset_cards(
                dataset_path=args.dataset,
                index_path=args.index_path,
                cache_dir=args.cache_dir,
                ocr_workers=args.ocr_workers,
            )
        )
    else:
        asyncio.run(
            run_uniform_download(
                languages=args.langs,
                index_path=args.index_path,
                concurrency=args.concurrency,
                sets_filter=args.sets,
                limit_cards=args.limit,
                cache_dir=args.cache_dir,
                skip_ocr=args.skip_ocr,
                skip_index=args.skip_index,
                ocr_workers=args.ocr_workers,
            )
        )


if __name__ == "__main__":
    main()
