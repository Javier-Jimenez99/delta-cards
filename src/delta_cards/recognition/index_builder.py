import asyncio
import logging
import os
import time
import urllib.request
from typing import Any, Dict, List, Optional

# Ensure TCGdex SDK (which uses urllib) sends a browser-like User-Agent to prevent 503 errors
try:
    _opener = urllib.request.build_opener()
    _opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")]
    urllib.request.install_opener(_opener)
except Exception:
    pass

import aiohttp
import cv2
import numpy as np
from tcgdexsdk import TCGdex

from concurrent.futures import ThreadPoolExecutor
from .card_ocr import CardOCR
from .card_ocr_cache import CandidateOCRCache
from .dinov3_embedder import DINOv3Embedder
from .vector_index import USearchCardIndex

logger = logging.getLogger(__name__)


async def fetch_card_image(
    session: aiohttp.ClientSession,
    image_url: str,
    cache_base: str,
    semaphore: asyncio.Semaphore,
) -> Optional[bytes]:
    """Downloads card image with local disk caching and bounded concurrency.

    Tries high-quality WebP first (5-6x smaller bandwidth), falling back to PNG.
    """
    webp_path = f"{cache_base}.webp"
    png_path = f"{cache_base}.png"

    if os.path.isfile(webp_path) and os.path.getsize(webp_path) > 1000:
        with open(webp_path, "rb") as f:
            return f.read()

    if os.path.isfile(png_path) and os.path.getsize(png_path) > 1000:
        with open(png_path, "rb") as f:
            return f.read()

    os.makedirs(os.path.dirname(os.path.abspath(cache_base)), exist_ok=True)

    clean_base = image_url.rstrip("/")
    if clean_base.endswith((".png", ".webp", ".jpg")):
        clean_base = clean_base.rsplit(".", 1)[0]
    if clean_base.endswith("/high") or clean_base.endswith("/low"):
        clean_base = clean_base.rsplit("/", 1)[0]

    urls_to_try = [
        (f"{clean_base}/high.webp", webp_path),
        (f"{clean_base}/high.png", png_path),
        (f"{clean_base}/low.webp", webp_path),
        (f"{clean_base}/low.png", png_path),
    ]

    async with semaphore:
        for url, dest_path in urls_to_try:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        if len(data) > 1000:
                            with open(dest_path, "wb") as f:
                                f.write(data)
                            return data
            except Exception:
                continue

    return None


async def build_index(
    sets_filter: Optional[List[str]] = None,
    limit_cards: Optional[int] = None,
    index_path: Optional[str] = None,
    cache_dir: str = "data/images_cache",
    concurrency: int = 16,
    lang: str = "en",
    language: Optional[str] = None,
    skip_ocr: bool = False,
    skip_index: bool = False,
    ocr_workers: int = 8,
    **kwargs,
) -> None:
    """Unified pipeline: downloads missing images, builds USearch index, and caches OCR in SQLite.

    Features:
    - High-throughput async image downloading with connection pooling.
    - Automatic disk caching with WebP preference.
    - Incremental: downloads missing images, indexes unindexed cards, extracts uncached OCR.
    - Multi-threaded OCR extraction across clean card scans with batch SQLite persistence.
    """
    t_start = time.perf_counter()
    target_lang = (language or lang or "en").lower()

    if not index_path:
        index_path = "data/pokemon_cards.usearch"

    logger.info(f"Starting unified pipeline for '{target_lang.upper()}' (download + index + OCR)...")

    os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
    lang_cache_dir = os.path.join(cache_dir, target_lang)
    os.makedirs(lang_cache_dir, exist_ok=True)
    meta_path = index_path.rsplit(".", 1)[0] + "_meta.sqlite"

    card_index = USearchCardIndex(
        index_path=index_path,
        meta_path=meta_path,
        lang=target_lang,
        view_only=False,
    )
    ocr_cache = CandidateOCRCache(db_path=meta_path, cache_dir=cache_dir)

    indexed_ids = card_index.get_indexed_tcgdex_ids(lang=target_lang)
    cached_ocr_ids = ocr_cache.get_cached_ocr_ids(lang=target_lang)

    embedder = DINOv3Embedder() if not skip_index else None
    ocr_engine = CardOCR() if not skip_ocr else None

    if ocr_engine is not None:
        _using_gpu = getattr(ocr_engine, "_gpu", False)
        logger.info(
            f"[{target_lang.upper()}] OCR engine ready — "
            + ("GPU (CUDAExecutionProvider) ✓" if _using_gpu else f"CPU ({ocr_workers} threads)")
        )

    tcg = TCGdex(target_lang)
    try:
        all_sets = await tcg.set.list()
        sets_map = {}
        for s in all_sets:
            s_id = s.id
            card_count = getattr(s, "cardCount", None)
            total_in_set = str(card_count.total) if card_count and hasattr(card_count, "total") else None
            sets_map[s_id] = {
                "name": getattr(s, "name", s_id),
                "total": total_in_set,
            }
    except Exception as e:
        logger.warning(f"Could not retrieve sets summary from TCGdex ({target_lang}): {e}")
        sets_map = {}

    try:
        all_cards = await tcg.card.list()
    except Exception as e:
        logger.error(f"Could not list cards from TCGdex ({target_lang}): {e}")
        return

    logger.info(f"Retrieved {len(all_cards)} total cards from TCGdex catalog ({target_lang}).")

    cards_with_images = [
        c for c in all_cards
        if getattr(c, "image", None) and getattr(c, "id", "")
    ]
    logger.info(f"[{target_lang.upper()}] Cards with scan images: {len(cards_with_images)} (out of {len(all_cards)} total cards).")

    if sets_filter:
        filter_lower = {s.lower() for s in sets_filter}
        cards_with_images = [
            c for c in cards_with_images
            if c.id.rsplit("-", 1)[0].lower() in filter_lower
        ]
        logger.info(f"Filtered to {len(cards_with_images)} cards matching sets: {sets_filter}")

    def _needs_work(card_item) -> bool:
        cid = getattr(card_item, "id", "")
        local_id = getattr(card_item, "localId", "")
        set_id = cid.rsplit("-", 1)[0] if "-" in cid else "unknown"
        webp_p = os.path.join(lang_cache_dir, f"{set_id}_{local_id}.webp")
        png_p = os.path.join(lang_cache_dir, f"{set_id}_{local_id}.png")
        img_exists = (os.path.isfile(webp_p) and os.path.getsize(webp_p) > 1000) or (os.path.isfile(png_p) and os.path.getsize(png_p) > 1000)
        needs_down = not img_exists
        needs_idx = (not skip_index) and (cid not in indexed_ids)
        needs_ocr = (not skip_ocr) and (cid not in cached_ocr_ids)
        return needs_down or needs_idx or needs_ocr

    cards_to_process = [c for c in cards_with_images if _needs_work(c)]

    if limit_cards and len(cards_to_process) > limit_cards:
        cards_to_process = cards_to_process[:limit_cards]

    logger.info(
        f"[{target_lang.upper()}] Status: {len(indexed_ids)} indexed in USearch, {len(cached_ocr_ids)} cached with OCR. "
        f"Cards requiring processing: {len(cards_to_process)}."
    )

    if not cards_to_process:
        logger.info(f"✓ [{target_lang.upper()}] All cards are already downloaded, indexed, and OCR-cached!")
        return

    total_indexed_added = 0
    total_ocr_added = 0
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency * 2, ttl_dns_cache=300)
    headers = {"User-Agent": "DeltaCards-Indexer/2.0 (Mozilla/5.0 compatible)"}

    from tqdm.auto import tqdm

    already_done = len(cards_with_images) - len(cards_to_process)
    pbar = tqdm(
        total=len(cards_with_images),
        initial=already_done,
        desc=f"[{target_lang.upper()}] Index & OCR",
        unit="card",
        dynamic_ncols=True,
    )

    batch_size = max(16, concurrency * 2)

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        for i in range(0, len(cards_to_process), batch_size):
            batch = cards_to_process[i : i + batch_size]

            async def download_worker(c_item):
                card_id = getattr(c_item, "id", "")
                local_id = getattr(c_item, "localId", "")
                set_id = card_id.rsplit("-", 1)[0] if "-" in card_id else "unknown"
                image_base = getattr(c_item, "image", None)
                if not image_base:
                    return c_item, None, set_id, local_id

                cache_base = os.path.join(lang_cache_dir, f"{set_id}_{local_id}")
                img_data = await fetch_card_image(session, image_base, cache_base, semaphore)
                return c_item, img_data, set_id, local_id

            download_tasks = [download_worker(c) for c in batch]
            download_results = await asyncio.gather(*download_tasks, return_exceptions=True)

            batch_indexed = 0
            ocr_queue: List[Tuple[str, np.ndarray]] = []

            for res in download_results:
                if isinstance(res, Exception) or res is None:
                    continue
                card_item, img_data, set_id, local_id = res
                if not img_data:
                    continue

                card_id = getattr(card_item, "id", "")
                if not card_id:
                    continue

                card_name = getattr(card_item, "name", "")
                image_base = getattr(card_item, "image", "")

                try:
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None:
                        continue

                    # 1. Visual Indexing (USearch)
                    if not skip_index and embedder is not None and card_id not in indexed_ids:
                        set_info = sets_map.get(set_id, {})
                        set_name = set_info.get("name", set_id)
                        total_in_set = set_info.get("total")

                        embedding = embedder.embed_image(img)
                        next_key = card_index.get_next_key()

                        metadata = {
                            "tcgdex_id": card_id,
                            "lang": target_lang,
                            "name": card_name,
                            "set_id": set_id,
                            "set_name": set_name,
                            "local_id": local_id,
                            "total_cards": total_in_set,
                            "card_fraction": f"{local_id}/{total_in_set}" if total_in_set else None,
                            "rarity": getattr(card_item, "rarity", None),
                            "hp": getattr(card_item, "hp", None),
                            "image_url": f"{image_base}/high.webp",
                        }
                        card_index.add_card(next_key, embedding, metadata)
                        indexed_ids.add(card_id)
                        total_indexed_added += 1
                        batch_indexed += 1

                    # 2. OCR Extraction
                    if not skip_ocr and ocr_engine is not None and card_id not in cached_ocr_ids:
                        ocr_queue.append((card_id, img))

                except Exception as ex:
                    logger.debug(f"Error processing card {card_id}: {ex}")

            # Process OCR queue for this batch.
            # On GPU: serialize in a single thread to avoid CUDA context contention.
            # On CPU: use the configured thread pool for parallel inference.
            if ocr_queue and ocr_engine is not None:
                _using_gpu = getattr(ocr_engine, "_gpu", False)
                effective_workers = 1 if _using_gpu else min(ocr_workers, len(ocr_queue))

                def _do_ocr(t):
                    cid, cimg = t
                    try:
                        extracted = ocr_engine.extract(cimg)
                        return (cid, target_lang, extracted)
                    except Exception as ex:
                        logger.debug(f"Error running OCR on {cid}: {ex}")
                        return None

                with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                    ocr_results = list(pool.map(_do_ocr, ocr_queue))

                valid_ocr = [r for r in ocr_results if r is not None]
                if valid_ocr:
                    ocr_cache.save_cached_ocr_batch(valid_ocr)
                    for cid, _, _ in valid_ocr:
                        cached_ocr_ids.add(cid)
                    total_ocr_added += len(valid_ocr)

            pbar.update(len(batch))
            t_now = time.perf_counter() - t_start
            rate = (total_indexed_added + total_ocr_added) / max(0.001, t_now)
            pbar.set_postfix({
                "index": len(card_index),
                "ocr": len(cached_ocr_ids),
                "+idx": total_indexed_added,
                "+ocr": total_ocr_added,
                "speed": f"{rate:.1f} op/s",
            })

            if batch_indexed > 0 and (total_indexed_added % 200 < batch_size or i + batch_size >= len(cards_to_process)):
                card_index.save()

    pbar.close()
    if not skip_index and total_indexed_added > 0:
        card_index.save()

    t_elapsed = time.perf_counter() - t_start
    logger.info(
        f"✓ [{target_lang.upper()}] Finished! Added {total_indexed_added} indexed cards, {total_ocr_added} OCR extractions. "
        f"Index total: {len(card_index)}, OCR cached total: {len(cached_ocr_ids)}. Elapsed: {t_elapsed:.1f}s."
    )
