"""Canonical Evaluation Runner for Pokémon TCG FGIR DINOv3 Scanner with LLM Verifier.

Evaluates the FGIR pipeline across the certified multi-category benchmark:
- Cat 1: Real Marketplace Photos (Pixel-audited)
- Cat 2: Multilingual Spanish Editions (Photorealistic smartphone captures)
- Cat 3: Canonical Multi-Version Reprints (Identical artwork across sets)
- Cat 4: Realistic Tabletop Perspective Angles (15° - 45° 3D warps)
- Cat 5: Severe Glare, Protective Sleeves & Finger Occlusions
- Cat 6: Marketplace Counterfeits & Fakes (Targeted false positive rejection)

Audits predictions through the local LLM False-Positive Verifier (20 questions).
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from delta_cards.models import OfferListing
from delta_cards.recognition.llm_verifier import LLMCardVerifier, get_llm_verifier
from delta_cards.recognition.pipeline import CardRecognitionPipeline

logging.basicConfig(level=logging.WARNING)

_eval_set_tests = Path("data/tests/evaluation_set")
if _eval_set_tests.exists():
    GROUND_TRUTH_PATH = str(_eval_set_tests / "ground_truth.json")
    IMAGES_DIR = str(_eval_set_tests / "images")
    OUTPUT_RESULTS_JSON = "data/tests/evaluation_results.json"
else:
    GROUND_TRUTH_PATH = "data/evaluation_set/ground_truth.json"
    IMAGES_DIR = "data/evaluation_set/images"
    OUTPUT_RESULTS_JSON = "data/evaluation_results.json"

console = Console()


def normalize_card_id(cid: str) -> str:
    """Normalizes card ID e.g. 'swsh4.5-44' vs 'swsh4.5-044'."""
    if not cid:
        return ""
    parts = cid.split("-")
    if len(parts) == 2:
        return f"{parts[0].lower()}-{parts[1].lstrip('0').lower()}"
    return cid.lower()


async def run_eval(
    category_filter: Optional[str] = None,
    limit: Optional[int] = None,
    threshold: float = 0.60,
    llm_min_conf: float = 70.0,
    use_llm: bool = True,
    use_gpu: bool = True,
):
    import llama_cpp
    has_gpu = use_gpu and getattr(llama_cpp, "llama_supports_gpu_offload", lambda: False)()
    gpu_label = "RTX 3070 (GPU)" if has_gpu else "CPU (8 threads)"
    llm_label = f"Active ({gpu_label}, min_conf={llm_min_conf}%)" if use_llm else "Disabled"

    console.print("\n[bold cyan]═" * 78 + "[/bold cyan]")
    console.print("[bold white]POKÉMON TCG FGIR SCANNER & LLM VERIFIER — EVALUATION BENCHMARK[/bold white]")
    console.print(f"[dim]Pipeline Threshold: {threshold:.2f} | LLM Verifier: {llm_label}[/dim]")
    console.print("[bold cyan]═" * 78 + "[/bold cyan]\n")

    if not os.path.isfile(GROUND_TRUTH_PATH):
        console.print(f"[red]Ground truth not found at '{GROUND_TRUTH_PATH}'[/red]")
        sys.exit(1)

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if category_filter:
        dataset = [d for d in dataset if d.get("category") == category_filter]

    if limit:
        dataset = dataset[:limit]

    console.print(f"Evaluating [bold green]{len(dataset)}[/bold green] certified evaluation samples.\n")

    fgir_pipe = CardRecognitionPipeline(use_fgir=True, acceptance_threshold=threshold)
    llm_verifier = (
        get_llm_verifier(
            n_threads=8,
            n_gpu_layers=-1 if has_gpu else 0,
            min_confidence=llm_min_conf,
        )
        if use_llm
        else None
    )

    cat_stats = defaultdict(lambda: {
        "count": 0,
        "base_correct": 0,
        "llm_correct": 0,
        "base_fp": 0,
        "llm_fp": 0,
        "fp_prevented_by_llm": 0,
        "false_vetoes_by_llm": 0,
        "fgir_latencies": [],
        "llm_latencies": [],
    })

    results_log = []

    for i, item in enumerate(dataset, 1):
        sid = item["sample_id"]
        cat = item["category"]
        fname = item["filename"]
        exp_id = item["expected_tcgdex_id"]
        exp_norm = normalize_card_id(exp_id)
        img_path = os.path.join(IMAGES_DIR, fname)

        img = cv2.imread(img_path)
        if img is None:
            console.print(f"[red]Image missing: {img_path}[/red]")
            continue

        t0 = time.perf_counter()
        f_res = await fgir_pipe.process_card_listing(
            img, title=item.get("title", ""), description=item.get("description", "")
        )
        t_fgir = (time.perf_counter() - t0) * 1000

        base_pred = (f_res.get("verified_card") or {}).get("id", "")
        conf = float(f_res.get("confidence", 0.0))

        # Determine baseline correctness
        is_fake = (exp_norm == "fake")
        if is_fake:
            base_correct = (not base_pred) or (f_res.get("status") != "success")
            base_fp = not base_correct
        else:
            base_correct = normalize_card_id(base_pred) == exp_norm and bool(exp_norm)
            base_fp = bool(base_pred) and not base_correct

        # Run LLM Verification step on any selected card candidate
        final_pred = base_pred
        llm_res = None
        t_llm = 0.0

        if base_pred and f_res.get("verified_card") and llm_verifier:
            eval_offer = OfferListing(
                id=sid,
                title=item.get("title") or f"Carta Pokemon {item.get('name', '')}",
                description=item.get("description", "") or "",
                price=item.get("price", 10.0),
                product_type="single_card",
                link=f"https://es.wallapop.com/item/{sid}",
                tcgdex_id=base_pred,
            )
            t_llm0 = time.perf_counter()
            llm_res = llm_verifier.verify_match(
                offer=eval_offer,
                query_ocr=f_res.get("query_ocr"),
                candidate_ocr=f_res.get("candidate_ocr"),
                tcgdex_card=f_res.get("verified_card"),
            )
            t_llm = (time.perf_counter() - t_llm0) * 1000
            cat_stats[cat]["llm_latencies"].append(t_llm)

            if not llm_res.is_verified:
                final_pred = ""  # Vetoed / blocked by LLM!

        # Determine LLM-verified correctness
        if is_fake:
            llm_correct = (not final_pred)
            llm_fp = not llm_correct
        else:
            llm_correct = normalize_card_id(final_pred) == exp_norm and bool(exp_norm)
            llm_fp = bool(final_pred) and not llm_correct

        # Track LLM impacts
        fp_prevented = base_fp and not llm_fp
        false_veto = base_correct and not llm_correct

        cat_stats[cat]["count"] += 1
        cat_stats[cat]["fgir_latencies"].append(t_fgir)
        if base_correct:
            cat_stats[cat]["base_correct"] += 1
        if llm_correct:
            cat_stats[cat]["llm_correct"] += 1
        if base_fp:
            cat_stats[cat]["base_fp"] += 1
        if llm_fp:
            cat_stats[cat]["llm_fp"] += 1
        if fp_prevented:
            cat_stats[cat]["fp_prevented_by_llm"] += 1
        if false_veto:
            cat_stats[cat]["false_vetoes_by_llm"] += 1

        # Format status message
        if is_fake:
            if fp_prevented:
                status_str = f"[bold yellow]🛡️ FP PREVENTED BY LLM[/bold yellow] (Blocked {base_pred})"
            elif llm_correct:
                status_str = "[bold green]✓ FAKE REJECTED[/bold green]"
            else:
                status_str = f"[red]✗ FALSE POSITIVE ({final_pred})[/red]"
        else:
            if llm_correct:
                score_str = f" [cyan]LLM={llm_res.confidence_score:.0f}%[/cyan]" if llm_res else ""
                status_str = f"[bold green]✓ MATCH[/bold green] ({final_pred}){score_str}"
            elif fp_prevented:
                status_str = f"[bold yellow]🛡️ WRONG MATCH BLOCKED BY LLM[/bold yellow] ({base_pred})"
            elif false_veto:
                status_str = f"[bold magenta]⚠️ FALSE VETO BY LLM[/bold magenta] ({base_pred})"
            else:
                status_str = f"[red]✗ MISS ({final_pred or base_pred or 'No match'})[/red]"

        lat_str = f"{t_fgir:5.0f}ms"
        if t_llm > 0:
            lat_str += f" + {t_llm:5.0f}ms(LLM)"

        console.print(
            f"[{i:02d}/{len(dataset):02d}] [cyan]{cat[:22]:22s}[/cyan] | "
            f"Exp=[yellow]{exp_id:12s}[/yellow] | {status_str} | {lat_str}"
        )

        results_log.append({
            "sample_id": sid,
            "category": cat,
            "expected_id": exp_id,
            "base_predicted_id": base_pred,
            "final_predicted_id": final_pred,
            "base_correct": base_correct,
            "llm_correct": llm_correct,
            "fp_prevented": fp_prevented,
            "false_veto": false_veto,
            "fgir_latency_ms": round(t_fgir, 1),
            "llm_latency_ms": round(t_llm, 1) if t_llm > 0 else None,
            "llm_confidence": llm_res.confidence_score if llm_res else None,
            "llm_verified": llm_res.is_verified if llm_res else None,
            "llm_summary": llm_res.verdict_summary if llm_res else None,
        })

    # Summary Table
    table = Table(
        title=f"\nComprehensive Benchmark Breakdown (Pipeline Thresh={threshold:.2f}, LLM={llm_label})",
        header_style="bold cyan",
    )
    table.add_column("Category", style="bold white")
    table.add_column("Samples", justify="center")
    table.add_column("Base Acc", justify="center")
    table.add_column("LLM Acc", justify="center", style="bold green")
    table.add_column("FP Baseline", justify="center")
    table.add_column("FP with LLM", justify="center", style="bold green")
    table.add_column("LLM Shielded", justify="center", style="bold yellow")
    table.add_column("Mean GPU Lat", justify="right", style="green")

    all_fgir_lat = []
    all_llm_lat = []
    tot_samples = 0
    tot_base_correct = 0
    tot_llm_correct = 0
    tot_base_fp = 0
    tot_llm_fp = 0
    tot_shielded = 0

    category_summary = {}
    for cat, d in cat_stats.items():
        n = d["count"]
        tot_samples += n
        tot_base_correct += d["base_correct"]
        tot_llm_correct += d["llm_correct"]
        tot_base_fp += d["base_fp"]
        tot_llm_fp += d["llm_fp"]
        tot_shielded += d["fp_prevented_by_llm"]

        b_acc = (d["base_correct"] / n * 100) if n > 0 else 0.0
        l_acc = (d["llm_correct"] / n * 100) if n > 0 else 0.0
        mean_llm_ms = float(np.mean(d["llm_latencies"])) if d["llm_latencies"] else 0.0

        all_fgir_lat.extend(d["fgir_latencies"])
        all_llm_lat.extend(d["llm_latencies"])

        category_summary[cat] = {
            "count": n,
            "base_accuracy": round(b_acc, 2),
            "llm_accuracy": round(l_acc, 2),
            "base_fp": d["base_fp"],
            "llm_fp": d["llm_fp"],
            "fp_prevented": d["fp_prevented_by_llm"],
            "mean_llm_latency_ms": round(mean_llm_ms, 1),
        }

        table.add_row(
            cat.replace("_", " ").title(),
            str(n),
            f"{b_acc:.1f}% ({d['base_correct']}/{n})",
            f"{l_acc:.1f}% ({d['llm_correct']}/{n})",
            str(d["base_fp"]),
            str(d["llm_fp"]),
            str(d["fp_prevented_by_llm"]),
            f"{mean_llm_ms:.0f} ms" if mean_llm_ms > 0 else "-",
        )

    overall_b_acc = (tot_base_correct / tot_samples * 100) if tot_samples > 0 else 0.0
    overall_l_acc = (tot_llm_correct / tot_samples * 100) if tot_samples > 0 else 0.0
    overall_llm_lat = float(np.mean(all_llm_lat)) if all_llm_lat else 0.0

    # Calculate Precisions: TP / (TP + FP)
    # Exclude pure fake samples from genuine denominator
    genuine_samples = [item for item in results_log if item["expected_id"] != "fake"]
    n_genuine = len(genuine_samples)
    base_tp = sum(1 for x in genuine_samples if x["base_correct"])
    llm_tp = sum(1 for x in genuine_samples if x["llm_correct"])

    base_precision = (base_tp / (base_tp + tot_base_fp) * 100) if (base_tp + tot_base_fp) > 0 else 0.0
    llm_precision = (llm_tp / (llm_tp + tot_llm_fp) * 100) if (llm_tp + tot_llm_fp) > 0 else 0.0

    table.add_section()
    table.add_row(
        "[bold]TOTAL / OVERALL[/bold]",
        f"[bold]{tot_samples}[/bold]",
        f"[bold]{overall_b_acc:.1f}%[/bold]",
        f"[bold green]{overall_l_acc:.1f}%[/bold green]",
        f"[bold red]{tot_base_fp}[/bold red]",
        f"[bold green]{tot_llm_fp}[/bold green]",
        f"[bold yellow]{tot_shielded}[/bold yellow]",
        f"[bold green]{overall_llm_lat:.0f} ms[/bold green]",
    )

    console.print(table)

    # Metric comparison callout
    console.print(
        f"\n[bold white]═══ IMPACTO DEL VERIFICADOR LLM ANTI-FALSOS POSITIVOS ═══[/bold white]\n"
        f"  • [bold]Precisión del Sistema:[/] {base_precision:.1f}% → [bold green]{llm_precision:.1f}%[/bold green] "
        f"([green]+{llm_precision - base_precision:.1f}%[/green])\n"
        f"  • [bold]Falsos Positivos Totales:[/] {tot_base_fp} → [bold green]{tot_llm_fp}[/bold green] "
        f"([yellow]{tot_shielded} ofertas falsas bloqueadas antes de notificar[/yellow])\n"
        f"  • [bold]Latencia Media LLM (GPU):[/bold] [bold cyan]{overall_llm_lat:.1f} ms[/bold cyan]\n"
    )

    # Save JSON results
    os.makedirs(os.path.dirname(OUTPUT_RESULTS_JSON), exist_ok=True)
    with open(OUTPUT_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": tot_samples,
            "threshold": threshold,
            "use_llm": use_llm,
            "base_precision": round(base_precision, 2),
            "llm_precision": round(llm_precision, 2),
            "overall_base_accuracy": round(overall_b_acc, 2),
            "overall_llm_accuracy": round(overall_l_acc, 2),
            "mean_llm_latency_ms": round(overall_llm_lat, 1),
            "total_false_positives_prevented": tot_shielded,
            "category_breakdown": category_summary,
            "detailed_results": results_log,
        }, f, indent=2)

    console.print(f"[green]✓ Guardados resultados detallados en '{OUTPUT_RESULTS_JSON}'[/green]\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Pokémon TCG FGIR DINOv3 Scanner with LLM Verifier")
    parser.add_argument("--category", type=str, default=None, help="Filter by category name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples")
    parser.add_argument("--threshold", type=float, default=0.60, help="Pipeline acceptance threshold (default: 0.60)")
    parser.add_argument("--llm-min-conf", type=float, default=70.0, help="LLM verifier minimum confidence (default: 70.0)")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM verification")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU offloading for LLM")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_eval(
            category_filter=args.category,
            limit=args.limit,
            threshold=args.threshold,
            llm_min_conf=args.llm_min_conf,
            use_llm=not args.no_llm,
            use_gpu=not args.no_gpu,
        )
    )
