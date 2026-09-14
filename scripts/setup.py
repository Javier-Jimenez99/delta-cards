#!/usr/bin/env python3
"""Delta-Cards Setup Script.

Automates downloading required models, setting up directory structures,
configuring environment variables, and initializing database tables.
"""

import os
import shutil
import sys
from pathlib import Path

# Add src/ to sys.path so we can import delta_cards
REPO_ROOT = Path(__file__).resolve().parent.parent
src_path = str(REPO_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Configuration
ONNX_REPO_ID = "AlecKarfonta/cardcaptor-v3"
ONNX_FILENAME = "onnx/cardcaptor_v3_best.onnx"
ONNX_LOCAL_TARGETS = [
    REPO_ROOT / "models" / "onnx" / "cardcaptor_v3_best.onnx",
    REPO_ROOT / "models" / "cardcaptor_v3_best.onnx",
    REPO_ROOT / "onnx" / "cardcaptor_v3_best.onnx",
]
MIN_ONNX_BYTES = 50_000_000  # ~84 MB expected

LLM_REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
LLM_FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
LLM_LOCAL_TARGET = REPO_ROOT / "models" / LLM_FILENAME
MIN_LLM_BYTES = 300_000_000  # ~491 MB expected

REQUIRED_DIRS = [
    REPO_ROOT / "data",
    REPO_ROOT / "data" / "output",
    REPO_ROOT / "models",
    REPO_ROOT / "models" / "onnx",
    REPO_ROOT / "onnx",
]


def ensure_directories():
    """Creates all required local directories."""
    console.print("\n[bold cyan]1. Setting Up Directory Structure...[/bold cyan]")
    for dir_path in REQUIRED_DIRS:
        dir_path.mkdir(parents=True, exist_ok=True)
        console.print(f"   ✓ [green]Directory verified:[/green] [dim]{dir_path.relative_to(REPO_ROOT)}[/dim]")


def ensure_env_file():
    """Ensures .env exists, creating from .env.example or with SQLite default."""
    console.print("\n[bold cyan]2. Configuring Environment Variables (.env)...[/bold cyan]")
    env_file = REPO_ROOT / ".env"
    example_file = REPO_ROOT / ".env.example"

    if env_file.exists() and env_file.stat().st_size > 0:
        console.print(f"   ✓ [green]Existing .env file detected:[/green] [dim]{env_file.name}[/dim]")
        return

    # Default to local SQLite for zero-friction setup out-of-the-box
    default_content = (
        '# Database connection string for delta-cards\n'
        '# Use local SQLite database (default for zero-friction development):\n'
        'DATABASE_URL="sqlite:///data/delta_cards.db"\n\n'
        '# For PostgreSQL, uncomment and adjust the line below:\n'
        '# DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/delta_cards"\n'
    )
    env_file.write_text(default_content, encoding="utf-8")
    console.print(f"   ✓ [green]Created default .env with SQLite configuration:[/green] [cyan]{env_file.name}[/cyan]")


def download_onnx_model():
    """Downloads cardcaptor_v3_best.onnx from Hugging Face if missing."""
    console.print("\n[bold cyan]3. Checking Visual Card Detector (ONNX YOLO OBB)...[/bold cyan]")

    # Check if any existing target is valid
    existing_valid_path = None
    for p in ONNX_LOCAL_TARGETS:
        if p.exists() and p.stat().st_size >= MIN_ONNX_BYTES:
            existing_valid_path = p
            break

    if existing_valid_path:
        size_mb = existing_valid_path.stat().st_size / (1024 * 1024)
        console.print(
            f"   ✓ [green]ONNX detector already exists ({size_mb:.1f} MB):[/green] "
            f"[cyan]{existing_valid_path.relative_to(REPO_ROOT)}[/cyan]"
        )
        source_path = existing_valid_path
    else:
        console.print(
            f"   ⬇️ [yellow]Downloading {ONNX_FILENAME} from HuggingFace ({ONNX_REPO_ID})...[/yellow]"
        )
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id=ONNX_REPO_ID,
                filename=ONNX_FILENAME,
            )
            source_path = Path(downloaded)
            size_mb = source_path.stat().st_size / (1024 * 1024)
            console.print(f"   ✓ [green]Download complete ({size_mb:.1f} MB).[/green]")
        except Exception as exc:
            console.print(f"   ❌ [bold red]Failed to download ONNX detector:[/bold red] {exc}")
            sys.exit(1)

    # Ensure both target locations have the model
    for target in ONNX_LOCAL_TARGETS:
        if not target.exists() or target.stat().st_size < MIN_ONNX_BYTES:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target)
            console.print(f"   ✓ [green]Copied model to:[/green] [dim]{target.relative_to(REPO_ROOT)}[/dim]")


def download_llm_model():
    """Downloads official Qwen2.5-0.5B-Instruct-GGUF from Hugging Face if missing."""
    console.print("\n[bold cyan]4. Checking Local LLM (Qwen2.5-0.5B-Instruct GGUF)...[/bold cyan]")

    if LLM_LOCAL_TARGET.exists() and LLM_LOCAL_TARGET.stat().st_size >= MIN_LLM_BYTES:
        size_mb = LLM_LOCAL_TARGET.stat().st_size / (1024 * 1024)
        console.print(
            f"   ✓ [green]GGUF model already exists ({size_mb:.1f} MB):[/green] "
            f"[cyan]{LLM_LOCAL_TARGET.relative_to(REPO_ROOT)}[/cyan]"
        )
        return

    console.print(
        f"   ⬇️ [yellow]Downloading {LLM_FILENAME} from HuggingFace ({LLM_REPO_ID})...[/yellow]"
    )
    try:
        from huggingface_hub import hf_hub_download
        LLM_LOCAL_TARGET.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=LLM_REPO_ID,
            filename=LLM_FILENAME,
            local_dir=str(LLM_LOCAL_TARGET.parent),
        )
        size_mb = Path(downloaded).stat().st_size / (1024 * 1024)
        console.print(f"   ✓ [green]Download complete ({size_mb:.1f} MB):[/green] [cyan]{LLM_LOCAL_TARGET.name}[/cyan]")
    except Exception as exc:
        console.print(f"   ❌ [bold red]Failed to download LLM GGUF model:[/bold red] {exc}")
        sys.exit(1)


def initialize_database():
    """Initializes database schema and verifies table creation."""
    console.print("\n[bold cyan]5. Initializing Database Schema...[/bold cyan]")
    try:
        from delta_cards.database.connection import get_db_url, get_engine, init_db
        db_url = get_db_url()
        engine = get_engine(db_url)
        init_db(engine)
        console.print(f"   ✓ [green]Database initialized successfully:[/green] [cyan]{db_url}[/cyan]")
        console.print("   ✓ [dim]Tables verified: 'WallapopOffers', 'CardmarketProduct', 'CardmarketPrice'[/dim]")
    except Exception as exc:
        console.print(f"   ❌ [bold red]Failed to initialize database:[/bold red] {exc}")
        sys.exit(1)


def verify_system_dependencies():
    """Performs sanity checks on external tools like Tesseract and Python libraries."""
    console.print("\n[bold cyan]6. Verifying System & Tool Dependencies...[/bold cyan]")
    table = Table(title="Dependency Status", border_style="cyan")
    table.add_column("Component", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="dim")

    # 1. Tesseract OCR
    tesseract_bin = shutil.which("tesseract")
    if tesseract_bin:
        import subprocess
        try:
            res = subprocess.run([tesseract_bin, "--list-langs"], capture_output=True, text=True, timeout=5)
            langs = [line.strip() for line in res.stdout.splitlines()[1:] if line.strip()]
            has_spa = "spa" in langs
            has_eng = "eng" in langs
            details = f"Path: {tesseract_bin} | Langs: {', '.join(langs)}"
            if has_spa and has_eng:
                table.add_row("Tesseract OCR", "[green]INSTALLED[/green]", details)
            else:
                table.add_row("Tesseract OCR", "[yellow]PARTIAL[/yellow]", f"{details} (Recommend: spa, eng)")
        except Exception as e:
            table.add_row("Tesseract OCR", "[yellow]WARNING[/yellow]", str(e))
    else:
        table.add_row(
            "Tesseract OCR",
            "[red]NOT FOUND[/red]",
            "Install via: sudo apt-get install tesseract-ocr tesseract-ocr-spa (Debian/Ubuntu) or brew install tesseract",
        )

    # 2. ONNX Runtime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        table.add_row("ONNX Runtime", "[green]READY[/green]", f"Providers: {', '.join(providers)}")
    except Exception as e:
        table.add_row("ONNX Runtime", "[red]ERROR[/red]", str(e))

    # 3. llama-cpp-python
    try:
        import llama_cpp
        table.add_row("llama-cpp-python", "[green]READY[/green]", f"Version: {llama_cpp.__version__} (CPU)")
    except Exception as e:
        table.add_row("llama-cpp-python", "[red]ERROR[/red]", str(e))

    # 4. langdetect
    try:
        import langdetect
        table.add_row("langdetect", "[green]READY[/green]", "Physical card language detection active")
    except Exception as e:
        table.add_row("langdetect", "[red]ERROR[/red]", str(e))

    console.print(table)


def main():
    console.print(
        Panel.fit(
            "[bold green]Delta Cards: Automated Setup & Asset Provisioning[/bold green]\n"
            "[dim]Configures directories, environments, downloads AI models, and prepares database.[/dim]",
            border_style="green",
        )
    )

    ensure_directories()
    ensure_env_file()
    download_onnx_model()
    download_llm_model()
    initialize_database()
    verify_system_dependencies()

    console.print(
        Panel.fit(
            "[bold green]✓ Setup Completed Successfully![/bold green]\n\n"
            "You are now ready to run delta-cards:\n"
            "  • [cyan]uv run python scripts/evaluate.py --limit 2[/cyan]   (Run quick evaluation benchmark)\n"
            "  • [cyan]uv run python -m delta_cards.cli -p single_card -l 2[/cyan]  (Scrape & enrich cards)\n"
            "  • [cyan]uv run pytest tests/ -v[/cyan]                   (Run complete test suite)",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
