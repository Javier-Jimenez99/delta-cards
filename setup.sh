#!/usr/bin/env bash
# ==============================================================================
# Delta Cards: One-Command Setup Script
# Installs all system/python dependencies, downloads AI models, and sets up DB
# ==============================================================================

set -e

# ANSI Color codes
BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${BOLD}${CYAN}==============================================================================${RESET}"
echo -e "${BOLD}${GREEN} 🃏 Delta Cards: Automated Environment & Asset Setup${RESET}"
echo -e "${BOLD}${CYAN}==============================================================================${RESET}"

# Ensure we are in the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Check & Install uv if missing
echo -e "\n${BOLD}[1/5] Checking uv package manager...${RESET}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv &>/dev/null; then
    echo -e "${YELLOW}uv not found. Installing astral-sh/uv automatically...${RESET}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if command -v uv &>/dev/null; then
    UV_VERSION=$(uv --version)
    echo -e "   ✓ ${GREEN}uv is installed:${RESET} ${UV_VERSION}"
else
    echo -e "   ${RED}❌ Error: Failed to install uv. Please install uv manually: https://docs.astral.sh/uv/${RESET}"
    exit 1
fi

# 2. Check Tesseract OCR
echo -e "\n${BOLD}[2/5] Checking Tesseract OCR engine...${RESET}"
if command -v tesseract &>/dev/null; then
    TESS_VER=$(tesseract --version 2>&1 | head -n 1)
    echo -e "   ✓ ${GREEN}Tesseract detected:${RESET} ${TESS_VER}"
else
    echo -e "   ${YELLOW}⚠️  Tesseract OCR is not installed or not in PATH.${RESET}"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo -e "   👉 To install on Debian/Ubuntu, run:"
        echo -e "      ${CYAN}sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng${RESET}"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo -e "   👉 To install on macOS, run:"
        echo -e "      ${CYAN}brew install tesseract tesseract-lang${RESET}"
    fi
fi

# 3. Synchronize Python Environment & Dependencies
echo -e "\n${BOLD}[3/5] Synchronizing Python virtual environment with uv...${RESET}"
uv sync

# 4. Run Asset Provisioning & Database Setup Script
echo -e "\n${BOLD}[4/5] Downloading models and preparing database...${RESET}"
uv run python scripts/setup.py

# 5. Quick Sanity Verification
echo -e "\n${BOLD}[5/5] Running quick test suite sanity check...${RESET}"
uv run pytest tests/test_recognition.py -k test_detect_card_language -q

echo -e "\n${BOLD}${GREEN}==============================================================================${RESET}"
echo -e "${BOLD}${GREEN} 🎉 ALL DONE! Delta Cards is fully installed and ready to use.${RESET}"
echo -e "${BOLD}${GREEN}==============================================================================${RESET}"
echo -e "\nTry running:"
echo -e "  • Quick evaluation benchmark on live samples: ${CYAN}uv run python scripts/evaluate.py --limit 2${RESET}"
echo -e "  • Scrape & enrich cards into database:        ${CYAN}uv run python -m delta_cards.cli -p single_card -l 2${RESET}"
echo -e "  • Run full automated test suite:             ${CYAN}uv run pytest tests/ -v${RESET}\n"
