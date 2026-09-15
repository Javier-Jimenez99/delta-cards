"""Constants and paths for Delta Cards Evaluation App."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

GROUND_TRUTH_PATH = os.path.join(PROJECT_ROOT, "data", "evaluation_set", "ground_truth.json")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation_set", "images")
SQLITE_PATH = os.path.join(PROJECT_ROOT, "data", "pokemon_cards_meta.sqlite")
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "images_cache")
LAST_EVAL_FILE = os.path.join(PROJECT_ROOT, "data", "evaluation_last_run.json")
DEBUG_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation_debug")
os.makedirs(DEBUG_DIR, exist_ok=True)
