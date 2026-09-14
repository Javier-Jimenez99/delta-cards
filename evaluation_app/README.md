# Delta Cards — Evaluation App 🎴

Interactive application built with **Streamlit** to manage and audit the Pokémon card evaluation dataset (`data/evaluation_set/ground_truth.json`), visually inspect marketplace photos against official TCGdex high-resolution CDN images, and run 1-click pipeline recognition benchmarks with real-time diagnostics.

---

## 🚀 How to Run the App

From the repository root (`delta-cards`):

```bash
uv run streamlit run evaluation_app/app.py
```

Or using an activated virtual environment:

```bash
streamlit run evaluation_app/app.py
```

The application will be accessible in your browser at `http://localhost:8501`.

---

## 🛠️ Key Features

### 1. 🗂️ Dataset Manager
- **3-Column Side-by-Side Layout**:
  - **Column 1**: Marketplace input image (`use_container_width=True`).
  - **Column 2**: Canonical high-resolution reference card from TCGdex (or dedicated FAKE card placeholder).
  - **Column 3**: Collapsed panels for Listing details, Canonical TCGdex details, and Ground Truth editing.
- **Unified Dark Theme**: Seamless integration with `#1e232a` dark containers and `#374151` borders.
- **Ground Truth Editing with SQLite Search Assistant**:
  - Edit `expected_tcgdex_id`, title, or evidence notes.
  - Interactive search across 30,000+ canonical cards with 1-click **"Assign"** button.
  - Three aligned action buttons: `[Save Changes]`, `[🚫 Mark FAKE]`, and `[🗑️ Delete]`.
  - Automatic atomic backup saved to `ground_truth.json.bak`.

### 2. ⚡ Benchmark Evaluator (1-Click)
- **1-Click Execution**: Press `▶️ Run 1-Click Benchmark Evaluation`.
- **Filters**: Run on the full dataset or filter by category (`real_wallapop`, `marketplace_fakes`, etc.) with optional sample limit.
- **Real-Time KPIs**: Overall Accuracy (%), mean latency per card (ms), and breakdown table by category.
- **Deep Visual Diagnostics**:
  - Side-by-side 3-column view: Input Photo vs Expected Card vs Predicted Card.
  - Step-by-step trace: YOLO11m OBB isolation crop, Micro-OCR top/bottom strips, Top visual similarity candidates, and latency profiling.
  - Inline Ground Truth correction for failed samples.

### 3. 🔍 TCGdex Database Explorer
- Search and verify canonical cards indexed in SQLite (`data/pokemon_cards_meta.sqlite`) with official high-res images, set IDs, collection numbers, fractions, and rarities.
