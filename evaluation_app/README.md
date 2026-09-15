# Delta Cards Evaluation App

Run from the repository root:

```bash
uv run streamlit run evaluation_app/app.py
```

The app has two views.

- **Pipeline inspector**: run an uploaded image or dataset sample and inspect card isolation, DINOv3 embedding, visual candidates, complete OCR text, cached candidate OCR text, full-text ranking, optional LLM result, and per-stage timing.
- **Dataset**: review the ground-truth table, edit one sample, mark it fake, delete it, or add an uploaded image.

The inspector’s batch control evaluates a chosen category and reports prediction, score, and accuracy for each selected sample.