import os, time
from llama_cpp import Llama

model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
model = Llama(
    model_path=model_path,
    n_ctx=512,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False,
)

prompt = """Card listing:
Title: Carta Pokemon Rayquaza EX 75/108
OCR Text: RAYQUAZA EX 170 HP Dragon Golpe Garra 30 Ascension Dragon 100 75/108

Database Candidate Card:
Name: Rayquaza EX
Collector Number: 75/108
Set: Roaring Skies (Cielos Rugientes)
HP: 170

Question: Does the card listing match the database candidate card? Answer with YES or NO."""

t0 = time.time()
resp = model.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a Pokemon card verification assistant. Answer only YES or NO."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.0,
    max_tokens=10,
)
t1 = time.time()
print(f"Time: {t1 - t0:.2f}s")
print("Response:", resp["choices"][0]["message"]["content"])
