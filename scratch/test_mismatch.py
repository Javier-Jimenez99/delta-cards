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

# Test mismatch: Charizard listing vs Pikachu card
prompt_mismatch = """Card listing:
Title: Carta Pokemon Charizard VMAX 020/189 Darkness Ablaze
OCR Text: CHARIZARD VMAX 330 HP Fuego G-Max Wildfire 300 020/189

Database Candidate Card:
Name: Pikachu
Collector Number: 025/185
Set: Vivid Voltage
HP: 60

Question: Does the card listing match the database candidate card? Answer with YES or NO."""

resp = model.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a Pokemon card verification assistant. Answer strictly YES or NO."},
        {"role": "user", "content": prompt_mismatch},
    ],
    temperature=0.0,
    max_tokens=10,
)
print("Mismatch Response:", resp["choices"][0]["message"]["content"])
