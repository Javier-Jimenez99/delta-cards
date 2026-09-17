import os, json
from llama_cpp import Llama

model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
model = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False,
)

prompt = """Examine this Pokémon card match:
[Photo OCR]: Text: RAYQUAZA EX 170 HP Dragon Golpe Garra 30 Ascension Dragon 100 75/108 | HP: 170 | Fractions: 75/108
[Candidate Official Scan]: Text: RAYQUAZA EX 170 HP Basico Dragon Golpe Garra 30 Ascension Dragon 100 75/108 | HP: 170 | Fractions: 75/108
[Wallapop Listing]:
Title: Carta Pokemon Rayquaza EX 75/108
Description: Rayquaza EX Roaring Skies original en buen estado
Price: 12.00 EUR
[Official TCGdex Card]: Name: Rayquaza EX | Number: 75/108 | Set: Cielos Rugientes | HP: 170 | Attacks: Golpe Garra, Ascensión Dragón

Questions:
1. Pokemon name match: 1 or 0
2. Stage or evolution match: 1 or 0
3. HP match: 1 or 0
4. Card element consistent: 1 or 0
5. Collector number match: 1 or 0
6. Total card count match: 1 or 0
7. Promo code aligns: 1 or 0
8. Set expansion consistent: 1 or 0
9. Primary attack name matches: 1 or 0
10. Attack damage matches: 1 or 0
11. Secondary attack matches: 1 or 0
12. Rules consistent: 1 or 0
13. OCR text subset valid: 1 or 0
14. Special mechanic matches: 1 or 0
15. Rarity consistent: 1 or 0
16. No contradictory pokemon: 1 or 0
17. Title mentions card: 1 or 0
18. Description consistent: 1 or 0
19. Single card not lot: 1 or 0
20. Genuine match verdict: 1 or 0
"""

# Let's test standard prompt with assistant prefill or clean instructions
messages = [
    {
        "role": "system",
        "content": (
            "You are an expert Pokemon card verification auditor. "
            "Analyze the evidence and evaluate each question (1 for true/match, 0 for false/mismatch). "
            "Respond ONLY with a valid JSON list of 20 integers, for example:\n"
            "[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]"
        ),
    },
    {"role": "user", "content": prompt},
]

resp = model.create_chat_completion(
    messages=messages,
    temperature=0.0,
    max_tokens=80,
)
print("RAW CONTENT:")
print(resp["choices"][0]["message"]["content"])
