import os
from llama_cpp import Llama

model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
model = Llama(model_path=model_path, n_ctx=2048, n_threads=4, n_gpu_layers=0, verbose=False)

# Genuine prompt: Rayquaza EX vs Rayquaza EX
prompt = """Examine this Pokémon card match to determine if it is genuine or a false positive:
[Photo OCR]: Text: RAYQUAZA EX 170 HP Dragon Golpe Garra 30 Ascension Dragon 100 75/108 | HP: 170 | Fractions: 75/108
[Candidate Official Scan]: Text: RAYQUAZA EX 170 HP Basico Dragon Golpe Garra 30 Ascension Dragon 100 75/108 | HP: 170 | Fractions: 75/108
[Wallapop Listing]:
Title: Carta Pokemon Rayquaza EX 75/108
Description: Rayquaza EX Roaring Skies original en buen estado
Price: 12.00 EUR
[Official TCGdex Card]: Name: Rayquaza EX | Number: 75/108 | Set: Cielos Rugientes | HP: 170 | Attacks: Golpe Garra, Ascensión Dragón

Evaluate each of the following 20 verification questions with 1 (confirmed/yes) or 0 (failed/no):
1. Pokemon name in photo OCR matches TCGdex candidate name
2. Stage or evolution matches
3. HP value matches
4. Card element consistent
5. Collector number matches
6. Total card count matches
7. Promo code aligns
8. Set name matches
9. Primary attack name matches
10. Attack damage matches
11. Secondary attack matches
12. Rules consistent
13. OCR text subset valid
14. Card mechanic matches
15. Rarity consistent
16. No contradictory Pokemon
17. Title mentions card
18. Description consistent
19. Single card not lot
20. Overall verdict genuine match

Output strictly the 20 answers as comma-separated digits (0 or 1), in exact order without extra words:"""

messages = [
    {
        "role": "system",
        "content": (
            "You are a Pokémon card verification auditor. "
            "Output strictly 20 comma-separated digits (0 or 1), without any other text or explanation: "
            "1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1"
        ),
    },
    {"role": "user", "content": prompt},
]

resp = model.create_chat_completion(messages=messages, temperature=0.0, max_tokens=150)
print("GENUINE OUTPUT WITH FORMAT EXAMPLE:")
print(resp["choices"][0]["message"]["content"])
