import os
from llama_cpp import Llama

model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
model = Llama(model_path=model_path, n_ctx=2048, n_threads=4, n_gpu_layers=0, verbose=False)

# Mismatched prompt: Charizard vs Pikachu
prompt = """Examine this Pokémon card match to determine if it is genuine or a false positive:
[Photo OCR]: Text: CHARIZARD VMAX 330 HP Fuego G-Max Wildfire 300 020/189 | HP: 330 | Fractions: 020/189
[Candidate Official Scan]: Text: PIKACHU 60 HP Rayo Pika Ball 20 025/185 | HP: 60 | Fractions: 025/185
[Wallapop Listing]:
Title: Carta Pokemon Charizard VMAX 020/189
Description: Charizard VMAX Darkness Ablaze original
Price: 50.00 EUR
[Official TCGdex Card]: Name: Pikachu | Number: 025/185 | Set: Vivid Voltage | HP: 60 | Attacks: Pika Ball

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
print("MISMATCH OUTPUT WITH ORIGINAL PROMPT:")
print(resp["choices"][0]["message"]["content"])
