import os
from llama_cpp import Llama

model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
model = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    n_gpu_layers=0,
    verbose=False,
)

# Test 1: JSON array format request
prompt = """Examine this Pokémon card match:
[Photo OCR]: Text: RAYQUAZA EX 170 HP Dragon Golpe Garra 30 Ascension Dragon 100 75/108 | HP: 170 | Fractions: 75/108
[Candidate Official Scan]: Text: RAYQUAZA EX 170 HP Basico Dragon Golpe Garra 30 Ascension Dragon 100 75/108 | HP: 170 | Fractions: 75/108
[Wallapop Listing]:
Title: Carta Pokemon Rayquaza EX 75/108
Description: Rayquaza EX Roaring Skies original en buen estado
Price: 12.00 EUR
[Official TCGdex Card]: Name: Rayquaza EX | Number: 75/108 | Set: Cielos Rugientes | HP: 170 | Attacks: Golpe Garra, Ascensión Dragón

Evaluate each question with 1 (yes/confirmed) or 0 (no/failed):
1. Pokemon name in photo OCR matches TCGdex candidate name
2. Stage or evolution (Basic, Stage 1, Stage 2, V, VMAX, ex) matches
3. HP value in photo OCR matches or aligns with official card
4. Card element or energy type is consistent
5. Local collector number (e.g. 75, 025) matches
6. Set total card count (e.g. /108, /198) matches
7. Promo code aligns or not applicable
8. Set name or expansion abbreviation matches or aligns
9. Primary attack name appears in photo OCR
10. Primary attack damage number matches
11. Secondary attack or special ability appears in photo OCR
12. Card rules (e.g. Pokemon V Rule) or retreat cost consistent
13. Photo OCR text is a plausible subset of official scan OCR
14. Special mechanics (EX, GX, V, VMAX, Tera, Radiant) match
15. Card rarity indicator is consistent with listing and card
16. No conflicting or contradictory Pokemon name appears in photo
17. Wallapop listing title references this card or its collector number
18. Wallapop description is consistent with the card and does not describe another item
19. Listing is for a single individual card (not a bulk lot, tin, or empty box)
20. Overall verdict: this is confirmed beyond doubt to be a genuine match
"""

messages = [
    {
        "role": "system",
        "content": "You are a Pokémon card verification auditor. Respond with a JSON array of 20 integers (0 or 1) indicating whether each question is satisfied. Example format: [1, 1, 0, 1, ...]",
    },
    {"role": "user", "content": prompt + "\nOutput the JSON array:"},
]

resp = model.create_chat_completion(messages=messages, temperature=0.0, max_tokens=100)
print("RESPONSE:")
print(resp["choices"][0]["message"]["content"])
