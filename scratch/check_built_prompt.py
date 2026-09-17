from delta_cards.recognition.llm_verifier import LLMCardVerifier
from delta_cards.models import OfferListing

offer = OfferListing(
    id="real_test_1",
    title="Carta Pokemon Rayquaza EX 75/108",
    description="Rayquaza EX Roaring Skies original en buen estado",
    price=12.0,
    product_type="single_card",
    link="https://wallapop.com/item/real_test_1",
)
query_ocr = {
    "full_text": "RAYQUAZA EX 170 HP Dragon Golpe Garra 30 Ascension Dragon 100 75/108",
    "hp": "170",
    "fractions": ["75/108"],
}
candidate_ocr = {
    "full_text": "RAYQUAZA EX 170 HP Basico Dragon Golpe Garra 30 Ascension Dragon 100 75/108",
    "hp": "170",
    "fractions": ["75/108"],
}
tcgdex_info = {
    "name": "Rayquaza EX",
    "local_id": "75",
    "total_cards": "108",
    "set_name": "Cielos Rugientes",
    "hp": "170",
    "attacks": [{"name": "Golpe Garra"}, {"name": "Ascensión Dragón"}],
}

verifier = LLMCardVerifier(n_threads=4)
prompt = verifier.build_verification_prompt(
    query_ocr=query_ocr,
    candidate_ocr=candidate_ocr,
    wallapop_info=offer,
    tcgdex_info=tcgdex_info,
)
print("PROMPT:")
print(prompt)
