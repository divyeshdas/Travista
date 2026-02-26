from nllb_translator import NLLBTranslator

translator = NLLBTranslator()

examples = [
    ("I love you", "eng_Latn", "fra_Latn"),
    ("I love you", "eng_Latn", "deu_Latn"),
    ("I love you", "eng_Latn", "spa_Latn"),
    ("I love you", "eng_Latn", "hin_Deva"),
    ("Je t'aime", "fra_Latn", "hin_Deva"),
]

for text, src, tgt in examples:
    result = translator.translate(text, src, tgt)
    print(f"\nInput: {text}")
    print(f"From {src} → {tgt}")
    print("Output:", result)