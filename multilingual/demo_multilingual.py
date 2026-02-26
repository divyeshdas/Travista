from nllb_translator import NLLBTranslator


def main():
    translator = NLLBTranslator()

    examples = [
        ("I love you", "eng_Latn", "fra_Latn"),
        ("I love you", "eng_Latn", "deu_Latn"),
        ("I love you", "eng_Latn", "spa_Latn"),
        ("I love you", "eng_Latn", "hin_Deva"),
        ("Je t'aime", "fra_Latn", "hin_Deva"),
        ("Ich liebe dich", "deu_Latn", "eng_Latn"),
        ("Te quiero", "spa_Latn", "eng_Latn"),
        ("मैं तुमसे प्यार करता हूँ", "hin_Deva", "eng_Latn"),
    ]

    for text, src, tgt in examples:
        print("\n" + "=" * 60)
        print(f"Input Text : {text}")
        print(f"Source Lang: {src}")
        print(f"Target Lang: {tgt}")

        result = translator.translate(text, src, tgt)

        print("Output     :", result)


if __name__ == "__main__":
    main()