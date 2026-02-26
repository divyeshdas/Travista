from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class NLLBTranslator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model_name = "facebook/nllb-200-distilled-600M"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def translate(self, text, src_lang, tgt_lang):

        # Set source language
        self.tokenizer.src_lang = src_lang

        encoded = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.device)

        # Correct way to get language token id
        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=tgt_lang_id,
            max_length=100
        )

        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]