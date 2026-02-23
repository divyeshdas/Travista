import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


VOCAB_SIZE = 8000
EN_TOKENIZER_FILE = "data/en_tokenizer.json"
FR_TOKENIZER_FILE = "data/fr_tokenizer.json"


def train_tokenizer(text_data, tokenizer_path):
    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[start]", "[end]", "[pad]"]
    )

    tokenizer.train_from_iterator(text_data, trainer=trainer)
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("[pad]"),
        pad_token="[pad]"
    )

    tokenizer.save(tokenizer_path)
    return tokenizer


def get_tokenizers(train_data):
    os.makedirs("data", exist_ok=True)

    if os.path.exists(EN_TOKENIZER_FILE) and os.path.exists(FR_TOKENIZER_FILE):
        print("Loading existing tokenizers...")
        en_tokenizer = Tokenizer.from_file(EN_TOKENIZER_FILE)
        fr_tokenizer = Tokenizer.from_file(FR_TOKENIZER_FILE)
    else:
        print("Training new tokenizers...")

        en_texts = [pair[0] for pair in train_data]
        fr_texts = [pair[1] for pair in train_data]

        en_tokenizer = train_tokenizer(en_texts, EN_TOKENIZER_FILE)
        fr_tokenizer = train_tokenizer(fr_texts, FR_TOKENIZER_FILE)

    return en_tokenizer, fr_tokenizer