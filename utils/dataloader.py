import torch
from torch.utils.data import DataLoader
from utils.dataset import TranslationDataset


def collate_fn(batch, en_tokenizer, fr_tokenizer):
    en_sentences = []
    fr_sentences = []

    for eng, fra in batch:
        en_sentences.append(eng)
        fr_sentences.append("[start] " + fra + " [end]")

    en_encoded = en_tokenizer.encode_batch(en_sentences)
    fr_encoded = fr_tokenizer.encode_batch(fr_sentences)

    en_ids = [enc.ids for enc in en_encoded]
    fr_ids = [enc.ids for enc in fr_encoded]

    return torch.tensor(en_ids), torch.tensor(fr_ids)


def get_dataloader(data, en_tokenizer, fr_tokenizer, batch_size=32, shuffle=True):
    dataset = TranslationDataset(data)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, en_tokenizer, fr_tokenizer)
    )