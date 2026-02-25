from utils.dataset import load_text_pairs, split_dataset
from utils.tokenizer import get_tokenizers
from utils.dataloader import get_dataloader
from models.transformer_model import TransformerModel
from training.train_transformer import train_transformer

import torch


if __name__ == "__main__":

    # ==========================
    # Load dataset
    # ==========================
    text_pairs = load_text_pairs()
    train_data, val_data, test_data = split_dataset(text_pairs)

    # ==========================
    # Load tokenizers
    # ==========================
    en_tokenizer, fr_tokenizer = get_tokenizers(train_data)

    # ==========================
    # Create DataLoaders
    # ==========================
    train_loader = get_dataloader(train_data, en_tokenizer, fr_tokenizer)
    val_loader = get_dataloader(val_data, en_tokenizer, fr_tokenizer)

    # Padding token
    pad_token = fr_tokenizer.token_to_id("[pad]")

    # ==========================
    # Device
    # ==========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ==========================
    # Vocabulary sizes
    # ==========================
    input_dim = len(en_tokenizer.get_vocab())
    output_dim = len(fr_tokenizer.get_vocab())

    # ==========================
    # Build Transformer Model
    # ==========================
    model = TransformerModel(
        src_vocab_size=input_dim,
        tgt_vocab_size=output_dim,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1
    ).to(device)

    print("Transformer Model created successfully.")
    print("Total parameters:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    # ==========================
    # Train Transformer
    # ==========================
    train_transformer(
        model,
        train_loader,
        val_loader,
        pad_token,
        device,
        epochs=5
    )