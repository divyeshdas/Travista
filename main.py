from utils.dataset import load_text_pairs, split_dataset
from utils.tokenizer import get_tokenizers
from utils.dataloader import get_dataloader

import torch
from models.lstm import Encoder, Decoder, Seq2Seq
from models.lstm_attention import Encoder, Decoder, Seq2Seq, Attention

if __name__ == "__main__":
    # Load dataset
    text_pairs = load_text_pairs()
    train_data, val_data, test_data = split_dataset(text_pairs)

    # Load tokenizers
    en_tokenizer, fr_tokenizer = get_tokenizers(train_data)

    # Create DataLoader
    train_loader = get_dataloader(train_data, en_tokenizer, fr_tokenizer)
    
    # Create Validation DataLoader for Validation
    val_loader = get_dataloader(val_data, en_tokenizer, fr_tokenizer)

    # Create Padding TokenID
    pad_token = fr_tokenizer.token_to_id("[pad]")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model dimensions
    input_dim = len(en_tokenizer.get_vocab())
    output_dim = len(fr_tokenizer.get_vocab())
    emb_dim = 256
    hidden_dim = 512

    # Build LSTM model
    encoder = Encoder(input_dim, emb_dim, hidden_dim)
    decoder = Decoder(output_dim, emb_dim, hidden_dim)

    model = Seq2Seq(encoder, decoder, device).to(device)

    train_model(
    model,
    train_loader,
    val_loader,
    pad_token,
    device,
    epochs=5
)

    print("LSTM Model created successfully.")
    print("Total parameters:",
          sum(p.numel() for p in model.parameters() if p.requires_grad))