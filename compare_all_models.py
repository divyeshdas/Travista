import torch
import torch.nn.functional as F

from models.lstm import Encoder as Encoder1, Decoder as Decoder1, Seq2Seq as Seq2Seq1
from models.lstm_attention import Encoder as Encoder2, Decoder as Decoder2, Seq2Seq as Seq2Seq2, Attention
from models.transformer_model import TransformerModel

from utils.tokenizer import get_tokenizers
from utils.dataset import load_text_pairs, split_dataset


# ==========================
# Device
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ==========================
# Load Tokenizers
# ==========================
text_pairs = load_text_pairs()
train_data, _, _ = split_dataset(text_pairs)
en_tokenizer, fr_tokenizer = get_tokenizers(train_data)

input_dim = len(en_tokenizer.get_vocab())
output_dim = len(fr_tokenizer.get_vocab())


# ======================================================
# 1️⃣ Load Baseline LSTM
# ======================================================
encoder1 = Encoder1(input_dim, 256, 512)
decoder1 = Decoder1(output_dim, 256, 512)
baseline_model = Seq2Seq1(encoder1, decoder1, device).to(device)

baseline_model.load_state_dict(
    torch.load("experiments/lstm_best.pth", map_location=device)
)
baseline_model.eval()


# ======================================================
# 2️⃣ Load Attention Model
# ======================================================
attention = Attention(512)
encoder2 = Encoder2(input_dim, 256, 512)
decoder2 = Decoder2(output_dim, 256, 512, attention)
attention_model = Seq2Seq2(encoder2, decoder2, device).to(device)

attention_model.load_state_dict(
    torch.load("experiments/lstm_attention_best.pth", map_location=device)
)
attention_model.eval()


# ======================================================
# 3️⃣ Load Transformer
# ======================================================
transformer_model = TransformerModel(
    src_vocab_size=input_dim,
    tgt_vocab_size=output_dim,
    d_model=256,
    nhead=8,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=512,
    dropout=0.1
).to(device)

transformer_model.load_state_dict(
    torch.load("experiments/transformer_best-2.pth", map_location=device)
)
transformer_model.eval()


# ======================================================
# Translation Functions
# ======================================================
def translate_baseline(model, sentence, max_len=50):
    tokens = en_tokenizer.encode(sentence).ids
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_indexes = [fr_tokenizer.token_to_id("[start]")]

    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == fr_tokenizer.token_to_id("[end]"):
            break

    return fr_tokenizer.decode(trg_indexes)


def translate_attention(model, sentence, max_len=50):
    tokens = en_tokenizer.encode(sentence).ids
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    trg_indexes = [fr_tokenizer.token_to_id("[start]")]

    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(
                trg_tensor, hidden, cell, encoder_outputs
            )

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == fr_tokenizer.token_to_id("[end]"):
            break

    return fr_tokenizer.decode(trg_indexes)


def translate_transformer(model, sentence, max_len=50):
    model.eval()

    # Encode source
    tokens = en_tokenizer.encode(sentence).ids
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)

    trg_indexes = [fr_tokenizer.token_to_id("[start]")]

    for _ in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        # Generate proper target mask
        tgt_mask = torch.triu(
            torch.ones((trg_tensor.size(1), trg_tensor.size(1)), device=device) * float("-inf"),
            diagonal=1
        )

        with torch.no_grad():
            output = model(src_tensor, trg_tensor)

        # Get last token prediction
        pred_token = output[:, -1, :].argmax(dim=-1).item()

        # Stop if EOS
        if pred_token == fr_tokenizer.token_to_id("[end]"):
            break

        # Repetition control (simple)
        if len(trg_indexes) > 3 and pred_token == trg_indexes[-1]:
            break

        trg_indexes.append(pred_token)

    return fr_tokenizer.decode(trg_indexes)

# ======================================================
# Demo Comparison
# ======================================================
if __name__ == "__main__":

    test_sentences = [
        "I love you",
        "I am tired",
        "He is a student",
        "She is happy",
        "This is my book"
    ]

    for sentence in test_sentences:
        print("\nInput:", sentence)

        baseline_output = translate_baseline(baseline_model, sentence)
        attention_output = translate_attention(attention_model, sentence)
        transformer_output = translate_transformer(transformer_model, sentence)

        print("Baseline Output    :", baseline_output)
        print("Attention Output   :", attention_output)
        print("Transformer Output :", transformer_output)
        print("-" * 60)