import torch
from models.lstm import Encoder, Decoder, Seq2Seq
from utils.tokenizer import get_tokenizers
from utils.dataset import load_text_pairs, split_dataset


# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -----------------------------
# Load dataset (only to get tokenizers)
# -----------------------------
text_pairs = load_text_pairs()
train_data, val_data, test_data = split_dataset(text_pairs)

en_tokenizer, fr_tokenizer = get_tokenizers(train_data)


# -----------------------------
# Recreate Model Architecture
# (MUST match training exactly)
# -----------------------------
input_dim = len(en_tokenizer.get_vocab())
output_dim = len(fr_tokenizer.get_vocab())

emb_dim = 256
hidden_dim = 512

encoder = Encoder(input_dim, emb_dim, hidden_dim)
decoder = Decoder(output_dim, emb_dim, hidden_dim)

model = Seq2Seq(encoder, decoder, device).to(device)


# -----------------------------
# Load Trained Weights
# -----------------------------
model.load_state_dict(
    torch.load("experiments/lstm_best.pth", map_location=device)
)

model.eval()


# -----------------------------
# Translation Function
# -----------------------------
def translate_sentence(sentence, max_len=50):

    model.eval()

    # Encode input sentence
    encoded = en_tokenizer.encode(sentence)
    tokens = encoded.ids

    # Shape: (batch_size=1, seq_len)
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    # Start token
    trg_indexes = [fr_tokenizer.token_to_id("[start]")]

    for _ in range(max_len):

        # IMPORTANT: shape should be (batch_size) NOT (1,1)
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == fr_tokenizer.token_to_id("[end]"):
            break

    return fr_tokenizer.decode(trg_indexes)


# -----------------------------
# Demo Run
# -----------------------------
if __name__ == "__main__":

    sentence = "I love you"
    translation = translate_sentence(sentence)

    print("\nInput :", sentence)
    print("Output:", translation)