import torch
import torch.nn as nn


# =====================================
# Encoder
# =====================================
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        # src: (batch_size, src_len)

        embedded = self.embedding(src)
        # embedded: (batch_size, src_len, emb_dim)

        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs: (batch_size, src_len, hidden_dim)

        return outputs, hidden, cell


# =====================================
# Attention Mechanism
# =====================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (1, batch_size, hidden_dim)
        # encoder_outputs: (batch_size, src_len, hidden_dim)

        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # Use last layer hidden state
        hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        # hidden: (batch_size, src_len, hidden_dim)

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2))
        )
        # energy: (batch_size, src_len, hidden_dim)

        attention = self.v(energy).squeeze(2)
        # attention: (batch_size, src_len)

        return torch.softmax(attention, dim=1)


# =====================================
# Decoder with Attention
# =====================================
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.lstm = nn.LSTM(
            emb_dim + hidden_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        # input: (batch_size)
        # hidden: (1, batch_size, hidden_dim)
        # encoder_outputs: (batch_size, src_len, hidden_dim)

        input = input.unsqueeze(1)
        # (batch_size, 1)

        embedded = self.embedding(input)
        # (batch_size, 1, emb_dim)

        # Compute attention weights
        attn_weights = self.attention(hidden, encoder_outputs)
        # (batch_size, src_len)

        attn_weights = attn_weights.unsqueeze(1)
        # (batch_size, 1, src_len)

        # Compute context vector
        context = torch.bmm(attn_weights, encoder_outputs)
        # (batch_size, 1, hidden_dim)

        # Concatenate embedded + context
        lstm_input = torch.cat((embedded, context), dim=2)
        # (batch_size, 1, emb_dim + hidden_dim)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # output: (batch_size, 1, hidden_dim)

        output = output.squeeze(1)
        context = context.squeeze(1)
        # both: (batch_size, hidden_dim)

        prediction = self.fc_out(torch.cat((output, context), dim=1))
        # (batch_size, output_dim)

        return prediction, hidden, cell


# =====================================
# Seq2Seq Wrapper
# =====================================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(
            batch_size,
            trg_len,
            trg_vocab_size
        ).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        input = trg[:, 0]

        for t in range(1, trg_len):

            output, hidden, cell = self.decoder(
                input,
                hidden,
                cell,
                encoder_outputs
            )

            outputs[:, t] = output

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input = trg[:, t] if teacher_force else top1

        return outputs