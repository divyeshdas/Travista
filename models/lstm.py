import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, src):
        # src shape: (batch_size, seq_len)
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_token, hidden, cell):
        # input_token shape: (batch_size)
        input_token = input_token.unsqueeze(1)  # (batch_size, 1)
        embedded = self.embedding(input_token)

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        output_dim = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, output_dim).to(self.device)

        hidden, cell = self.encoder(src)

        input_token = trg[:, 0]  # start token

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t] = output

            top1 = output.argmax(1)
            input_token = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs