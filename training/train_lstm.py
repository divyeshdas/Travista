import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_model(model, train_loader, val_loader, pad_token, device, epochs=10):

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float("inf")

    for epoch in range(epochs):

        # -------------------
        # TRAINING PHASE
        # -------------------
        model.train()
        train_loss = 0

        for src, trg in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):

            src = src.to(device)
            trg = trg.to(device)

            optimizer.zero_grad()

            output = model(src, trg)

            # Shift for loss calculation
            output_dim = output.shape[-1]

            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # -------------------
        # VALIDATION PHASE
        # -------------------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for src, trg in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):

                src = src.to(device)
                trg = trg.to(device)

                output = model(src, trg, teacher_forcing_ratio=0)

                output_dim = output.shape[-1]

                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)

                loss = criterion(output, trg)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "experiments/lstm_attention_best.pth")
            print("Model saved.")