import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_transformer(
    model,
    train_loader,
    val_loader,
    pad_token,
    device,
    epochs=5
):

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    best_val_loss = float("inf")

    for epoch in range(epochs):

        # ======================
        # TRAINING
        # ======================
        model.train()
        train_loss = 0

        for src, trg in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):

            src = src.to(device)
            trg = trg.to(device)

            optimizer.zero_grad()

            # Shift target for teacher forcing
            output = model(src, trg[:, :-1])

            output_dim = output.shape[-1]

            loss = criterion(
                output.reshape(-1, output_dim),
                trg[:, 1:].reshape(-1)
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ======================
        # VALIDATION
        # ======================
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for src, trg in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):

                src = src.to(device)
                trg = trg.to(device)

                output = model(src, trg[:, :-1])
                output_dim = output.shape[-1]

                loss = criterion(
                    output.reshape(-1, output_dim),
                    trg[:, 1:].reshape(-1)
                )

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "experiments/transformer_best.pth")
            print("Model saved.")