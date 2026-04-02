#!/usr/bin/env python3
"""
Standalone NN training script — runs in isolated subprocess to avoid
loky/OpenMP deadlocks that occur when PyTorch and joblib coexist.

Usage: python _nn_train.py <data_path.npz> <result_path.pt> <n_classes>
"""
import sys
import numpy as np


def main():
    data_path = sys.argv[1]
    result_path = sys.argv[2]
    n_classes = int(sys.argv[3])

    import torch
    import torch.nn as nn
    import torch.optim as optim

    torch.set_num_threads(4)
    torch.manual_seed(42)

    data = np.load(data_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    n_features = X_train.shape[1]
    print(
        f"Loaded: X_train={X_train.shape}, X_val={X_val.shape}, "
        f"n_classes={n_classes}",
        flush=True,
    )

    class StormNet(nn.Module):
        def __init__(self, n_in, n_out):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, n_out),
            )

        def forward(self, x):
            return self.net(x)

    model = StormNet(n_features, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, factor=0.5
    )

    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)

    batch_size = 512
    n_train = len(X_train)
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    max_epochs = 50

    print(
        f"Training: {n_train:,} samples, batch_size={batch_size}, "
        f"max_epochs={max_epochs}",
        flush=True,
    )

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        perm = np.random.permutation(n_train)

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb = torch.FloatTensor(X_train[idx])
            yb = torch.LongTensor(y_train[idx])
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)

        train_losses.append(epoch_loss / total)
        train_accs.append(correct / total)

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = criterion(val_out, y_val_t).item()
            val_acc = (val_out.argmax(1) == y_val_t).float().mean().item()
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        scheduler.step(val_loss)

        if epoch % 5 == 0 or epoch == max_epochs - 1:
            print(
                f"Epoch {epoch+1}/{max_epochs}: "
                f"train_loss={train_losses[-1]:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}",
                flush=True,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 8:
                print(f"Early stopping at epoch {epoch+1}", flush=True)
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_out = model(X_val_t)
        y_pred = val_out.argmax(1).numpy()

    from sklearn.metrics import f1_score

    val_f1 = f1_score(y_val, y_pred, average="macro")
    print(f"Final Validation Macro-F1: {val_f1:.4f}", flush=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_features": n_features,
            "n_classes": n_classes,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "val_f1": val_f1,
        },
        result_path,
    )
    print("Results saved.", flush=True)


if __name__ == "__main__":
    main()
