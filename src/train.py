import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ShortcutDataset
from src.model import build_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_model(
    bias_prob=0.0,
    epochs=5,
    batch_size=64,
    lr=1e-4,
    output_dir="./checkpoints",
    data_dir="./data"
):
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    print(f"[*] Training model with Shortcut Bias rho={bias_prob} on {device}")

    # Datasets
    train_dataset = ShortcutDataset(split='train', bias_prob=bias_prob, mode='train_biased', data_dir=data_dir)
    val_dataset = ShortcutDataset(split='val', bias_prob=0.0, mode='test_clean', data_dir=data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(num_classes=2, pretrained=True, upsample_size=64, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    best_val_acc = 0.0
    checkpoint_path = os.path.join(output_dir, f"model_bias_{bias_prob:.2f}.pt")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'bias_prob': bias_prob,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc
            }, checkpoint_path)

    print(f"[+] Saved best checkpoint to: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bias", type=float, default=0.0, help="Shortcut bias probability (0.0 to 1.0)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train_model(bias_prob=args.bias, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
