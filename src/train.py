import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import ShortcutDataset
from src.model import build_model, auto_arch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    """Seed every RNG and request deterministic cuDNN for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def checkpoint_name(bias_prob, seed, arch):
    return f"model_{arch}_rho_{bias_prob:.2f}_seed_{seed}.pt"


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
        correct += (outputs.argmax(dim=1) == labels).sum().item()
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
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_model(
    bias_prob=0.0,
    seed=0,
    arch=None,
    epochs=10,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-2,
    marker_size=4,
    target_size=28,
    upsample_size=64,
    pretrained=True,
    val_clean=False,
    output_dir="./checkpoints",
    data_dir="./data",
):
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    if arch is None:
        arch = auto_arch(device)
    set_seed(seed)

    print(f"[*] Training rho={bias_prob:.2f} seed={seed} arch={arch} on {device}")

    train_dataset = ShortcutDataset(
        split='train', bias_prob=bias_prob, mode='train_biased',
        marker_size=marker_size, target_size=target_size, seed=seed, data_dir=data_dir,
    )
    # By default, validate on the SAME biased distribution used for training, so checkpoint
    # selection reflects a realistic in-distribution pipeline rather than secretly rewarding
    # the least shortcut-reliant model (which clean-val selection would do). --val_clean reverts.
    if val_clean:
        val_dataset = ShortcutDataset(
            split='val', bias_prob=0.0, mode='test_clean',
            marker_size=marker_size, target_size=target_size, seed=seed, data_dir=data_dir,
        )
    else:
        val_dataset = ShortcutDataset(
            split='val', bias_prob=bias_prob, mode='train_biased',
            marker_size=marker_size, target_size=target_size, seed=seed, data_dir=data_dir,
        )

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(
        num_classes=2, arch=arch, pretrained=pretrained,
        upsample_size=upsample_size, device=device,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_acc = -1.0
    ckpt_path = os.path.join(output_dir, checkpoint_name(bias_prob, seed, arch))
    config = {
        'bias_prob': bias_prob, 'seed': seed, 'arch': arch, 'epochs': epochs,
        'batch_size': batch_size, 'lr': lr, 'weight_decay': weight_decay,
        'marker_size': marker_size, 'target_size': target_size,
        'upsample_size': upsample_size, 'pretrained': pretrained,
        'val_clean': val_clean,
    }

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_epoch(model, val_loader, criterion, device)
        print(f"  Epoch {epoch:02d}/{epochs:02d} | train acc {tr_acc*100:5.2f}% | val acc {va_acc*100:5.2f}%")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({**config, 'epoch': epoch, 'val_acc': va_acc,
                        'model_state_dict': model.state_dict()}, ckpt_path)

    with open(ckpt_path.replace('.pt', '.config.json'), 'w') as f:
        json.dump({**config, 'best_val_acc': best_val_acc}, f, indent=2)
    print(f"[+] Saved best checkpoint (val acc {best_val_acc*100:.2f}%): {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arch", type=str, default=None, choices=[None, 'resnet18', 'smallcnn'])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--marker_size", type=int, default=4)
    parser.add_argument("--target_size", type=int, default=28)
    parser.add_argument("--val_clean", action="store_true",
                        help="Select checkpoint on a CLEAN val set (default: biased, matching training)")
    args = parser.parse_args()
    train_model(
        bias_prob=args.bias, seed=args.seed, arch=args.arch, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, marker_size=args.marker_size,
        target_size=args.target_size, val_clean=args.val_clean,
    )
