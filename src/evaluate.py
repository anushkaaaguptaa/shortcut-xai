import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

from src.dataset import ShortcutDataset
from src.model import build_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_trained_model(checkpoint_path, device):
    model = build_model(num_classes=2, pretrained=False, upsample_size=64, device=device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt.get('bias_prob', 0.0)


def evaluate_behavior(checkpoint_path, data_dir="./data", batch_size=64):
    device = get_device()
    model, bias_prob = load_trained_model(checkpoint_path, device)
    
    # Test Splits
    test_clean = ShortcutDataset(split='test', mode='test_clean', data_dir=data_dir)
    test_concordant = ShortcutDataset(split='test', mode='test_concordant', data_dir=data_dir)
    test_discordant = ShortcutDataset(split='test', mode='test_discordant', data_dir=data_dir)

    loaders = {
        'clean': DataLoader(test_clean, batch_size=batch_size, shuffle=False),
        'concordant': DataLoader(test_concordant, batch_size=batch_size, shuffle=False),
        'discordant': DataLoader(test_discordant, batch_size=batch_size, shuffle=False)
    }

    results = {'bias_prob': bias_prob}

    for name, loader in loaders.items():
        total, correct = 0, 0
        flipped_class0 = 0
        total_class0 = 0

        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                logits = model(images)
                preds = logits.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                # Specifically monitor Class 0 (healthy) in discordant/swap mode
                if name == 'discordant':
                    c0_mask = (labels == 0)
                    total_class0 += c0_mask.sum().item()
                    # How many healthy were tricked into predicting 1 (Pneumonia)?
                    flipped_class0 += ((preds == 1) & c0_mask).sum().item()

        acc = correct / total if total > 0 else 0.0
        results[f'acc_{name}'] = acc
        if name == 'discordant' and total_class0 > 0:
            results['adversarial_flip_rate'] = flipped_class0 / total_class0

    results['drop_rate'] = results['acc_concordant'] - results['acc_clean']

    print(f"\n[+] Behavioral Results for Model rho={bias_prob:.2f}:")
    print(f"    - Clean Test Accuracy:       {results['acc_clean']*100:.2f}%")
    print(f"    - Concordant Test Accuracy:  {results['acc_concordant']*100:.2f}%")
    print(f"    - Performance Drop (Clean):  {results['drop_rate']*100:.2f}%")
    print(f"    - Adversarial Flip Rate:     {results.get('adversarial_flip_rate', 0.0)*100:.2f}%\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    evaluate_behavior(args.checkpoint)
