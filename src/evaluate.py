import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import ShortcutDataset
from src.model import build_model
from src.train import get_device


def _load_config(checkpoint_path, ckpt):
    """Prefer the sidecar config.json; fall back to fields stored in the checkpoint."""
    cfg_path = checkpoint_path.replace('.pt', '.config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return json.load(f)
    return {k: ckpt.get(k) for k in
            ['bias_prob', 'seed', 'arch', 'marker_size', 'target_size', 'upsample_size']}


def load_trained_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = _load_config(checkpoint_path, ckpt)
    arch = cfg.get('arch', 'resnet18')
    model = build_model(
        num_classes=2, arch=arch, pretrained=False,
        upsample_size=cfg.get('upsample_size', 64) or 64, device=device,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg


def _predict_by_index(model, dataset, device, batch_size):
    """Return dicts {index: pred} and {index: label} for a dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, labels = {}, {}
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['image'].to(device))
            p = logits.argmax(dim=1).cpu().numpy()
            idxs = batch['index'].numpy()
            lbls = batch['label'].numpy()
            for k, pr, lb in zip(idxs, p, lbls):
                preds[int(k)] = int(pr)
                labels[int(k)] = int(lb)
    return preds, labels


def evaluate_behavior(checkpoint_path, data_dir="./data", batch_size=128):
    """
    Behavioral shortcut-reliance metrics (METHODOLOGY.md section 5a):
      - acc_clean, acc_concordant, drop_rate = acc_concordant - acc_clean
      - adversarial_flip_rate (AFR): among normal images the model classifies correctly when
        CLEAN, the fraction that flip to 'pneumonia' once the marker is stamped on them.
    """
    device = get_device()
    model, cfg = load_trained_model(checkpoint_path, device)
    ms, ts, seed = cfg.get('marker_size', 4), cfg.get('target_size', 28), cfg.get('seed', 0)
    common = dict(split='test', marker_size=ms, target_size=ts, seed=seed, data_dir=data_dir)

    ds_clean = ShortcutDataset(mode='test_clean', **common)
    ds_conc = ShortcutDataset(mode='test_concordant', **common)
    ds_disc = ShortcutDataset(mode='test_discordant', **common)

    clean_pred, labels = _predict_by_index(model, ds_clean, device, batch_size)
    conc_pred, _ = _predict_by_index(model, ds_conc, device, batch_size)
    disc_pred, _ = _predict_by_index(model, ds_disc, device, batch_size)

    n = len(labels)
    acc_clean = np.mean([clean_pred[i] == labels[i] for i in range(n)])
    acc_conc = np.mean([conc_pred[i] == labels[i] for i in range(n)])

    # AFR: normal images (label 0) correctly classified when clean, that flip to 1 under marker.
    eligible = [i for i in range(n) if labels[i] == 0 and clean_pred[i] == 0]
    flipped = [i for i in eligible if disc_pred[i] == 1]
    afr = (len(flipped) / len(eligible)) if eligible else 0.0

    results = {
        'bias_prob': cfg.get('bias_prob', 0.0), 'seed': seed,
        'acc_clean': float(acc_clean), 'acc_concordant': float(acc_conc),
        'drop_rate': float(acc_conc - acc_clean),
        'adversarial_flip_rate': float(afr),
        'n_afr_eligible': len(eligible),
    }
    print(f"[+] rho={results['bias_prob']:.2f} seed={seed} | "
          f"clean {acc_clean*100:.2f}% | concordant {acc_conc*100:.2f}% | "
          f"drop {results['drop_rate']*100:+.2f}% | AFR {afr*100:.2f}% (n={len(eligible)})")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    evaluate_behavior(args.checkpoint)
