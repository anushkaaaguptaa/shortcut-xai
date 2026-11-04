import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dataset import ShortcutDataset
from src.train import train_model, get_device, set_seed, checkpoint_name
from src.model import auto_arch
from src.evaluate import evaluate_behavior, load_trained_model
from src.explain import ExplanationEngine
from src.metrics import compute_am_above_chance, random_box_null, compute_pointing_game_hit
from src.visualize import plot_attribution_comparison, plot_sensitivity_curve, plot_faithfulness_scatter

import torch

METHODS = ['gradcam', 'integrated_gradients', 'occlusion']


def attribute_on_conflict(ckpt_path, device, marker_size, target_size, seed,
                          sample_limit, fig_dir, rho):
    """
    Run the three attribution methods on the CONFLICT set (marker stamped on normal images),
    explaining the model's PREDICTED class. Returns per-method aggregates.
    """
    model, _ = load_trained_model(ckpt_path, device)
    explainer = ExplanationEngine(model, device=device)
    ds = ShortcutDataset(split='test', mode='test_discordant', marker_size=marker_size,
                         target_size=target_size, seed=seed, data_dir='./data')

    eval_idx = [i for i, f in enumerate(ds.shortcut_flags) if f][:sample_limit]
    rng = np.random.RandomState(seed)
    agg = {m: {'amstar': [], 'am': [], 'hit': [], 'nullbeat': []} for m in METHODS}
    n_fig = 0

    occ_win = marker_size
    occ_stride = max(1, marker_size // 2)

    for i, idx in enumerate(tqdm(eval_idx, desc=f"  attribute rho={rho:.2f} seed={seed}", leave=False)):
        sample = ds[idx]
        img, bbox = sample['image'], sample['bbox'].numpy()
        with torch.no_grad():
            pred = int(model(img.unsqueeze(0).to(device)).argmax(dim=1).item())

        heatmaps = {}
        for m in METHODS:
            hm = explainer.attribute(img, target_class=pred, method=m,
                                     n_steps=50, window_size=occ_win, stride=occ_stride)
            heatmaps[m] = hm
            am, _chance, amstar, _ratio = compute_am_above_chance(hm, bbox)
            _null_mean, null_p95 = random_box_null(hm, (marker_size, marker_size),
                                                   n_samples=500, rng=rng, pct=95.0)
            agg[m]['am'].append(am)
            agg[m]['amstar'].append(amstar)
            agg[m]['hit'].append(compute_pointing_game_hit(hm, bbox))
            agg[m]['nullbeat'].append(1.0 if am > null_p95 else 0.0)

        if n_fig < 2:
            plot_attribution_comparison(
                img, bbox, heatmaps,
                title=f"Conflict set (rho={rho:.2f}, seed={seed}, pred={pred})",
                save_path=os.path.join(fig_dir, f"sample_rho{rho:.2f}_seed{seed}_{idx}.png"))
            n_fig += 1

    out = {}
    for m in METHODS:
        out[f'am_{m}'] = float(np.mean(agg[m]['am'])) if agg[m]['am'] else 0.0
        out[f'amstar_{m}'] = float(np.mean(agg[m]['amstar'])) if agg[m]['amstar'] else 0.0
        out[f'hit_{m}'] = float(np.mean(agg[m]['hit'])) if agg[m]['hit'] else 0.0
        out[f'nullbeat_{m}'] = float(np.mean(agg[m]['nullbeat'])) if agg[m]['nullbeat'] else 0.0
    return out


def run_benchmark(rhos, seeds, epochs, arch, marker_size, target_size,
                  sample_limit, afr_threshold, output_dir, data_dir, val_clean=False):
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    device = get_device()
    if arch is None:
        arch = auto_arch(device)

    print("=" * 70)
    print(f"Dose-response shortcut-XAI benchmark | device={device} arch={arch}")
    print(f"rhos={rhos}  seeds={seeds}  epochs={epochs}  marker={marker_size}px@{target_size}")
    print("=" * 70)

    rows = []
    for rho in rhos:
        for seed in seeds:
            set_seed(seed)
            ckpt = os.path.join("./checkpoints", checkpoint_name(rho, seed, arch))
            if not os.path.exists(ckpt):
                train_model(bias_prob=rho, seed=seed, arch=arch, epochs=epochs,
                            marker_size=marker_size, target_size=target_size,
                            val_clean=val_clean, output_dir="./checkpoints", data_dir=data_dir)
            else:
                print(f"[+] Found checkpoint {ckpt}")

            behavior = evaluate_behavior(ckpt, data_dir=data_dir)
            attr = attribute_on_conflict(ckpt, device, marker_size, target_size, seed,
                                         sample_limit, fig_dir, rho)

            row = {'bias_prob': rho, 'seed': seed,
                   'acc_clean': behavior['acc_clean'],
                   'acc_concordant': behavior['acc_concordant'],
                   'drop_rate': behavior['drop_rate'],
                   'adversarial_flip_rate': behavior['adversarial_flip_rate'],
                   **attr}
            # Detection rule: localization beats the random-box null for a majority of images
            # AND the model is behaviorally reliant on the marker.
            reliant = behavior['adversarial_flip_rate'] > afr_threshold
            for m in METHODS:
                row[f'detected_{m}'] = bool(row[f'nullbeat_{m}'] > 0.5 and reliant)
            rows.append(row)

            print(f"    behaviorally reliant: {reliant} (AFR {behavior['adversarial_flip_rate']*100:.1f}%)")
            for m in METHODS:
                print(f"    {m:<20} AM* {row[f'amstar_{m}']*100:+6.2f}%  "
                      f"hit {row[f'hit_{m}']*100:5.1f}%  null-beat {row[f'nullbeat_{m}']*100:5.1f}%  "
                      f"=> detected={row[f'detected_{m}']}")

    per_model = pd.DataFrame(rows)
    per_model_path = os.path.join(output_dir, "per_model_metrics.csv")
    per_model.to_csv(per_model_path, index=False)
    print(f"\n[+] Saved per-model metrics: {per_model_path}")

    # Aggregate across seeds for each rho.
    metric_cols = (['acc_clean', 'drop_rate', 'adversarial_flip_rate']
                   + [f'{p}_{m}' for p in ['am', 'amstar', 'hit', 'nullbeat'] for m in METHODS])
    agg = per_model.groupby('bias_prob')[metric_cols].agg(['mean', 'std'])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg = agg.reset_index()
    agg_path = os.path.join(output_dir, "summary_by_rho.csv")
    agg.to_csv(agg_path, index=False)
    print(f"[+] Saved rho-aggregated summary: {agg_path}")

    # Figures.
    plot_sensitivity_curve(agg, metric_prefix='amstar',
                           ylabel="Attribution mass above chance (%)",
                           title="Shortcut localization vs correlation strength",
                           save_path=os.path.join(fig_dir, "amstar_vs_rho.png"))
    plot_sensitivity_curve(agg, metric_prefix='hit',
                           ylabel="Pointing-game hit rate (%)",
                           title="Peak-in-marker rate vs correlation strength",
                           save_path=os.path.join(fig_dir, "hit_vs_rho.png"))
    plot_faithfulness_scatter(per_model, save_path=os.path.join(fig_dir, "afr_vs_amstar.png"))

    with open(os.path.join(output_dir, "run_config.json"), 'w') as f:
        json.dump({'rhos': rhos, 'seeds': seeds, 'epochs': epochs, 'arch': arch,
                   'marker_size': marker_size, 'target_size': target_size,
                   'sample_limit': sample_limit, 'afr_threshold': afr_threshold}, f, indent=2)

    print("\n[✓] Benchmark complete. See per_model_metrics.csv and summary_by_rho.csv.")
    return per_model, agg


def _parse_floats(s):
    return [float(x) for x in s.split(',')]


def _parse_ints(s):
    return [int(x) for x in s.split(',')]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rhos", type=_parse_floats, default=[0.0, 0.7, 0.9, 1.0])
    p.add_argument("--seeds", type=_parse_ints, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--arch", type=str, default=None, choices=[None, 'resnet18', 'smallcnn'])
    p.add_argument("--marker_size", type=int, default=4)
    p.add_argument("--target_size", type=int, default=28)
    p.add_argument("--samples", type=int, default=100, help="attribution images per model")
    p.add_argument("--afr_threshold", type=float, default=0.3)
    p.add_argument("--val_clean", action="store_true",
                   help="Select checkpoints on a CLEAN val set (default: biased, matching training)")
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--quick_test", action="store_true", help="2 rhos x 1 seed, few images")
    args = p.parse_args()

    if args.quick_test:
        args.rhos, args.seeds, args.samples = [0.0, 1.0], [0], 10

    run_benchmark(args.rhos, args.seeds, args.epochs, args.arch, args.marker_size,
                  args.target_size, args.samples, args.afr_threshold,
                  args.output_dir, args.data_dir, val_clean=args.val_clean)
