import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.dataset import ShortcutDataset
from src.model import build_model
from src.train import train_model, get_device
from src.evaluate import evaluate_behavior, load_trained_model
from src.explain import ExplanationEngine
from src.metrics import compute_attribution_mass, compute_pointing_game_hit, compute_faithfulness_correlation
from src.visualize import plot_attribution_comparison, plot_sensitivity_curve


def run_xai_benchmark(
    bias_levels=[0.0, 0.7, 0.9, 1.0],
    epochs=3,
    sample_eval_limit=50,
    output_dir="./results",
    data_dir="./data"
):
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    device = get_device()
    print(f"================================================================")
    print(f"🚀 Starting Shortcut Learning Explainability Benchmark on {device}")
    print(f"   Bias sweep: {bias_levels}")
    print(f"================================================================")

    summary_records = []

    for rho in bias_levels:
        ckpt_path = os.path.join("./checkpoints", f"model_bias_{rho:.2f}.pt")

        # Step 1: Train model if not already cached
        if not os.path.exists(ckpt_path):
            print(f"\n--- [1/4] Training Model with rho={rho:.2f} ---")
            train_model(bias_prob=rho, epochs=epochs, output_dir="./checkpoints", data_dir=data_dir)
        else:
            print(f"\n[+] Found existing checkpoint: {ckpt_path}")

        # Step 2: Behavioral evaluation
        print(f"--- [2/4] Evaluating Behavioral Shortcut Reliance for rho={rho:.2f} ---")
        behavior = evaluate_behavior(ckpt_path, data_dir=data_dir)

        # Step 3: Explainability attribution
        print(f"--- [3/4] Extracting XAI Attributions (Grad-CAM, IG, Occlusion) ---")
        model, _ = load_trained_model(ckpt_path, device)
        explainer = ExplanationEngine(model, device=device)

        # Evaluate on test set with concordant shortcuts to test localization of the marker
        test_dataset = ShortcutDataset(split='test', mode='test_concordant', data_dir=data_dir)
        
        # Test clean set to measure logit drops when marker is removed
        test_clean_dataset = ShortcutDataset(split='test', mode='test_clean', data_dir=data_dir)

        am_records = {'gradcam': [], 'integrated_gradients': [], 'occlusion': []}
        hit_records = {'gradcam': [], 'integrated_gradients': [], 'occlusion': []}
        logit_drops = []

        eval_indices = [i for i, flag in enumerate(test_dataset.shortcut_flags) if flag][:sample_eval_limit]

        for i, idx in enumerate(tqdm(eval_indices, desc=f"XAI Attribution rho={rho:.2f}")):
            sample = test_dataset[idx]
            img = sample['image']
            bbox = sample['bbox'].numpy()
            target_cls = int(sample['label'])

            # Behavioral shift: logit difference between image with marker vs clean
            sample_clean = test_clean_dataset[idx]
            with torch.no_grad():
                out_marked = model(img.unsqueeze(0).to(device))
                out_clean = model(sample_clean['image'].unsqueeze(0).to(device))
                logit_drop = float((out_marked[0, target_cls] - out_clean[0, target_cls]).item())
                logit_drops.append(logit_drop)

            heatmaps = {}
            for method in ['gradcam', 'integrated_gradients', 'occlusion']:
                hm = explainer.attribute(img, target_class=target_cls, method=method)
                heatmaps[method] = hm
                am = compute_attribution_mass(hm, bbox)
                hit = compute_pointing_game_hit(hm, bbox)
                am_records[method].append(am)
                hit_records[method].append(hit)

            # Save qualitative visual figure for the first 3 samples
            if i < 3:
                fig_path = os.path.join(fig_dir, f"sample_{idx}_rho_{rho:.2f}.png")
                plot_attribution_comparison(
                    img, bbox, heatmaps,
                    title=f"Attribution Map (rho={rho:.2f}, Label={target_cls})",
                    save_path=fig_path
                )

        # Step 4: Quantitative metric compilation
        print(f"--- [4/4] Compiling Statistics for rho={rho:.2f} ---")
        record = {
            'bias_prob': rho,
            'acc_clean': behavior['acc_clean'],
            'acc_concordant': behavior['acc_concordant'],
            'drop_rate': behavior['drop_rate'],
            'adversarial_flip_rate': behavior.get('adversarial_flip_rate', 0.0),
        }

        for method in ['gradcam', 'integrated_gradients', 'occlusion']:
            mean_am = float(np.mean(am_records[method]))
            mean_hit = float(np.mean(hit_records[method]))
            faith = compute_faithfulness_correlation(am_records[method], logit_drops)

            record[f'am_{method}'] = mean_am
            record[f'hit_{method}'] = mean_hit
            record[f'faith_r_{method}'] = faith['pearson_r']
            record[f'faith_rho_{method}'] = faith['spearman_rho']

            print(f"    • {method.replace('_', ' ').title():<20} "
                  f"AM: {mean_am*100:5.2f}% | "
                  f"Hit Rate: {mean_hit*100:5.2f}% | "
                  f"Faithfulness (r): {faith['pearson_r']:+.3f}")

        summary_records.append(record)

    # Save summary dataframe
    summary_df = pd.DataFrame(summary_records)
    csv_path = os.path.join(output_dir, "summary_metrics.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\n[+] Saved full summary table to: {csv_path}")

    # Plot sensitivity curve
    plot_sensitivity_curve(summary_df, save_path=os.path.join(fig_dir, "shortcut_sensitivity_curve.png"))

    print("\n✅ Benchmark experiment complete!")
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per model")
    parser.add_argument("--samples", type=int, default=30, help="Number of test samples to attribute")
    parser.add_argument("--quick_test", action="store_true", help="Run quick 2-level sweep (0.0 and 1.0)")
    args = parser.parse_args()

    bias_levels = [0.0, 1.0] if args.quick_test else [0.0, 0.7, 0.9, 1.0]
    run_xai_benchmark(bias_levels=bias_levels, epochs=args.epochs, sample_eval_limit=args.samples)
