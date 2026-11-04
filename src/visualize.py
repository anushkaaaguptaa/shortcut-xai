import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

_METHODS = ['gradcam', 'integrated_gradients', 'occlusion']
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)


def _ensure_dir(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def denormalize_image(tensor):
    img = tensor.cpu().numpy() * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(img.transpose(1, 2, 0), 0, 1)


def plot_attribution_comparison(img_tensor, bbox, heatmaps_dict, title, save_path):
    """Row: Input | GT box | one panel per attribution method, with the marker box overlaid."""
    _ensure_dir(save_path)
    img_disp = denormalize_image(img_tensor)
    methods = list(heatmaps_dict.keys())
    fig, axes = plt.subplots(1, 2 + len(methods), figsize=(3.3 * (2 + len(methods)), 3.3))
    x1, y1, x2, y2 = [int(v) for v in bbox]

    axes[0].imshow(img_disp); axes[0].set_title("Input", fontweight='bold'); axes[0].axis('off')
    axes[1].imshow(img_disp)
    axes[1].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none'))
    axes[1].set_title("GT marker box", color='darkred', fontweight='bold'); axes[1].axis('off')

    for i, m in enumerate(methods):
        ax = axes[2 + i]
        ax.imshow(img_disp)
        ax.imshow(heatmaps_dict[m], cmap='jet', alpha=0.55)
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5,
                                        edgecolor='white', linestyle='--', facecolor='none'))
        ax.set_title(m.replace('_', ' ').title(), fontweight='bold'); ax.axis('off')

    plt.suptitle(title, fontweight='heavy', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved comparison plot: {save_path}")


def plot_sensitivity_curve(agg_df, metric_prefix='amstar', ylabel=None, title=None, save_path='curve.png'):
    """
    Mean +/- std across seeds vs rho, one line per method.
    agg_df must have columns: bias_prob, {prefix}_{method}_mean, {prefix}_{method}_std.
    """
    _ensure_dir(save_path)
    plt.figure(figsize=(8, 5))
    for m in _METHODS:
        mean_col, std_col = f"{metric_prefix}_{m}_mean", f"{metric_prefix}_{m}_std"
        if mean_col not in agg_df.columns:
            continue
        x = agg_df['bias_prob'].values
        y = agg_df[mean_col].values * 100
        e = np.nan_to_num(agg_df[std_col].values) * 100
        plt.plot(x, y, marker='o', linewidth=2.5, label=m.replace('_', ' ').title())
        plt.fill_between(x, y - e, y + e, alpha=0.18)
    plt.axhline(0, color='gray', linewidth=1, linestyle=':')
    plt.xlabel(r"Shortcut correlation strength $\rho$", fontweight='bold')
    plt.ylabel(ylabel or "Attribution mass above chance (%)", fontweight='bold')
    plt.title(title or "Shortcut localization vs correlation strength", fontweight='heavy')
    plt.legend(frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved sensitivity curve: {save_path}")


def plot_faithfulness_scatter(per_model_df, save_path='faithfulness.png'):
    """Across models: behavioral reliance (AFR, x) vs localization (AM*, y), one series per method."""
    _ensure_dir(save_path)
    plt.figure(figsize=(7, 5.5))
    for m in _METHODS:
        col = f"amstar_{m}"
        if col not in per_model_df.columns:
            continue
        plt.scatter(per_model_df['adversarial_flip_rate'] * 100, per_model_df[col] * 100,
                    s=60, label=m.replace('_', ' ').title())
    plt.xlabel("Adversarial flip rate (%)  —  behavioral reliance", fontweight='bold')
    plt.ylabel("Attribution mass above chance (%)", fontweight='bold')
    plt.title("Does localization track true behavioral reliance?", fontweight='heavy')
    plt.axhline(0, color='gray', linewidth=1, linestyle=':')
    plt.legend(frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved faithfulness scatter: {save_path}")
