import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns


def denormalize_image(tensor):
    """Converts normalized PyTorch image tensor [3, H, W] to displayable numpy [H, W, 3]."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = tensor.cpu().numpy() * std + mean
    img = np.clip(img.transpose(1, 2, 0), 0, 1)
    return img


def plot_attribution_comparison(img_tensor, bbox, heatmaps_dict, title="Attribution Comparison", save_path="comparison.png"):
    """
    Plots a multi-panel visual row:
    Input Image | GT Shortcut Bounding Box | Grad-CAM | Integrated Gradients | Occlusion
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    img_disp = denormalize_image(img_tensor)
    methods = list(heatmaps_dict.keys())
    n_panels = 2 + len(methods)

    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 3.5))

    # Panel 1: Original Image
    axes[0].imshow(img_disp)
    axes[0].set_title("Input Image", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Ground Truth Bbox Overlay
    axes[1].imshow(img_disp)
    x1, y1, x2, y2 = bbox
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
    axes[1].add_patch(rect)
    axes[1].set_title("GT Shortcut Box", fontsize=11, fontweight='bold', color='darkred')
    axes[1].axis('off')

    # Heatmap panels
    for i, method in enumerate(methods):
        ax = axes[2 + i]
        hm = heatmaps_dict[method]
        ax.imshow(img_disp)
        im = ax.imshow(hm, cmap='jet', alpha=0.55)
        # Add bounding box outline for visual reference
        rect_ref = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor='white', linestyle='--', facecolor='none')
        ax.add_patch(rect_ref)
        ax.set_title(method.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle(title, fontsize=13, fontweight='heavy', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved comparison plot: {save_path}")


def plot_sensitivity_curve(results_df, save_path="sensitivity_curve.png"):
    """
    Plots Shortcut Bias Strength (x-axis) vs. Attribution Mass (y-axis) across XAI methods.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid")

    for col in results_df.columns:
        if col.startswith("am_"):
            method = col.replace("am_", "").replace("_", " ").title()
            plt.plot(results_df['bias_prob'], results_df[col] * 100, marker='o', linewidth=2.5, label=method)

    plt.xlabel(r"Shortcut Bias Strength ($\rho$)", fontsize=12, fontweight='bold')
    plt.ylabel("Attribution Mass in Marker Box (%)", fontsize=12, fontweight='bold')
    plt.title("Can XAI Detect Shortcut Learning as Bias Increases?", fontsize=13, fontweight='heavy')
    plt.ylim(0, 100)
    plt.legend(frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved sensitivity curve: {save_path}")
