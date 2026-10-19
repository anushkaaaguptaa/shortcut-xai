import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_attribution_mass(heatmap, bbox):
    """
    Computes the fraction of total attribution mass that falls inside the bounding box.
    bbox is (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = bbox
    total_mass = float(np.sum(np.abs(heatmap)))
    if total_mass < 1e-10:
        return 0.0

    h, w = heatmap.shape
    x1, x2 = max(0, int(x1)), min(w - 1, int(x2))
    y1, y2 = max(0, int(y1)), min(h - 1, int(y2))

    box_mass = float(np.sum(np.abs(heatmap[y1:y2 + 1, x1:x2 + 1])))
    return box_mass / total_mass


def compute_pointing_game_hit(heatmap, bbox):
    """
    Checks if the maximum attribution peak (argmax) falls inside the shortcut bounding box.
    """
    x1, y1, x2, y2 = bbox
    max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    peak_y, peak_x = max_idx

    is_hit = (x1 <= peak_x <= x2) and (y1 <= peak_y <= y2)
    return 1.0 if is_hit else 0.0


def compute_faithfulness_correlation(attribution_mass_list, logit_drop_list):
    """
    Calculates Pearson and Spearman rank correlation between the explanation's
    attribution mass on the shortcut and the actual behavioral drop in logit/confidence.
    """
    arr_am = np.array(attribution_mass_list)
    arr_drop = np.array(logit_drop_list)

    if len(arr_am) < 3 or np.all(arr_am == arr_am[0]) or np.all(arr_drop == arr_drop[0]):
        return {'pearson_r': 0.0, 'pearson_p': 1.0, 'spearman_rho': 0.0, 'spearman_p': 1.0}

    p_r, p_pval = pearsonr(arr_am, arr_drop)
    s_rho, s_pval = spearmanr(arr_am, arr_drop)

    return {
        'pearson_r': float(p_r),
        'pearson_p': float(p_pval),
        'spearman_rho': float(s_rho),
        'spearman_p': float(s_pval)
    }
