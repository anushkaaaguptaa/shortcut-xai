import numpy as np
from scipy.stats import pearsonr, spearmanr


def _clip_box(bbox, h, w):
    x1, y1, x2, y2 = bbox
    return max(0, int(x1)), max(0, int(y1)), min(w - 1, int(x2)), min(h - 1, int(y2))


def compute_attribution_mass(heatmap, bbox):
    """Fraction of total |attribution| that falls inside the bbox (x1, y1, x2, y2), inclusive."""
    total_mass = float(np.sum(np.abs(heatmap)))
    if total_mass < 1e-10:
        return 0.0
    h, w = heatmap.shape
    x1, y1, x2, y2 = _clip_box(bbox, h, w)
    box_mass = float(np.sum(np.abs(heatmap[y1:y2 + 1, x1:x2 + 1])))
    return box_mass / total_mass


def box_area_fraction(bbox, h, w):
    """Chance-level attribution mass for a box of this size = |box| / |image|."""
    x1, y1, x2, y2 = _clip_box(bbox, h, w)
    return ((x2 - x1 + 1) * (y2 - y1 + 1)) / float(h * w)


def compute_am_above_chance(heatmap, bbox):
    """
    Returns (AM, chance, AM_star, ratio) where AM_star = AM - chance and ratio = AM / chance.
    AM_star > 0 (ratio > 1) means the box concentrates more mass than an equal-area box would
    by chance. This controls the area-bias confound in raw AM.
    """
    h, w = heatmap.shape
    am = compute_attribution_mass(heatmap, bbox)
    chance = box_area_fraction(bbox, h, w)
    ratio = am / chance if chance > 1e-12 else 0.0
    return am, chance, am - chance, ratio


def random_box_null(heatmap, box_hw, n_samples=1000, rng=None, pct=95.0):
    """
    Empirical null for AM: slide a box of the SAME size to n random positions on this heatmap
    and return (mean_AM, percentile_AM). The observed AM beating the top percentile is evidence
    the attribution genuinely concentrates on the marker rather than on a random equal-area patch.
    """
    if rng is None:
        rng = np.random.RandomState(0)
    h, w = heatmap.shape
    bh, bw = box_hw
    total_mass = float(np.sum(np.abs(heatmap)))
    if total_mass < 1e-10 or bh > h or bw > w:
        return 0.0, 0.0
    samples = np.empty(n_samples, dtype=np.float64)
    for k in range(n_samples):
        y1 = int(rng.randint(0, h - bh + 1))
        x1 = int(rng.randint(0, w - bw + 1))
        samples[k] = float(np.sum(np.abs(heatmap[y1:y1 + bh, x1:x1 + bw]))) / total_mass
    return float(samples.mean()), float(np.percentile(samples, pct))


def compute_pointing_game_hit(heatmap, bbox):
    """1.0 if argmax(heatmap) is inside the bbox, else 0.0."""
    x1, y1, x2, y2 = bbox
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return 1.0 if (x1 <= peak_x <= x2 and y1 <= peak_y <= y2) else 0.0


def compute_faithfulness_correlation(x_list, y_list):
    """
    Pearson and Spearman correlation between two paired sequences (e.g. localization vs
    behavioral reliance across models). Returns zeros/ones if degenerate or n < 3.
    """
    x, y = np.asarray(x_list, dtype=float), np.asarray(y_list, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {'pearson_r': 0.0, 'pearson_p': 1.0, 'spearman_rho': 0.0, 'spearman_p': 1.0, 'n': int(len(x))}
    p_r, p_p = pearsonr(x, y)
    s_r, s_p = spearmanr(x, y)
    return {'pearson_r': float(p_r), 'pearson_p': float(p_p),
            'spearman_rho': float(s_r), 'spearman_p': float(s_p), 'n': int(len(x))}
