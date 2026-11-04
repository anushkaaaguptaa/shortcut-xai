# Phase 5 — Results Analysis

Source: `results/per_model_metrics.csv` (12 models = 4 ρ × 3 seeds), ResNet-18 @ 64px, Colab GPU.
All aggregates below were **computed from the per-model rows**, not transcribed. n=3 seeds →
**descriptive trends only; no significance claimed.** These are the experiment's actual numbers.

## Behavioral reliance (mean ± SD)
| ρ | Clean acc | ΔAcc | Adversarial flip rate |
|---|---|---|---|
| 0.0 | 85.0 ± 1.9% | −0.3 ± 0.3% | 6.8 ± 6.1% |
| 0.7 | 90.7 ± 1.3% | +0.6 ± 0.2% | 90.6 ± 3.2% |
| 0.9 | 89.6 ± 1.7% | +1.7 ± 0.6% | 91.2 ± 6.9% |
| 1.0 | 48.0 ± 15.2% | +51.7 ± 15.8% | 98.7 ± 0.4% |

- Control (ρ=0) ignores the marker behaviorally → valid negative control.
- Reliance saturates fast (AFR ~91% by ρ=0.7).
- At ρ=1.0 the shortcut *displaces* genuine learning: clean acc collapses to 48% (high variance).

## Attribution localization: AM* (above chance), hit rate, detections (of 3)
| ρ | GradCAM AM* / hit / det | IG AM* / hit / det | Occlusion AM* / hit / det |
|---|---|---|---|
| 0.0 | −0.2% / 2% / 0 | 3.0% / 22% / 0 | 0.1% / 3% / 0 |
| 0.7 | 1.0% / 3% / 0 | 8.5% / 75% / 3 | 7.8% / 66% / 3 |
| 0.9 | 0.9% / 3% / 0 | 11.0% / 84% / 3 | 13.7% / 84% / 3 |
| 1.0 | 1.2% / 3% / 0 | 19.2% / 98% / 3 | 17.2% / 90% / 3 |
| **Total** | **0/12** | **9/12** | **9/12** |

## Faithfulness (Spearman across models)
| | AM* vs ρ | AM* vs ΔAcc | AM* vs AFR |
|---|---|---|---|
| GradCAM | +0.61 | +0.59 | +0.46 |
| IG | +0.95 | +0.97 | +0.86 |
| Occlusion | +0.89 | +0.91 | +0.71 |

Reliant-only (ρ≥0.7, n=9), AM* vs ρ: GradCAM +0.05, IG +0.90, Occlusion +0.74. Since AFR
saturates, AM* tracks shortcut *strength* more than it tracks the (pinned) behavioral reliance.

## Findings
1. **IG and Occlusion detect the shortcut** in all 9 reliant models and in 0 controls. AM* rises
   monotonically with ρ. (Answers RQ1, RQ2, RQ4 positively for these two methods.)
2. **Grad-CAM detects in 0/12** — BUT this is **confounded by resolution**: ResNet-18 at 64px has a
   2×2 final feature map, which cannot localize a 4px marker. This is NOT evidence Grad-CAM is blind
   to shortcuts. **Must ablate** (SmallCNN 14×14 grid, or higher input resolution) before any claim.
3. **False-positive caution:** at ρ=0, IG already puts above-chance mass on the marker (null-beat
   48%, hit 22%) though the model does not rely on it. Attribution ≠ reliance; the behavioral gate
   is what prevents a false alarm (echoes Adebayo 2022).
4. Method ranking (this setup): IG ≈ Occlusion ≫ Grad-CAM.

## Threats to validity
- 3 seeds, 1 binary task → descriptive only.
- Grad-CAM resolution confound (above).
- Visible, known shortcut only (easy regime).
- Occlusion window/stride and IG baseline fixed a priori.
