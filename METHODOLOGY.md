# Phase 3 — Experimental Methodology (Option 1: Dose-Response Benchmark)

**Status:** DRAFT for approval (2026-11-03). No experiments run. No results exist.
**Question:** *How consistently do Grad-CAM, Integrated Gradients, and Occlusion Sensitivity
localize an artificial shortcut marker, and how does that localization track the model's
true behavioral reliance on the marker, as the training correlation strength ρ varies?*
(Research questions RQ3 consistency + RQ4 robustness. This is a controlled replication-and-
extension of Adebayo et al. 2020/2022 in a medical-imaging, dose-response setting — not a
novel question.)

---

## 1. Dataset & preprocessing
- **Task:** PneumoniaMNIST (MedMNIST v2). Binary: class 0 = normal, class 1 = pneumonia.
- **Splits:** Use the official train / val / test split shipped with MedMNIST (do not reshuffle).
- **Resolution:** 28×28 grayscale (safe path). *Upgrade path:* 64×64 + ResNet-18 if a GPU is present.
- **Normalization:** per-dataset mean/std computed on the training split only.
- **No augmentation** in the mini-study (augmentation interacts with the marker and adds confounds).

## 2. Shortcut (marker) definition
- **Appearance:** a solid square patch of side `s` pixels at a fixed high intensity
  (post-normalization max value). Safe path: `s = 4` on 28×28 (≈2% of pixels). The marker is
  deliberately **visible** — we are testing the visible-shortcut regime, not Adebayo's non-visible case.
- **Placement:** **random per image**, sampled uniformly so the full `s×s` patch fits inside the
  image. The exact pixel bounding box `B` is recorded per image (seeded RNG) and saved to disk, so
  attribution ground truth is exact and reproducible.
- **Why random (not fixed corner):** a fixed corner lets the model learn *position* instead of
  *marker*, confounding "localizes the marker" with "localizes a corner." Random placement forces
  the localization metrics to actually find the marker. Cost: higher variance — mitigated by seeds
  and by reporting trends, not significance.

## 3. Injection scheme (the ρ knob)
At training strength **ρ ∈ [0,1]**:
- **Disease images (y=1):** receive a marker with probability **ρ**.
- **Normal images (y=0):** never receive a marker.

Consequences:
- **ρ = 0.0** → no markers anywhere → *standard training* (prompt Condition A, the control).
- **ρ = 0.7, 0.9** → marker present on most diseased images → a *noisy* predictive shortcut.
- **ρ = 1.0** → marker on every diseased image, none on normal → *maximally predictive* shortcut
  (prompt Condition B at full strength).

"Marker present ⇒ disease" is a one-sided cue (mirrors the real Clever-Hans tag-on-target-class
case). Rationale for one-sided rather than symmetric (markers on both classes anti-/correlated):
it keeps ρ=0 a clean baseline and keeps the causal story unambiguous.

**Models trained:** ρ ∈ {0.0, 0.7, 0.9, 1.0} × seeds {0,1,2} = **12 models**.
(CPU-only fallback: drop to ρ ∈ {0.0, 0.9, 1.0} × seeds {0,1} = 6 models.)

## 4. Test-time conditions (applied to the held-out official test split)
Each trained model is evaluated on four transformed copies of the **same** test images:
1. **Clean** — no marker. Measures reliance on genuine pathology features.
2. **Concordant** — marker injected following the training rule (disease→marker, prob 1.0 at test).
   Measures performance when the shortcut agrees with the label.
3. **Conflict (adversarial)** — marker stamped on **normal** test images (the "wrong" class).
   If the model relies on the shortcut, these flip to "pneumonia."
4. **Shortcut-free** — identical to Clean under this one-sided scheme (noted explicitly; the
   informative contrasts are Clean vs Concordant vs Conflict).

## 5. Metrics (exact definitions)

### 5a. Behavioral reliance (establish this FIRST — it is the ground truth)
- **Clean accuracy:** `Acc_clean` on condition (1).
- **Shortcut gain:** `ΔAcc = Acc_concordant − Acc_clean`. Positive ⇒ the model gains from the marker
  being present, i.e. it relies on the shortcut.
- **Adversarial flip rate:**
  `AFR = (# normal images predicted "pneumonia" in Conflict) / (# normal images)`,
  counting only images the model classified correctly when Clean. AFR→1 is strong shortcut reliance.

### 5b. Attribution localization (computed for Grad-CAM, IG, Occlusion)
For an attribution map `A(i,j)` and marker box `B`:
- **Attribution Mass:** `AM = Σ_{(i,j)∈B} |A(i,j)| / Σ_{all (i,j)} |A(i,j)|`.
- **Chance level:** a random box of the same size captures `|B|/|image|` of the mass in expectation.
  Report **AM-above-chance:** `AM* = AM − |B|/|image|` (and the ratio `AM / (|B|/|image|)`).
  This controls the area-bias confound.
- **Pointing Game hit rate:** fraction of images where `argmax A ∈ B`. Chance = `|B|/|image|`.
- **Evaluation set for attribution:** the **Conflict** images (marker on normal images). We ask the
  attribution for the *predicted* class: if the model took the bait and the method is faithful, mass
  should concentrate on `B`. Attribution is also computed on Concordant images as a secondary view.

### 5c. Faithfulness trend (secondary, descriptive)
Treat each (model) as one datapoint: plot/correlate behavioral reliance (`AFR`) against localization
(`AM*`) across the 12 models, per method. With n≈12 this is a **descriptive** association
(report Spearman ρ with a CI, state clearly it is not powered for significance).

## 6. "Detection succeeded" — decision rule
For a given model and method, the shortcut is **detected** if `AM*` is above the per-image random-box
null (bootstrap over 1000 random boxes, 95%) for a majority of Conflict images **and** the model is
behaviorally shortcut-reliant (`AFR` above a stated threshold, e.g. 0.3). Tying detection to measured
behavior is the guard against the Adebayo-2022 false-positive trap (attribution flagging a marker the
model does not actually use).

## 7. Reproducibility
- Single seed controls weights, data order, and marker RNG; `torch`/`numpy`/`cudnn` set deterministic.
- One config file per run (dataset, ρ, seed, `s`, epochs, lr, batch) saved beside outputs.
- Save: model checkpoint, per-image marker boxes, per-condition predictions, metrics CSV, logs.
- Training: Adam, lr 1e-3, batch 128, ≤15 epochs, early stop on clean-val accuracy, cross-entropy.

## 8. Confounds and how each is handled
| Confound | Handling |
|---|---|
| Marker small vs method resolution (Grad-CAM coarse) | Report `s` in pixels and in last-conv receptive-field units; interpret Grad-CAM accordingly. |
| Occlusion occluder size / stride; IG baseline | Fix a priori (occluder = `s`, stride = `s/2`; IG baseline = black image, 50 steps); note as fixed choices; ablate IG baseline if time allows. |
| AM area bias | `AM*` against random-box null (§5b). |
| Attribution of an unused marker (false positive) | Detection requires behavioral reliance too (§6). |
| Few seeds | Report mean±std and trends only; state n and that significance is not claimed. |

## 9. What this study will and will NOT claim
- **Will:** describe how each method's localization of a *known, visible* injected marker behaves as
  ρ varies, and whether localization tracks behavioral reliance, on one binary medical task.
- **Will NOT:** claim statistical significance, generalization across datasets/architectures,
  detection of *unknown* or *non-visible* shortcuts, or that any method is reliable for clinical use.

## 10. 2-day timeline
- **Day 1:** finalize config; validate injector (visual check of 6 sample images + saved boxes);
  train 12 models (ρ×seed); save checkpoints + behavioral metrics.
- **Day 2:** run the three attribution methods on Clean/Concordant/Conflict; compute AM*, Pointing
  Game, AFR; generate figures (localization-vs-ρ curves per method; AFR-vs-AM* scatter); write results.

## 10b. Documented deviations from this spec (Phase 4 implementation)
- **Normalization:** ImageNet mean/std is used (not per-dataset stats), because the ResNet-18
  backbone is ImageNet-pretrained; for the from-scratch SmallCNN it is a harmless fixed affine.
- **Architecture:** auto-selects — SmallCNN on CPU (no download, 14×14 Grad-CAM grid), ResNet-18
  on GPU/MPS. Overridable with `--arch`.
- **Attribution target:** on the Conflict set we explain the model's *predicted* class (what it
  used to decide), per §5b.
- **Checkpoint selection (added after first smoke test):** select the best epoch on a validation
  set drawn from the SAME biased distribution as training (not a clean val set). Clean-val
  selection secretly rewards the least shortcut-reliant model — an implicit mitigation that
  biases the dose-response curve. `--val_clean` reverts to clean-val selection if ever needed.
  Rationale confirmed empirically: the ρ=1.0 smoke run validated at 25-72% on clean data
  (because the marker-trained model fails on unmarked images) and saved an early, under-trained
  epoch; biased-val selection reflects the realistic in-distribution pipeline a practitioner runs.

## 11. Open defaults awaiting confirmation
- Marker shape/intensity (solid white square vs a small distinct glyph) — defaulting to solid square.
- `s = 4` at 28px (≈2% area) — adjustable.
- AFR reliance threshold for the detection rule (0.3) — adjustable.
