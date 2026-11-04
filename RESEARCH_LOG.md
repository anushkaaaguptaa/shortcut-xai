# Research Log — "Can Explainability Detect Shortcut Learning?"

A chronological record of decisions, their rationale, and their status.
Dates are absolute. Nothing here is a result; results will be logged only after experiments run.

---

## 2026-11-03 — Phase 1 & 2 (Literature review + gap analysis)

**Context.** Repo already contains a scaffold (MedMNIST loader, ResNet-18, Captum-based
Grad-CAM / Integrated Gradients / Occlusion, AM / Pointing-Game / faithfulness metrics,
a rho-sweep runner). Per the project plan, implementation is frozen until the methodology
is approved. This session covers Phase 1 and Phase 2 only.

**Integrity rule adopted.** No paper, author, venue, DOI, finding, or result is recorded
unless verified against a real source this session. Where only an abstract/listing was read
(not full text), the finding is tagged *(abstract-level)*. Metadata was verified via web
search for 16 papers.

**Key decision / finding affecting novelty.** Two papers by Adebayo et al. (NeurIPS 2020,
ICLR 2022) already pose essentially this project's question — "can post-hoc explanations
detect reliance on a known/injected spurious signal?" — and give a *nuanced* answer
(works for visible/known artifacts, fails for non-visible/unknown ones). This means the
project as literally framed ("Can XAI detect shortcut learning?") is **not novel** and must
be repositioned. See Phase 2 analysis. Decision on the specific contribution is **pending
user approval**.

**Open items before Phase 3.**
- User to choose a contribution direction (3 options proposed).
- Dataset licensing/suitability review (Phase 3) not yet done; MedMNIST in scaffold not yet
  justified against alternatives.
- No experiments run. No results exist.

---

## 2026-11-03 — Phase 3 (Methodology draft)

**Decisions (user-confirmed):** Option 1 (dose-response benchmark); 2-day mini-study;
dataset = PneumoniaMNIST; marker placement = random per image; compute = unknown/Colab, so
design the safe path (28×28 + small CNN) with a GPU upgrade path (64×64 + ResNet-18).

**Design choices (mine, documented in METHODOLOGY.md):** one-sided injection scheme so ρ=0 is a
clean baseline; ρ ∈ {0.0,0.7,0.9,1.0} × 3 seeds = 12 models (CPU fallback = 6); behavioral reliance
measured before trusting any attribution; AM normalized against a random-box null to kill area bias;
"detection" requires attribution localization AND measured behavioral reliance (guards against the
Adebayo-2022 false-positive failure mode); n is small so only descriptive trends are claimed.

**Status:** METHODOLOGY.md written; user approved; proceeded to Phase 4.
Three minor defaults flagged for confirmation (marker shape, size s=4, AFR threshold 0.3).

---

## 2026-11-03 — Phase 4 (Implementation)

**Reconciled scaffold with the approved methodology.** Changes: (1) marker placement rewritten
from fixed top-right corner to seeded **random** placement with exact per-image bbox; (2) added
full seed control (`set_seed`, seeded DataLoader) and a `seed` arg — training-set injection was
previously non-deterministic; runner now trains ρ×seed = 12 models by default; (3) added
`AM*` (AM minus random-box chance) and a bootstrap random-box null to `metrics.py`; (4) added a
`SmallCNN` (CPU-safe, 14×14 CAM grid) with auto arch selection; (5) attribution now runs on the
**Conflict** set for the model's predicted class; (6) added the detection rule (null-beat >0.5
AND AFR > threshold); (7) figures now show mean±std across seeds and an AFR-vs-AM* scatter.

**Documented deviations:** ImageNet normalization kept (pretrained backbone); IG 50 steps;
occluder size = marker size, stride = marker/2. All recorded in METHODOLOGY.md §10b.

**Verification done locally:** all 9 modules byte-compile; `metrics.py` numerically validated
(AM, AM*, chance, random-box null, faithfulness corr all correct). **NOT yet run:** the
torch-dependent path (dataset/model/train/evaluate/explain/runner) — torch/torchvision/captum/
medmnist are not installed in the local Python 3.10. End-to-end execution pending a torch env
(Colab GPU or a local CPU install). **No experimental results exist yet.**

**Next:** smoke-test the full pipeline (`--quick_test`) in a torch env before the real 12-model run.

---

## 2026-11-03 — Phase 4 smoke test (Colab GPU, ResNet-18) + fixes

**First execution.** `--quick_test` ran end-to-end on Colab (CUDA, ResNet-18). The phenomenon
reproduced cleanly:
- ρ=0.0 control: clean acc 87.2%, ΔAcc -0.3% (~0), AFR 13.6% (below 0.3 threshold -> not reliant).
- ρ=1.0 shortcut: clean acc 74.8% (DEGRADED from 87%), concordant 96.3%, ΔAcc +21.5%, AFR 96.2%.
  Strong, unambiguous shortcut reliance. (These are smoke-test numbers, 1 seed, NOT final results.)

**Two fixes applied:**
1. Added `tests/__init__.py` so `python -m unittest tests.test_pipeline` imports (was
   ModuleNotFoundError: tests is not a package).
2. **Checkpoint selection changed to biased-val (default), with `--val_clean` escape hatch.**
   The ρ=1.0 run exposed that clean-val selection saved an early under-trained epoch and works
   against the study's goal (see METHODOLOGY.md §10b). Also added per-method AM*/hit/null-beat
   console printout.

**Caveat for re-runs:** cached checkpoints in ./checkpoints were trained under the OLD clean-val
selection. Delete ./checkpoints (and ./results) before re-running so the new selection takes effect.

**Status:** pipeline validated end-to-end; still no FINAL results (need the multi-seed run with
the corrected selection). Phase 5 analysis begins only after that run.

---

## 2026-11-03 — Phase 5 (Results) + Phase 6 start (Paper)

**Full 12-model run completed** (user, Colab GPU, ResNet-18, biased-val selection). Aggregates
were computed from per_model_metrics.csv (not transcribed) and written to RESULTS.md. Headline:
IG and Occlusion detect 9/9 reliant models and 0/3 controls; Grad-CAM 0/12 but **confounded by a
2x2 feature map** (cannot resolve a 4px marker). IG shows a false-positive tendency at rho=0 that
the behavioral gate correctly suppresses. n=3 -> descriptive only.

**Decision:** user confirmed this is a college project; the Grad-CAM resolution ablation is
OPTIONAL (not run yet). Agreed constraint: the paper must frame Grad-CAM's failure as
resolution-confounded, not as a clean finding.

**Phase 6 started.** Created paper/references.bib (16 verified refs + MedMNIST/PyTorch/Captum/
ResNet; MedMNIST v2 metadata verified this session) and paper/main.tex. Data-anchored sections
(Methodology, Experimental Setup, Results, Abstract, Limitations) written in full from the real
numbers; Intro/Related Work/Background/Discussion/Conclusion are first drafts to expand next.
LaTeX not compilable locally (no toolchain) -> compile on Overleaf; main.tex uses a \safefig
macro so it builds before figures are added.

**Next:** (optional) Grad-CAM ablation; expand Intro/Related/Discussion; then Phase 7 review.
