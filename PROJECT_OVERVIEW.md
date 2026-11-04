# Project Overview — "Can Explainability Detect Shortcut Learning?"

A plain-language walkthrough of the entire project: the question, why it matters, how the
experiment works, what every file does, what we found, and what it does and does not prove.
Written as a learning guide. For the formal design see [METHODOLOGY.md](METHODOLOGY.md); for the
numbers see [RESULTS.md](RESULTS.md); for the decision history see [RESEARCH_LOG.md](RESEARCH_LOG.md);
for the write-up see [paper/main.tex](paper/main.tex).

---

## 1. The one-sentence idea
We deliberately teach an AI a "cheating" shortcut, then test whether three popular "explanation"
tools can catch it cheating.

## 2. Why this matters
A neural network can score high on a test for the *wrong reason*. Instead of learning the real
signal, it latches onto an accidental cue that happens to line up with the answer. This is called
**shortcut learning**. Real, documented examples in medicine:
- A pneumonia model that secretly learned to recognise *which hospital* an X-ray came from (Zech et al. 2018).
- COVID X-ray models that keyed on scanner/text artifacts, not lungs (DeGrave et al. 2021).
- A skin-cancer model fooled by a doctor's purple skin-marker pen (Winkler et al. 2019).

These models looked accurate but were unsafe. People often suggest **Explainable AI (XAI)** —
tools that draw a "heatmap" of what the model looked at — as a safety check. Our project asks:
**do those heatmaps actually reveal the cheating?**

## 3. The experiment in four moves
1. **Plant a shortcut.** We take real chest X-rays and stamp a tiny bright 4×4-pixel square (a
   fake "marker") onto *diseased* images, at a *random spot* each time.
2. **Control how strong the cheat is.** A dial called **ρ (rho)** sets how reliably the marker
   appears on diseased images:
   - ρ = 0.0 → no marker at all (honest baseline / control)
   - ρ = 0.7, 0.9 → marker usually present (a tempting but noisy cheat)
   - ρ = 1.0 → marker always on disease, never on healthy (a perfect cheat)
3. **Train models and check if they cheat.** We train 12 models (4 strengths × 3 random seeds).
   Then we *behaviorally* test reliance: if we stamp the marker onto a **healthy** lung, does the
   model suddenly cry "pneumonia"? If yes, it's relying on the shortcut.
4. **Ask the explanation tools.** We run three XAI methods and measure whether their heatmaps land
   on the marker. If a method's heatmap lights up the fake marker, it "detected" the shortcut.

## 4. The three explanation tools
- **Grad-CAM** — looks at the last convolutional layer and makes a coarse heatmap. Fast, but
  low-resolution.
- **Integrated Gradients (IG)** — traces how the prediction changes as the image fades in from a
  blank baseline; gives a fine, pixel-level map.
- **Occlusion Sensitivity** — slides a little blank patch over the image and watches where covering
  up changes the prediction most.

They use three different mechanisms, which is why comparing them is interesting.

## 5. What we measure (and why)
**Behavioral metrics — "is the model actually cheating?"** (we check this FIRST)
- **Clean accuracy** — accuracy on normal, unmarked images (how good is its *real* skill?).
- **ΔAcc** — how much better it does when the marker is present vs absent (a gain means it leans on
  the marker).
- **Adversarial Flip Rate (AFR)** — the headline cheat detector: of healthy lungs the model got
  right when clean, what fraction flip to "pneumonia" once we stamp the marker on them? High AFR =
  heavy reliance.

**Attribution metrics — "does the heatmap find the marker?"**
- **Attribution Mass (AM)** — fraction of the heatmap's "heat" that falls inside the marker box.
- **AM\*** (AM above chance) — AM minus what a *random* box of the same size would catch by luck.
  This is the fair version; a big box catches heat by accident, and AM\* removes that advantage.
- **Pointing-game hit rate** — does the single hottest pixel land inside the marker?

**The detection rule (the careful part):** a method only counts as "detecting the shortcut" if
(a) its heatmap beats the random-box null AND (b) the model is *actually* behaviorally reliant
(AFR > 0.3). Condition (b) stops us from being fooled by a heatmap that lights up a marker the
model doesn't even use.

## 6. What we found (12 models, 3 seeds — descriptive trends, not statistical proof)
1. **The shortcut works as designed.** At ρ=0 the model ignores the marker (AFR ~7%). At ρ≥0.7 it's
   hooked (AFR ~91%). At ρ=1.0 it's *so* dependent that its real skill collapses — clean accuracy
   drops from ~85% to ~48%.
2. **IG and Occlusion catch the cheat.** Both localize the marker in all 9 cheating models and in
   none of the 3 honest controls, and their signal (AM\*) rises steadily as the cheat gets stronger
   (Spearman +0.95 and +0.89 vs ρ).
3. **Grad-CAM appears to fail — but it's a trap.** Grad-CAM caught 0/12. HOWEVER, on our network its
   heatmap is only 2×2 pixels of real resolution, which physically cannot pinpoint a 4-pixel marker.
   So we *cannot* say "Grad-CAM is blind to shortcuts" — the test was unfair to it. This needs a
   follow-up ("ablation") before any claim.
4. **A warning about false alarms.** Even at ρ=0 (no cheating), IG put some heat on the marker just
   because it's a bright patch. That's why we insist on the behavioral check — a heatmap reacting to
   something visible is NOT proof the model relies on it.

## 7. The code, file by file
```
src/dataset.py      Loads PneumoniaMNIST; stamps the random marker; records its exact box.
src/model.py        The classifier: ResNet-18 (on GPU) or a small CNN (on CPU).
src/train.py        Training loop; fixes all random seeds for reproducibility.
src/evaluate.py     Behavioral metrics (clean acc, ΔAcc, AFR).
src/explain.py      Runs Grad-CAM / IG / Occlusion via the Captum library.
src/metrics.py      AM, AM*, random-box null, pointing-game, correlations.
src/visualize.py    Makes the figures (dose-response curves, scatter, heatmap panels).
run_experiments.py  The conductor: trains all 12 models, runs XAI, applies the detection rule,
                    writes results/per_model_metrics.csv + summary_by_rho.csv + figures.
tests/              Unit tests that check the pieces work.
paper/              The LaTeX write-up (main.tex) and verified references (references.bib).
```

## 8. What this project does and does NOT establish
**Does:** In a controlled medical-image setting with a *visible, known* shortcut, perturbation- and
gradient-based attributions (Occlusion, IG) can localize the shortcut, and their signal scales with
how strong the shortcut is — *provided* you confirm reliance behaviorally first.

**Does NOT:** It does not prove this generalizes to other datasets, to subtle/invisible shortcuts,
or to other architectures. It does not claim statistical significance (only 3 seeds). It does not
show Grad-CAM is fundamentally worse (that result is resolution-confounded).

## 9. Honest limitations
- Small scale: 3 seeds, one binary task, one architecture → trends, not proof.
- Grad-CAM comparison is unfair until the resolution ablation is run.
- Only a visible, obvious shortcut (the easy case).
- A few knobs (occluder size, IG baseline) are fixed by choice and could shift results.

## 10. Where the project stands (the 7 phases)
- **Phase 1 Literature review** ✅ — 16 verified papers.
- **Phase 2 Gap analysis** ✅ — honest verdict: the broad question isn't novel (Adebayo 2020/2022);
  our angle is the dose-response + behavioral gate.
- **Phase 3 Methodology** ✅ — [METHODOLOGY.md](METHODOLOGY.md).
- **Phase 4 Implementation** ✅ — code above, runs on Colab.
- **Phase 5 Results** ✅ — [RESULTS.md](RESULTS.md).
- **Phase 6 Paper** 🔄 — data sections done; Intro/Related Work done; Background/Discussion/
  Conclusion still drafts.
- **Phase 7 Critical review** ⏳ — not done yet.

**Realistic verdict:** excellent as a college project; would need the Grad-CAM ablation, more seeds,
and one more axis of generality to be workshop-competitive.

## 11. How to reproduce it
```bash
pip install -r requirements.txt
python -m unittest tests.test_pipeline          # sanity check
python run_experiments.py --quick_test          # ~2 min smoke test
python run_experiments.py --epochs 10           # full 12-model run (GPU recommended)
```
Outputs land in `results/` (gitignored). On a GPU the full run auto-selects ResNet-18; on CPU it
uses the small CNN.

## 12. Mini-glossary
- **Shortcut / spurious correlation** — an accidental cue the model uses instead of the real signal.
- **Attribution / saliency / heatmap** — a map of which pixels mattered to the prediction.
- **ρ (rho)** — our dial for how strongly the fake marker is tied to the disease label.
- **Conflict set** — test images where we put the marker on the *wrong* class to expose reliance.
- **AFR** — adversarial flip rate; how often a healthy image flips to "diseased" when marked.
- **AM\*** — attribution mass on the marker, corrected for what a random box would catch.
- **Seed** — a fixed random number so a run can be repeated exactly.
- **Ablation** — a follow-up experiment that changes one thing to test whether it caused a result.
