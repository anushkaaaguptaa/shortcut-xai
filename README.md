# Can Explainability Detect Shortcut Learning?

A rigorous, controlled experimental benchmark investigating whether standard Explainable AI (XAI) attribution methods—**Grad-CAM**, **Integrated Gradients**, and **Occlusion Sensitivity**—can reliably detect when a convolutional neural network relies on an artificial shortcut rather than disease-relevant features.

---

## 🔬 Core Research Questions
1. **Localization**: Do explanation heatmaps reliably place attribution mass on the shortcut bounding box?
2. **Faithfulness**: Does the attribution score on the shortcut correlate with the model's actual behavioral dependency when the shortcut is removed or swapped?

---

## 🛠️ Project Structure

```
shortcut-xai/
├── src/
│   ├── dataset.py        # MedMNIST loader + synthetic shortcut injector with GT bbox
│   ├── model.py          # ResNet-18 with adaptive upsampling for medical images
│   ├── train.py          # Training loop across controlled bias strengths (rho)
│   ├── evaluate.py       # Behavioral evaluation: Accuracy drop & Adversarial flip rate
│   ├── explain.py        # Captum wrapper (Grad-CAM, Integrated Gradients, Occlusion)
│   ├── metrics.py        # Attribution Mass (AM), Pointing Game, Spearman correlation
│   └── visualize.py      # Heatmap comparison panels and sensitivity curves
├── tests/
│   └── test_pipeline.py  # Automated unit tests for end-to-end verification
├── run_experiments.py    # Master runner executing the 4-level bias sweep
├── requirements.txt      # Dependency list
└── README.md
```

---

## 🚀 Quick Start

### 1. Run Pipeline Unit Tests
```bash
python3 -m unittest tests/test_pipeline.py
```

### 2. Run a Quick Test (Bias rho = 0.0 vs 1.0)
```bash
python3 run_experiments.py --quick_test --epochs 3 --samples 20
```

### 3. Run the Full Controlled 4-Level Bias Sweep (rho = 0.0, 0.7, 0.9, 1.0)
```bash
python3 run_experiments.py --epochs 5 --samples 50
```

---

## 📊 Evaluation Metrics

### 1. Behavioral Shortcut Reliance
- **Clean Accuracy ($\text{Acc}_{\text{clean}}$)**: Performance on unaltered original images.
- **Performance Drop ($\Delta \text{Acc}$)**: $\text{Acc}_{\text{concordant}} - \text{Acc}_{\text{clean}}$ (how much the model degrades when the shortcut is absent).
- **Adversarial Flip Rate**: The percentage of healthy lungs (Class 0) misclassified as diseased (Class 1) purely because the shortcut marker was stamped onto them.

### 2. XAI Attribution & Faithfulness
- **Attribution Mass in Bounding Box ($\text{AM}$)**:
  $$\text{AM} = \frac{\sum_{(i,j) \in \text{Marker Box}} |\text{Attribution}(i,j)|}{\sum_{\text{all pixels}} |\text{Attribution}(i,j)|}$$
- **Pointing Game Hit Rate**: 1 if $\arg\max(\text{Attribution})$ falls inside the shortcut bounding box, 0 otherwise.
- **Faithfulness Correlation ($r$)**: Spearman/Pearson correlation between the attribution mass on the shortcut and the logit drop $\Delta f(x)$ when that shortcut is removed.

---

## 📅 14-Day Research Schedule
- **Days 1–3**: Environment setup, synthetic dataset validation, baseline model training ($\rho=0.0$).
- **Days 4–6**: Controlled training across bias levels ($\rho \in \{0.0, 0.7, 0.9, 1.0\}$) & behavioral analysis.
- **Days 7–9**: Explanation extraction (Grad-CAM, Integrated Gradients, Occlusion) on test sets.
- **Days 10–11**: Quantitative metric calculation ($\text{AM}$, Hit Rate, Faithfulness correlation).
- **Days 12–14**: Figure generation, paper/report synthesis, and conclusions.
