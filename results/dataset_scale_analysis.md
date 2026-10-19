# VaaniShield Dataset Scale & Statistical Generalization Analysis

> **Audit Status**: Academic Proof-of-Concept Pilot → Production Banking Roadmap

## 1. Scale Comparison: Pilot Corpus vs. Enterprise Banking Deployment

| Dimension | Current Pilot Corpus | Commercial Banking Requirement | Scaling Factor |
| :--- | :--- | :--- | :--- |
| **Total Utterances** | 400 | 50,000 | **125x** |
| **Distinct Speakers** | 8 | 2,000 | **250x** |
| **Held-Out Split Size** | 50 (1 speaker: `spk_5945`) | 10,000 (400+ unseen speakers) | **200x** |
| **Held-Out 95% CI Margin** | **±12.7%** ([0.22, 0.48]) | **±0.93%** ([0.3308, 0.3493]) | **13.7x tighter** |
| **Indian States Represented** | 3 (Delhi, TN, Karnataka) | 28 States + 8 Union Territories | **12x geographical coverage** |
| **Language Pairs** | 3 (hi-en, ta-en, kn-en) | 12 Indic-English Code-Switched Pairs | **4x linguistic diversity** |
| **Minimum Clips for 0.1% FAR** | Insufficient (N=50) | >= 3,000 fraud attempts | **Statistically Certified** |

## 2. Statistical Confidence Intervals & Empirical Bounds

- **Current Held-Out EER**: 0.3400 with 95% Wilson Score Interval `[0.2244, 0.4785]`.
- **Statistical Uncertainty**: On N=50 clips, the margin of error is ±12.7%. While this empirically establishes that the model does not suffer random guessing (chance = 0.50), it cannot certify fine-grained sub-1% false rejection or fraud leakage rates.
- **Production Target**: At N=10,000 held-out clips, the margin of error compresses to ±0.93%, satisfying Reserve Bank of India (RBI) and European Banking Authority (EBA) RTS criteria for biometric anti-spoofing certification.
