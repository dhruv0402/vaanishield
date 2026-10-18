# VaaniShield Three-Way Ablation Study

Ablation evaluating individual and combined stream contributions across validation, in-domain test, and held-out sets.

## 1. Ablation Variant Performance (Validation Split)

| Stream / Variant | Validation EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Boundary-Local Stream Only | 0.4400 | [0.3331, 0.6361] | 0.5556 | 0.2000 |
| Speaker-Nulled Global Stream Only | 0.1200 | [0.0000, 0.2000] | 1.0000 | 0.6800 |
| **Full VaaniShield Fused Model** | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0.7200 |

## 2. Ablation Variant Performance (In-Domain Test Set)

| Stream / Variant | EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Boundary-Local Stream Only | 0.5100 | [0.4002, 0.6199] | 0.4737 | 0.3600 |
| Speaker-Nulled Global Stream Only | 0.2200 | [0.1099, 0.2899] | 0.8810 | 0.7400 |
| **Full VaaniShield Fused Model** | 0.1400 | [0.0200, 0.3001] | 0.8837 | 0.7600 |

## 3. Generalization to Held-Out Synthesizer

| Stream / Variant | Held-Out Generalization EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Boundary-Local Stream Only | 0.7000 | [0.5695, 0.8424] | 0.2800 | 0.2800 |
| Speaker-Nulled Global Stream Only | 0.6600 | [0.5200, 0.8820] | 0.0000 | 0.0000 |
| **Full VaaniShield Fused Model** | 0.7400 | [0.3623, 1.0000] | 0.0000 | 0.0000 |
