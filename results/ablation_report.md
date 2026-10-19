# VaaniShield Three-Way Ablation Study

Ablation evaluating individual and combined stream contributions across validation, in-domain test, and held-out sets.

## 1. Ablation Variant Performance (Validation Split)

| Stream / Variant | Validation EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Boundary-Local Stream Only | 0.4400 | [0.3331, 0.6361] | 0.5556 | 0.2000 |
| Speaker-Nulled Global Stream Only | 0.0000 | [0.0000, 0.0000] | 0.5814 | 1.0000 |
| **Full VaaniShield Fused Model** | 0.2800 | [0.1583, 0.5000] | 0.5556 | 0.8000 |

## 2. Ablation Variant Performance (In-Domain Test Set)

| Stream / Variant | EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Boundary-Local Stream Only | 0.5100 | [0.4002, 0.6199] | 0.4737 | 0.3600 |
| Speaker-Nulled Global Stream Only | 0.3100 | [0.2099, 0.3994] | 0.8065 | 0.5000 |
| **Full VaaniShield Fused Model** | 0.3600 | [0.2072, 0.4986] | 0.6226 | 0.6600 |

## 3. Generalization to Held-Out Synthesizer

| Stream / Variant | Held-Out Generalization EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Boundary-Local Stream Only | 0.7000 | [0.5695, 0.8424] | 0.2800 | 0.2800 |
| Speaker-Nulled Global Stream Only | 0.8800 | [0.7997, 1.0000] | 0.0000 | 0.0000 |
| **Full VaaniShield Fused Model** | 0.3400 | [0.1377, 0.4792] | 0.5882 | 0.8000 |
