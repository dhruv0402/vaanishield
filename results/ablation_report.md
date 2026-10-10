# VaaniShield Three-Way Ablation Study

Ablation evaluating individual and combined stream contributions across in-domain test and held-out sets.

## 1. Ablation Variant Performance (In-Domain Test Set)

| Stream / Variant | EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Boundary-Local Stream Only | 0.4800 | [0.3397, 0.6333] | 1.0000 | 0.2000 |
| Speaker-Nulled Global Stream Only | 0.2800 | [0.1523, 0.4059] | 0.7500 | 0.6000 |
| **Full VaaniShield Fused Model** | 0.2000 | [0.0862, 0.3237] | 0.7778 | 0.8400 |

## 2. Generalization to Held-Out Synthesizer

| Stream / Variant | Held-Out Generalization EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Boundary-Local Stream Only | 0.5400 | [0.4050, 0.6827] | 0.0000 | 0.0000 |
| Speaker-Nulled Global Stream Only | 0.2800 | [0.1499, 0.4319] | 0.7083 | 0.6800 |
| **Full VaaniShield Fused Model** | 0.1800 | [0.0714, 0.2915] | 0.7857 | 0.8800 |
