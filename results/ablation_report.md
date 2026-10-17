# VaaniShield Three-Way Ablation Study

Ablation evaluating individual and combined stream contributions across in-domain test and held-out sets.

## 1. Ablation Variant Performance (In-Domain Test Set)

| Stream / Variant | EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Boundary-Local Stream Only | 0.5900 | [0.4998, 0.6791] | 0.4098 | 0.5000 |
| Speaker-Nulled Global Stream Only | 0.2700 | [0.1797, 0.3700] | 0.7391 | 0.6800 |
| **Full VaaniShield Fused Model** | 0.1600 | [0.1200, 0.2000] | 0.9211 | 0.7000 |

## 2. Generalization to Held-Out Synthesizer

| Stream / Variant | Held-Out Generalization EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Boundary-Local Stream Only | 0.3800 | [0.2386, 0.5909] | 0.4706 | 0.6400 |
| Speaker-Nulled Global Stream Only | 0.2200 | [0.1200, 0.3831] | 0.6667 | 0.8000 |
| **Full VaaniShield Fused Model** | 0.1600 | [0.0172, 0.2485] | 0.7667 | 0.9200 |
