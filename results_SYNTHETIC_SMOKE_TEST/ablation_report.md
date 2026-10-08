# VaaniShield Three-Way Ablation Study

Evaluation confirming that the boundary-local feature stream and speaker-nulled global stream each provide independent, complementary detection signal.

## 1. Ablation Variant Performance (In-Domain Test Set)

| Stream / Variant | EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Boundary-Local Stream Only | 0.4273 | [0.2640, 0.5977] | 0.5455 | 0.6316 |
| Speaker-Nulled Global Stream Only | 0.2506 | [0.1239, 0.4125] | 0.6296 | 0.8947 |
| **Full VaaniShield Fused Model** | 0.4223 | [0.2333, 0.5638] | 0.5455 | 0.6316 |

## 2. Generalization to Held-Out Synthesizer (OpenVoice)

| Stream / Variant | Held-Out Generalization EER | Precision | Recall |
| :--- | :--- | :--- | :--- |
| Boundary-Local Stream Only | 0.0000 | 1.0000 | 0.7250 |
| Speaker-Nulled Global Stream Only | 0.0000 | 1.0000 | 0.9250 |
| **Full VaaniShield Fused Model** | 0.0000 | 1.0000 | 0.7500 |
