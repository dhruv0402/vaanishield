# VaaniShield Three-Way Ablation Study

Ablation evaluating individual and combined stream contributions across validation, in-domain test, and held-out sets.

## 1. Ablation Variant Performance (Validation Split)

| Stream / Variant | Validation EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Boundary-Local Stream Only | 0.3200 | [0.2000, 0.4622] | 0.7143 | 0.6000 |
| Speaker-Nulled Global Stream Only | 0.1000 | [0.0208, 0.2159] | 0.7576 | 1.0000 |
| **Full VaaniShield Fused Model** | 0.0600 | [0.0000, 0.2003] | 0.6944 | 1.0000 |

## 2. Ablation Variant Performance (In-Domain Test Set)

| Stream / Variant | EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Boundary-Local Stream Only | 0.4200 | [0.3080, 0.5104] | 0.5918 | 0.5800 |
| Speaker-Nulled Global Stream Only | 0.1800 | [0.0696, 0.2597] | 0.9211 | 0.7000 |
| **Full VaaniShield Fused Model** | 0.1500 | [0.1200, 0.2400] | 0.9231 | 0.7200 |

## 3. Generalization to Held-Out Synthesizer

| Stream / Variant | Held-Out Generalization EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Boundary-Local Stream Only | 0.3200 | [0.2000, 0.4425] | 0.5238 | 0.8800 |
| Speaker-Nulled Global Stream Only | 0.1000 | [0.0200, 0.2596] | 0.8462 | 0.8800 |
| **Full VaaniShield Fused Model** | 0.1200 | [0.0178, 0.2431] | 0.7667 | 0.9200 |
