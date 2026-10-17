# VaaniShield Three-Way Ablation Study

Ablation evaluating individual and combined stream contributions across validation, in-domain test, and held-out sets.

## 1. Ablation Variant Performance (Validation Split)

| Stream / Variant | Validation EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Boundary-Local Stream Only | 0.3600 | [0.2275, 0.4856] | 0.5897 | 0.9200 |
| Speaker-Nulled Global Stream Only | 0.1600 | [0.0192, 0.2415] | 0.9524 | 0.8000 |
| **Full VaaniShield Fused Model** | 0.1000 | [0.0172, 0.2231] | 1.0000 | 0.8000 |

## 2. Ablation Variant Performance (In-Domain Test Set)

| Stream / Variant | EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Boundary-Local Stream Only | 0.3600 | [0.2726, 0.4584] | 0.5750 | 0.9200 |
| Speaker-Nulled Global Stream Only | 0.1600 | [0.0901, 0.2404] | 0.8235 | 0.8400 |
| **Full VaaniShield Fused Model** | 0.1200 | [0.0404, 0.2099] | 0.8182 | 0.9000 |

## 3. Generalization to Held-Out Synthesizer

| Stream / Variant | Held-Out Generalization EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Boundary-Local Stream Only | 0.3400 | [0.2225, 0.5385] | 0.6286 | 0.8800 |
| Speaker-Nulled Global Stream Only | 0.0800 | [0.0000, 0.2000] | 0.8571 | 0.9600 |
| **Full VaaniShield Fused Model** | 0.0600 | [0.0000, 0.1981] | 0.8889 | 0.9600 |
