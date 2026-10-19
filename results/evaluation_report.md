# VaaniShield Evaluation & Baseline Comparison Report

**Dataset Partition**: Train = `200` | Validation (`val`) = `50` | In-Domain Test (`test`) = `100` | Held-Out Synthesizer (`heldout`) = `50`

## 1. Validation Split Performance (Calibration & Hyperparameter Selection Set)

> **Verdict**: VaaniShield achieves comparable performance to baselines (EER 0.2800 [CI: 0.1583, 0.5000] vs best baseline 0.2400); differences are within statistical margin of error.

| Model | Validation EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Baseline A (Generic SSL) | 0.2400 | [0.1795, 0.5593] | 0.5714 | 0.6400 |
| Baseline B (Indian Accent Mel+LFCC) | 0.3400 | [0.1410, 0.4725] | 0.5000 | 0.8000 |
| **VaaniShield (Boundary + SNAP Fusion)** | 0.2800 | [0.1583, 0.5000] | 0.5556 | 0.8000 |

## 2. Overall Performance Comparison (In-Domain Test Set)

> **Verdict**: VaaniShield achieves comparable performance to baselines (EER 0.3600 [CI: 0.2072, 0.4986] vs best baseline 0.3900); differences are within statistical margin of error.

| Model | EER (Equal Error Rate) | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Baseline A (Generic SSL) | 0.3900 | [0.2586, 0.5201] | 0.6596 | 0.6200 |
| Baseline B (Indian Accent Mel+LFCC) | 0.4800 | [0.2881, 0.5700] | 0.5690 | 0.6600 |
| **VaaniShield (Boundary + SNAP Fusion)** | 0.3600 | [0.2072, 0.4986] | 0.6226 | 0.6600 |

## 3. Generalization to Unseen Synthesizer (Held-Out Set)

> **Verdict**: VaaniShield demonstrates an EER of 0.3400 compared to best baseline 0.1200; baseline performance is higher on this configuration.

| Model | Generalization EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Baseline A (Generic SSL) | 0.1200 | [0.0435, 0.3751] | 0.6897 | 0.8000 |
| Baseline B (Indian Accent Mel+LFCC) | 0.1400 | [0.0582, 0.3809] | 0.6250 | 0.8000 |
| **VaaniShield (Boundary + SNAP Fusion)** | 0.3400 | [0.1377, 0.4792] | 0.5882 | 0.8000 |

## 4. Performance Breakdown by Telecom Noise Condition (VaaniShield Test Split)

| Noise Condition | EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| background_noise_injection | 0.5000 | [0.1010, 0.6907] | 1.0000 | 0.1000 |
| clean | 0.3000 | [0.0000, 0.6042] | 0.6154 | 0.8000 |
| codec_compression | 0.3000 | [0.0385, 0.6500] | 0.5556 | 1.0000 |
| amr_wb_compression | 0.1500 | [0.0000, 0.5052] | 0.6667 | 0.8000 |
| packet_loss_simulation | 0.4500 | [0.0455, 0.6906] | 0.6667 | 0.6000 |
