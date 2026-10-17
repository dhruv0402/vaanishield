# VaaniShield Evaluation & Baseline Comparison Report

**Dataset Partition**: Train = `200` | Validation (`val`) = `50` | In-Domain Test (`test`) = `100` | Held-Out Synthesizer (`heldout`) = `50`

## 1. Validation Split Performance (Calibration & Hyperparameter Selection Set)

> **Verdict**: VaaniShield achieves comparable performance to baselines (EER 0.0600 [CI: 0.0000, 0.2003] vs best baseline 0.2000); differences are within statistical margin of error.

| Model | Validation EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Baseline A (Generic SSL) | 0.3200 | [0.1800, 0.4799] | 0.6667 | 0.7200 |
| Baseline B (Indian Accent Mel+LFCC) | 0.2000 | [0.0994, 0.3870] | 0.7692 | 0.8000 |
| **VaaniShield (Boundary + SNAP Fusion)** | 0.0600 | [0.0000, 0.2003] | 0.6944 | 1.0000 |

## 2. Overall Performance Comparison (In-Domain Test Set)

> **Verdict**: VaaniShield achieves comparable performance to baselines (EER 0.1500 [CI: 0.1200, 0.2400] vs best baseline 0.1700); differences are within statistical margin of error.

| Model | EER (Equal Error Rate) | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Baseline A (Generic SSL) | 0.1900 | [0.1000, 0.2400] | 0.7213 | 0.8800 |
| Baseline B (Indian Accent Mel+LFCC) | 0.1700 | [0.1600, 0.1800] | 0.7692 | 0.8000 |
| **VaaniShield (Boundary + SNAP Fusion)** | 0.1500 | [0.1200, 0.2400] | 0.9231 | 0.7200 |

## 3. Generalization to Unseen Synthesizer (Held-Out Set)

> **Verdict**: VaaniShield achieves comparable performance to baselines (EER 0.1200 [CI: 0.0178, 0.2431] vs best baseline 0.1600); differences are within statistical margin of error.

| Model | Generalization EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Baseline A (Generic SSL) | 0.3200 | [0.1795, 0.4628] | 0.6842 | 0.5200 |
| Baseline B (Indian Accent Mel+LFCC) | 0.1600 | [0.0750, 0.3782] | 0.8333 | 0.8000 |
| **VaaniShield (Boundary + SNAP Fusion)** | 0.1200 | [0.0178, 0.2431] | 0.7667 | 0.9200 |

## 4. Performance Breakdown by Telecom Noise Condition (VaaniShield Test Split)

| Noise Condition | EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| amr_wb_compression | 0.0500 | [0.0000, 0.0500] | 1.0000 | 0.9000 |
| codec_compression | 0.0500 | [0.0000, 0.0500] | 1.0000 | 0.6000 |
| background_noise_injection | 0.2500 | [0.2000, 0.6000] | 0.8571 | 0.6000 |
| clean | 0.2500 | [0.0000, 0.2500] | 0.8000 | 0.8000 |
| packet_loss_simulation | 0.0500 | [0.0000, 0.2000] | 1.0000 | 0.7000 |
