# VaaniShield Evaluation & Baseline Comparison Report

## 1. Overall Performance Comparison (In-Domain Test Set)

> **Verdict**: VaaniShield demonstrates an EER of 0.2000 compared to best baseline 0.0600; baseline performance is higher on this configuration.

| Model | EER (Equal Error Rate) | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Baseline A (Generic SSL) | 0.0600 | [0.0000, 0.2000] | 1.0000 | 0.7200 |
| Baseline B (Indian Accent Mel+LFCC) | 0.2400 | [0.0582, 0.3233] | 0.8333 | 0.8000 |
| **VaaniShield (Boundary + SNAP Fusion)** | 0.2000 | [0.0862, 0.3237] | 0.7778 | 0.8400 |

## 2. Generalization to Unseen Synthesizer (Held-Out Set)

> **Verdict**: VaaniShield achieves comparable performance to baselines (EER 0.1800 [CI: 0.0714, 0.2915] vs best baseline 0.1400); differences are within statistical margin of error.

| Model | Generalization EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Baseline A (Generic SSL) | 0.1400 | [0.0584, 0.3456] | 0.8214 | 0.9200 |
| Baseline B (Indian Accent Mel+LFCC) | 0.2200 | [0.0446, 0.3253] | 0.8333 | 0.8000 |
| **VaaniShield (Boundary + SNAP Fusion)** | 0.1800 | [0.0714, 0.2915] | 0.7857 | 0.8800 |

## 3. Performance Breakdown by Telecom Noise Condition (VaaniShield)

| Noise Condition | EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| background_noise_injection | 0.3000 | [0.0000, 0.8333] | 0.6250 | 1.0000 |
| codec_compression | 0.2000 | [0.0000, 0.5424] | 1.0000 | 0.6000 |
| amr_wb_compression | 0.0000 | [0.0000, 0.0000] | 1.0000 | 1.0000 |
| packet_loss_simulation | 0.2000 | [0.0000, 0.5495] | 1.0000 | 0.6000 |
| clean | 0.3000 | [0.0000, 0.6905] | 0.6250 | 1.0000 |
