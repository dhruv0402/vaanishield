# VaaniShield Evaluation & Baseline Comparison Report

## 1. Overall Performance Comparison (In-Domain Test Set)

> **Verdict**: VaaniShield demonstrates an EER of 0.3600 compared to best baseline 0.0600; baseline performance is higher on this configuration.

| Model | EER (Equal Error Rate) | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Baseline A (Generic SSL) | 0.0600 | [0.0000, 0.2000] | 1.0000 | 0.7200 |
| Baseline B (Indian Accent Mel+LFCC) | 0.2400 | [0.0582, 0.3233] | 0.8333 | 0.8000 |
| **VaaniShield (Boundary + SNAP Fusion)** | 0.3600 | [0.2585, 0.4536] | 0.5854 | 0.9600 |

## 2. Generalization to Unseen Synthesizer (Held-Out Set)

> **Verdict**: VaaniShield demonstrates an EER of 0.3200 compared to best baseline 0.1400; baseline performance is higher on this configuration.

| Model | Generalization EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Baseline A (Generic SSL) | 0.1400 | [0.0584, 0.3456] | 0.8214 | 0.9200 |
| Baseline B (Indian Accent Mel+LFCC) | 0.2200 | [0.0446, 0.3253] | 0.8333 | 0.8000 |
| **VaaniShield (Boundary + SNAP Fusion)** | 0.3200 | [0.2166, 0.4130] | 0.6098 | 1.0000 |

## 3. Performance Breakdown by Telecom Noise Condition (VaaniShield)

| Noise Condition | EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| amr_wb_compression | 0.4000 | [0.2000, 0.5000] | 0.5556 | 1.0000 |
| packet_loss_simulation | 0.3000 | [0.0000, 0.5000] | 0.6250 | 1.0000 |
| codec_compression | 0.3000 | [0.0000, 0.6121] | 0.6667 | 0.8000 |
| background_noise_injection | 0.4000 | [0.2000, 0.5000] | 0.5556 | 1.0000 |
| clean | 0.4000 | [0.2000, 0.5000] | 0.5556 | 1.0000 |
