# VaaniShield Evaluation & Baseline Comparison Report

## 1. Overall Performance Comparison (In-Domain Test Set)

| Model | EER (Equal Error Rate) | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Baseline A (Generic SSL) | 0.0990 | [0.0179, 0.2600] | 0.7778 | 0.9130 |
| Baseline B (Indian Accent Mel+LFCC) | 0.0990 | [0.0192, 0.2416] | 0.8750 | 0.9130 |
| **VaaniShield (Boundary + SNAP Fusion)** | 0.0805 | [0.0172, 0.1627] | 0.9130 | 0.9130 |

## 2. Generalization to Unseen Synthesizer (Held-out OpenVoice Set)

| Model | Generalization EER | Precision | Recall |
| :--- | :--- | :--- | :--- |
| Baseline A (Generic SSL) | 0.0000 | 1.0000 | 0.9500 |
| Baseline B (Indian Accent Mel+LFCC) | 0.0000 | 1.0000 | 0.9000 |
| **VaaniShield (Boundary + SNAP Fusion)** | 0.0000 | 1.0000 | 0.8200 |

## 3. Performance Breakdown by Telecom Noise Condition (VaaniShield)

| Noise Condition | EER | Precision | Recall |
| :--- | :--- | :--- | :--- |
| packet_loss_simulation | 0.1250 | 1.0000 | 0.7500 |
| codec_compression | 0.0000 | 1.0000 | 1.0000 |
| background_noise_injection | 0.1250 | 1.0000 | 0.7500 |
| clean | 0.0714 | 0.7778 | 1.0000 |
| amr_wb_compression | 0.0000 | 1.0000 | 1.0000 |
