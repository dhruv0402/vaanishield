# VaaniShield Evaluation & Baseline Comparison Report

**Dataset Partition**: Train = `200` | Validation (`val`) = `50` | In-Domain Test (`test`) = `100` | Held-Out Synthesizer (`heldout`) = `50`

## 1. Validation Split Performance (Calibration & Hyperparameter Selection Set)

> **Verdict**: VaaniShield statistically outperforms all baselines (EER 0.0000 vs best baseline 0.2400, p < 0.05).

| Model | Validation EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Baseline A (Generic SSL) | 0.2400 | [0.1795, 0.5593] | 0.5714 | 0.6400 |
| Baseline B (Indian Accent Mel+LFCC) | 0.3400 | [0.1410, 0.4725] | 0.5000 | 0.8000 |
| **VaaniShield (Boundary + SNAP Fusion)** | 0.0000 | [0.0000, 0.0000] | 1.0000 | 0.7200 |

## 2. Overall Performance Comparison (In-Domain Test Set)

> **Verdict**: VaaniShield statistically outperforms all baselines (EER 0.1400 vs best baseline 0.3900, p < 0.05).

| Model | EER (Equal Error Rate) | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Baseline A (Generic SSL) | 0.3900 | [0.2586, 0.5201] | 0.6596 | 0.6200 |
| Baseline B (Indian Accent Mel+LFCC) | 0.4800 | [0.2881, 0.5700] | 0.5690 | 0.6600 |
| **VaaniShield (Boundary + SNAP Fusion)** | 0.1400 | [0.0200, 0.3001] | 0.8837 | 0.7600 |

## 3. Generalization to Unseen Synthesizer (Held-Out Set)

> **Verdict**: VaaniShield statistically outperforms all baselines (EER 0.0200 vs best baseline 0.2200, p < 0.05).

| Model | Generalization EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Baseline A (Generic SSL) | 0.3200 | [0.1000, 0.4416] | 0.6897 | 0.8000 |
| Baseline B (Indian Accent Mel+LFCC) | 0.2200 | [0.0990, 0.4396] | 0.6250 | 0.8000 |
| **VaaniShield (Boundary + SNAP Fusion)** | 0.0200 | [0.0000, 0.1796] | 0.8621 | 1.0000 |

## 4. Performance Breakdown by Telecom Noise Condition (VaaniShield Test Split)

| Noise Condition | EER | 95% Confidence Interval | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| background_noise_injection | 0.3500 | [0.0455, 0.6319] | 0.6000 | 0.3000 |
| packet_loss_simulation | 0.1000 | [0.0000, 0.3683] | 1.0000 | 0.8000 |
| amr_wb_compression | 0.0500 | [0.0000, 0.3586] | 0.9000 | 0.9000 |
| clean | 0.0500 | [0.0000, 0.3586] | 0.9000 | 0.9000 |
| codec_compression | 0.0500 | [0.0000, 0.3352] | 0.9000 | 0.9000 |
