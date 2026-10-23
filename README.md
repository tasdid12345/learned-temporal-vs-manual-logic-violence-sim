# Violence detection: temporal features + LightGBM vs. hand rules + MLP sequence head

## Motivation
The paper "Video violence detection using pre-trained VGG19 combined with manual logic, LSTM layers and Bi-LSTM layers" (2026) detects violent clips by combining CNN frame features, hand-set logic and recurrent layers. This experiment asks whether learning the clip decision from temporal summary features with a gradient-boosted model can replace the hand rules and recurrent stack, especially when the test clips come from cameras never seen in training.

## Question
Can a class-balanced LightGBM on learned temporal summary features of compact frame embeddings beat a hand-rule plus recurrent-style sequence baseline at detecting violent clips on held-out clips from an unseen simulated camera?

## Dataset
Seeded NumPy simulation (no download): 30-frame clips of 128-dim frame embeddings, a stand-in for VGG19 features. Violent clips contain a burst of high frame-to-frame change and a small appearance shift. Each camera has its own background offset and noise level; the test split consists of held-out cameras. This is **not** the paper's dataset, and the baseline is a stand-in, not the paper's exact method.

## Baseline vs. improved
- **Baseline:** per-frame sklearn MLP scorer, a fixed hand rule (at least k frames above a score threshold), and an MLP on the concatenated frame scores in place of the LSTM/Bi-LSTM (sklearn has no recurrent layers).
- **Improved:** PCA to 16 dims, temporal features (frame differences, max, std, burst length, per-camera normalisation) and a class-balanced LightGBM tuned with camera-grouped cross-validation on the training set.

Main metric: violent-class F1 on the held-out-camera test clips; secondary: ROC-AUC and inference time per clip. Seeds 42, 43, 44.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Results (latest run)

On simulated held-out-camera clips (300 test clips), a standard frame-scorer with hand-set rules and an MLP sequence head reached violent-class F1 0.212, versus 0.953 for class-balanced LightGBM on temporal features; the synthetic data and per-camera normalisation caveat this.

| Metric | Value |
|---|---|
| baseline_score | 0.2123 |
| improved_score | 0.9531 |
| improvement | 0.7407 |
| baseline_score_std | 0.05323 |
| improved_score_std | 0.005238 |
| baseline_f1 | 0.2123 |
| improved_f1 | 0.9531 |
| baseline_auc | 0.5766 |
| improved_auc | 0.997 |
| baseline_auc_std | 0.01284 |
| improved_auc_std | 0.0001936 |
| baseline_ms_per_clip | 0.01002 |
| improved_ms_per_clip | 0.01667 |
| hand_rule_only_f1 | 0.4555 |
| test_violent_fraction | 0.3033 |

Run on 2026-10-23. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Synthetic data: results reflect the simulator's assumptions (burst + small shift, camera offsets), not real video. Baseline hyper-parameters are fixed by hand while LightGBM gets a small CV grid. The improved model uses per-camera normalisation computed from the unlabeled clips of each camera (including test cameras), which the baseline does not get; this is part of the method but advantages it. Decision threshold 0.5 for both. Test set used once per model/seed; frame scores for the baseline head are out-of-fold by camera.

## Reference

This project is motivated by the following paper:

> P Negre, RS Alonso, J Prieto, O García (2026). *Video violence detection using pre-trained VGG19 combined with manual logic, LSTM layers and Bi-LSTM layers*. Applied Intelligence. https://link.springer.com/article/10.1007/s10489-026-07122-3

Paper author: Prof. Javier Prieto Tejedor, University of Salamanca (javierp@usal.es)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
