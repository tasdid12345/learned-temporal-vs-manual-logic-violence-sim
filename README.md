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

## Reference

This project is motivated by the following paper:

> P Negre, RS Alonso, J Prieto, O García (2026). *Video violence detection using pre-trained VGG19 combined with manual logic, LSTM layers and Bi-LSTM layers*. Applied Intelligence. https://link.springer.com/article/10.1007/s10489-026-07122-3

Paper author: Prof. Javier Prieto Tejedor, University of Salamanca (javierp@usal.es)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
