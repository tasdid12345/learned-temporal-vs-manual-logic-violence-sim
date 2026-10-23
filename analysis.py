"""Violence detection on simulated frame-embedding clips: hand-rule + MLP sequence head vs. class-balanced LightGBM on temporal features."""
import json, os, sys, time
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = [42, 43, 44]
N_CAM, PER_CAM, T, D, L = 40, 30, 30, 128, 20
THR, K = 0.5, 10  # fixed hand-set rule: >=K frames with score > THR


def simulate(seed=42):
    """Clips = T frames of D-dim embeddings; each camera has its own background offset and noise."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(D, L)) / np.sqrt(L)  # fixed latent->embedding map (VGG stand-in)
    u = rng.normal(size=L); u /= np.linalg.norm(u)  # appearance-shift direction
    X, y, cam = [], [], []
    for c in range(N_CAM):
        bg, sig = rng.normal(0, 1.5, L), rng.uniform(0.3, 0.9)
        for _ in range(PER_CAM):
            v = int(rng.random() < 0.3)
            z = bg + rng.normal(0, 0.5, L) + sig * rng.normal(size=(T, L))
            amp, n = 0.0, 0
            if v:
                z += 0.6 * u  # small appearance shift
                amp, n = 0.8, int(rng.integers(6, 15))  # violent burst
            elif rng.random() < 0.25:
                amp, n = 0.4, int(rng.integers(3, 7))  # benign hard-negative motion
            if n:
                s = int(rng.integers(0, T - n + 1))
                z[s:s + n] += amp * rng.normal(size=(n, L))
            X.append(z @ W.T + 0.1 * rng.normal(size=(T, D))); y.append(v); cam.append(c)
    return np.array(X, dtype=np.float32), np.array(y), np.array(cam)


def evaluate(y, p):
    """Shared metric function: violent-class F1 at 0.5, ROC-AUC."""
    return f1_score(y, p >= 0.5), roc_auc_score(y, p)


# ---------------- baseline: frame scorer + hand rule + MLP head ----------------
def fit_scorer(X, y, seed):
    m = make_pipeline(StandardScaler(), MLPClassifier((64,), max_iter=50, random_state=seed))
    return m.fit(X.reshape(-1, D), np.repeat(y, T))


def frame_scores(m, X):
    return m.predict_proba(X.reshape(-1, D))[:, 1].reshape(len(X), T)


def rule_feats(S):
    return np.c_[S, ((S > THR).sum(1) >= K).astype(float)]


def fit_baseline(Xtr, ytr, camtr, seed):
    oof = np.zeros((len(Xtr), T))  # out-of-fold frame scores (by camera) to train the head honestly
    for a, b in GroupKFold(3).split(Xtr, ytr, camtr):
        oof[b] = frame_scores(fit_scorer(Xtr[a], ytr[a], seed), Xtr[b])
    head = MLPClassifier((32,), max_iter=500, random_state=seed).fit(rule_feats(oof), ytr)
    return fit_scorer(Xtr, ytr, seed), head


def predict_baseline(model, X, cam=None):
    scorer, head = model
    S = frame_scores(scorer, X)
    return head.predict_proba(rule_feats(S))[:, 1], rule_feats(S)[:, -1]


# ---------------- improved: PCA + temporal features + LightGBM ----------------
def longest_run(m):
    best = cur = np.zeros(len(m), int)
    for t in range(m.shape[1]):
        cur = np.where(m[:, t], cur + 1, 0); best = np.maximum(best, cur)
    return best


def temporal_feats(Z, cam):
    """Z: (n,T,16) PCA frames. Per-camera normalisation uses only that camera's unlabeled clips."""
    d = np.diff(Z, axis=1); dn = np.linalg.norm(d, axis=2); ad = np.abs(d).mean(1)
    mean_c, ad_n, r = np.zeros((len(Z), Z.shape[2])), np.zeros_like(ad), np.zeros_like(dn)
    for c in np.unique(cam):
        i = cam == c
        mean_c[i] = Z[i].mean(1) - Z[i].mean((0, 1))  # remove camera background offset
        ad_n[i] = ad[i] / (np.median(ad[i], 0) + 1e-6)
        r[i] = dn[i] / (np.median(dn[i]) + 1e-6)  # change relative to camera's typical change
    roll = np.stack([r[:, s:s + 5].mean(1) for s in range(r.shape[1] - 4)], 1)
    sc = np.c_[r.max(1), r.mean(1), r.std(1), np.percentile(r, 90, axis=1), roll.max(1),
               (r > 1.3).sum(1), longest_run(r > 1.3), np.linalg.norm(mean_c, axis=1)]
    return np.c_[mean_c, ad_n, sc]


def fit_improved(Xtr, ytr, camtr, seed):
    pca = PCA(16, random_state=42).fit(Xtr.reshape(-1, D))
    F = temporal_feats(pca.transform(Xtr.reshape(-1, D)).reshape(len(Xtr), T, 16), camtr)
    base = LGBMClassifier(class_weight="balanced", subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          random_state=seed, n_jobs=1, verbose=-1)
    grid = {"num_leaves": [7, 15], "n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}
    gs = GridSearchCV(base, grid, scoring="f1", cv=GroupKFold(3), n_jobs=1)  # CV on train only
    gs.fit(F, ytr, groups=camtr)
    return pca, gs.best_estimator_, gs.best_params_


def predict_improved(model, X, cam):
    pca, clf, _ = model
    F = temporal_feats(pca.transform(X.reshape(-1, D)).reshape(len(X), T, 16), cam)
    return clf.predict_proba(F)[:, 1], None


def main():
    X, y, cam = simulate(42)
    print(f"Simulated {len(X)} clips from {N_CAM} cameras, violent rate {y.mean():.2f}", flush=True)
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=42).split(X, y, cam))  # held-out cameras
    Xtr, ytr, ctr, Xte, yte, cte = X[tr], y[tr], cam[tr], X[te], y[te], cam[te]
    print(f"train {len(tr)} clips ({len(set(ctr))} cams), test {len(te)} clips ({len(set(cte))} unseen cams)", flush=True)
    res = {k: [] for k in ["b_f1", "b_auc", "b_ms", "r_f1", "i_f1", "i_auc", "i_ms"]}
    for s in SEEDS:
        mb = fit_baseline(Xtr, ytr, ctr, s)
        t0 = time.perf_counter(); pb, flag = predict_baseline(mb, Xte, cte); tb = time.perf_counter() - t0
        f, a = evaluate(yte, pb)
        res["b_f1"].append(f); res["b_auc"].append(a); res["b_ms"].append(1000 * tb / len(Xte))
        res["r_f1"].append(f1_score(yte, flag))
        mi = fit_improved(Xtr, ytr, ctr, s)
        t0 = time.perf_counter(); pi, _ = predict_improved(mi, Xte, cte); ti = time.perf_counter() - t0
        f2, a2 = evaluate(yte, pi)
        res["i_f1"].append(f2); res["i_auc"].append(a2); res["i_ms"].append(1000 * ti / len(Xte))
        print(f"seed {s}: baseline F1 {f:.3f} AUC {a:.3f} | improved F1 {f2:.3f} AUC {a2:.3f} | {mi[2]}", flush=True)
    m = {k: float(np.mean(v)) for k, v in res.items()}
    if not all(np.isfinite(list(m.values()))) or yte.sum() == 0 or m["i_f1"] == 0 and m["b_f1"] == 0:
        sys.exit("No meaningful metrics")
    metrics = {
        "baseline_score": m["b_f1"], "improved_score": m["i_f1"], "improvement": m["i_f1"] - m["b_f1"],
        "baseline_score_std": float(np.std(res["b_f1"])), "improved_score_std": float(np.std(res["i_f1"])),
        "baseline_f1": m["b_f1"], "improved_f1": m["i_f1"], "baseline_auc": m["b_auc"], "improved_auc": m["i_auc"],
        "baseline_auc_std": float(np.std(res["b_auc"])), "improved_auc_std": float(np.std(res["i_auc"])),
        "baseline_ms_per_clip": m["b_ms"], "improved_ms_per_clip": m["i_ms"],
        "hand_rule_only_f1": m["r_f1"], "test_violent_fraction": float(yte.mean()),
    }
    out = {
        "question": "Can a class-balanced LightGBM on learned temporal summary features of compact frame embeddings beat a hand-rule plus recurrent-style sequence baseline at detecting violent clips on held-out clips from an unseen simulated camera?",
        "method": "Simulated 30-frame clips of 128-d frame embeddings (VGG19 stand-in) from 40 cameras with own background offset/noise; test = 25% held-out cameras. Baseline: per-frame MLP scorer + fixed hand rule + MLP head on frame scores. Improved: PCA(16) + temporal/per-camera-normalised features + CV-tuned class-balanced LightGBM.",
        "dataset": {"name": "Simulated camera frame-embedding clips (seeded NumPy)", "source_url": "none (generated in analysis.py)",
                    "n_rows": int(len(X)), "n_train": int(len(tr)), "n_test": int(len(te)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "Violent-class F1 on held-out-camera test clips (mean of seeds 42-44)",
        "higher_is_better": True,
        "baseline": "Per-frame sklearn MLP scorer (clip label on every frame), fixed hand rule (>=10 frames with score>0.5), and an MLP head on the 30 concatenated frame scores plus rule flag (stand-in for LSTM/Bi-LSTM).",
        "improved": "Replaces the frame scorer/rules/MLP head by PCA(16) embeddings, frame-difference, max/std/burst-length and per-camera-normalised features, fed to a class-balanced LightGBM tuned by grouped-by-camera CV on training data.",
        "metrics": metrics,
        "notes": "Synthetic data: results reflect the simulator's assumptions (burst + small shift, camera offsets), not real video. Baseline hyper-parameters are fixed by hand while LightGBM gets a small CV grid. The improved model uses per-camera normalisation computed from the unlabeled clips of each camera (including test cameras), which the baseline does not get; this is part of the method but advantages it. Decision threshold 0.5 for both. Test set used once per model/seed; frame scores for the baseline head are out-of-fold by camera.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        os.makedirs("figures", exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 3.5)); w = 0.35; xs = np.arange(2)
        ax.bar(xs - w / 2, [m["b_f1"], m["b_auc"]], w, label="baseline")
        ax.bar(xs + w / 2, [m["i_f1"], m["i_auc"]], w, label="improved")
        ax.set_xticks(xs, ["F1 (violent)", "ROC-AUC"]); ax.set_ylim(0, 1.05); ax.legend(); ax.set_title("Held-out cameras")
        fig.tight_layout(); fig.savefig("figures/comparison.png", dpi=120)
    except Exception as e:  # chart is optional
        print("plot skipped:", e)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
