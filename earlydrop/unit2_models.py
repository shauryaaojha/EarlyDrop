"""Unit 2: linear models for regression and classification, incl. two from-scratch Bayesian models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import BayesianRidge, HuberRegressor, LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config as C
from .plots import MUTED, SERIES, plt, save


# ---------------------------------------------------------------- from-scratch models
class BayesianLinearRegression:
    """Bishop §3.3–3.5: Gaussian prior N(0, α⁻¹I), noise precision β, evidence maximisation."""

    def fit(self, X, y, iters: int = 100):
        Phi = np.c_[np.ones(len(X)), X]
        N, M = Phi.shape
        alpha, beta = 1.0, 1.0
        eig = np.linalg.eigvalsh(Phi.T @ Phi)
        for _ in range(iters):
            S_inv = alpha * np.eye(M) + beta * Phi.T @ Phi
            S = np.linalg.inv(S_inv)
            m = beta * S @ Phi.T @ y
            lam = beta * eig
            gamma = float(np.sum(lam / (lam + alpha)))           # effective number of parameters
            alpha_new = gamma / float(m @ m)
            beta_new = (N - gamma) / float(np.sum((y - Phi @ m) ** 2))
            done = abs(alpha_new - alpha) < 1e-8 and abs(beta_new - beta) < 1e-8
            alpha, beta = alpha_new, beta_new
            if done:
                break
        self.alpha_, self.beta_, self.m_, self.S_, self.gamma_ = alpha, beta, m, S, gamma
        return self

    def predict(self, X, return_std: bool = False):
        Phi = np.c_[np.ones(len(X)), X]
        mean = Phi @ self.m_
        if not return_std:
            return mean
        var = 1 / self.beta_ + np.einsum("ij,jk,ik->i", Phi, self.S_, Phi)
        return mean, np.sqrt(var)


class BayesianLogisticRegression(ClassifierMixin, BaseEstimator):
    """Bishop §4.5: Gaussian prior, MAP via Newton–Raphson (IRLS), Laplace posterior,
    probit approximation to the predictive distribution."""

    def __init__(self, alpha: float = 1.0, max_iter: int = 100):
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, X, y):
        Phi = np.c_[np.ones(len(X)), X]
        w = np.zeros(Phi.shape[1])
        A = self.alpha * np.eye(Phi.shape[1])
        for _ in range(self.max_iter):
            p = expit(Phi @ w)
            grad = Phi.T @ (p - y) + A @ w
            H = Phi.T @ (Phi * (p * (1 - p))[:, None]) + A
            step = np.linalg.solve(H, grad)
            w -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        p = expit(Phi @ w)
        H = Phi.T @ (Phi * (p * (1 - p))[:, None]) + A
        self.w_map_, self.S_N_ = w, np.linalg.inv(H)  # Laplace: posterior ≈ N(w_MAP, H⁻¹)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        Phi = np.c_[np.ones(len(X)), X]
        mu_a = Phi @ self.w_map_
        var_a = np.einsum("ij,jk,ik->i", Phi, self.S_N_, Phi)
        kappa = 1 / np.sqrt(1 + np.pi * var_a / 8)
        p1 = expit(kappa * mu_a)
        return np.c_[1 - p1, p1]

    def predict_with_uncertainty(self, X, n_samples: int = 2000, seed: int = 0):
        """Monte-Carlo over the Laplace posterior: mean probability and a 90 % band."""
        rng = np.random.default_rng(seed)
        Phi = np.c_[np.ones(len(X)), X]
        W = rng.multivariate_normal(self.w_map_, self.S_N_, size=n_samples)
        P = expit(Phi @ W.T)
        return P.mean(1), np.quantile(P, 0.05, axis=1), np.quantile(P, 0.95, axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------- helpers
def split(df: pd.DataFrame, seed: int = C.SEED):
    X = df[C.FEATURES].to_numpy(float)
    return train_test_split(X, df[C.TARGET].to_numpy(), df[C.REG_TARGET].to_numpy(),
                            test_size=0.25, stratify=df[C.TARGET], random_state=seed)


def classifiers() -> dict:
    sc = StandardScaler
    return {
        "Logistic Regression": make_pipeline(sc(), LogisticRegression(max_iter=2000)),
        "Bayesian Logistic (Laplace)": make_pipeline(sc(), BayesianLogisticRegression(alpha=1.0)),
        "LDA (generative)": make_pipeline(sc(), LinearDiscriminantAnalysis()),
        "Gaussian Naive Bayes": make_pipeline(sc(), GaussianNB()),
        "SVM (linear)": make_pipeline(sc(), SVC(kernel="linear", C=1.0, probability=True, random_state=C.SEED)),
        "SVM (RBF)": make_pipeline(sc(), SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                                             random_state=C.SEED)),
        "SVM (polynomial)": make_pipeline(sc(), SVC(kernel="poly", degree=3, C=1.0, probability=True,
                                                    random_state=C.SEED)),
    }


def _clf_metrics(y, p, thr=0.5) -> dict:
    yhat = (p >= thr).astype(int)
    return {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat), "f1": f1_score(y, yhat), "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p)}


# ---------------------------------------------------------------- regression
def regression(df: pd.DataFrame) -> dict:
    Xtr, Xte, _, _, ytr, yte = split(df)
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

    # OLS by the normal equations (maximum likelihood under Gaussian noise)
    Phi = np.c_[np.ones(len(Ztr)), Ztr]
    w_ols = np.linalg.lstsq(Phi, ytr, rcond=None)[0]
    pred_ols = np.c_[np.ones(len(Zte)), Zte] @ w_ols
    sk_ols = LinearRegression().fit(Ztr, ytr)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Ztr, ytr)
    huber = HuberRegressor(max_iter=1000).fit(Ztr, ytr)
    blr = BayesianLinearRegression().fit(Ztr, ytr)
    blr_mean, blr_std = blr.predict(Zte, return_std=True)
    skb = BayesianRidge().fit(Ztr, ytr)

    preds = {"OLS (normal equations)": pred_ols, "Ridge": ridge.predict(Zte),
             "Huber (robust)": huber.predict(Zte), "Bayesian Linear Regression": blr_mean}
    table = [{"model": k, "r2": r2_score(yte, v), "mae": mean_absolute_error(yte, v),
              "rmse": float(np.sqrt(mean_squared_error(yte, v)))} for k, v in preds.items()]
    cover = float(np.mean(np.abs(yte - blr_mean) <= 1.96 * blr_std))

    # robustness: 8 % of training targets replaced by data-entry errors
    rng = np.random.default_rng(C.SEED)
    ybad = ytr.copy()
    bad = rng.choice(len(ybad), int(0.08 * len(ybad)), replace=False)
    ybad[bad] = rng.uniform(0, 2, len(bad))
    robust = {}
    for name, model in (("OLS", LinearRegression()), ("Huber", HuberRegressor(max_iter=1000))):
        robust[name] = {"clean_rmse": float(np.sqrt(mean_squared_error(yte, model.fit(Ztr, ytr).predict(Zte)))),
                        "corrupted_rmse": float(np.sqrt(mean_squared_error(yte, model.fit(Ztr, ybad).predict(Zte))))}

    order = np.argsort(blr_mean)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    ax = axes[0]
    xs = np.arange(len(order))
    ax.fill_between(xs, (blr_mean - 1.96 * blr_std)[order], (blr_mean + 1.96 * blr_std)[order],
                    color=SERIES[0], alpha=0.18, linewidth=0, label="95 % predictive interval")
    ax.plot(xs, blr_mean[order], color=SERIES[0], label="posterior mean")
    ax.scatter(xs, yte[order], s=9, color=SERIES[4], zorder=3, label="actual end-sem SGPA")
    ax.set_xlabel("test students (sorted by prediction)")
    ax.set_ylabel("SGPA")
    ax.set_title(f"Bayesian linear regression: {cover:.0%} of students inside the 95 % band")
    ax.legend(fontsize=7.5)
    ax = axes[1]
    names = list(robust)
    ax.bar(np.arange(2) - 0.18, [robust[n]["clean_rmse"] for n in names], 0.34, label="clean training data")
    ax.bar(np.arange(2) + 0.18, [robust[n]["corrupted_rmse"] for n in names], 0.34,
           label="8 % corrupted targets")
    ax.set_xticks(range(2), names)
    ax.set_ylabel("test RMSE")
    ax.set_title("Robust regression under data-entry errors")
    ax.legend(fontsize=7.5)
    save(fig, "u2_regression")

    return {
        "target": C.REG_TARGET, "n_train": len(ytr), "n_test": len(yte),
        "table": _rd(table),
        "ols_matches_sklearn": bool(np.allclose(w_ols[1:], sk_ols.coef_, atol=1e-6)),
        "ols_coefficients": dict(zip(["intercept"] + C.FEATURES, np.round(w_ols, 4).tolist())),
        "ridge_alpha": round(float(ridge.alpha_), 4),
        "blr": {"alpha": round(blr.alpha_, 5), "beta": round(blr.beta_, 4), "gamma_eff_params": round(blr.gamma_, 3),
                "noise_std": round(1 / np.sqrt(blr.beta_), 4), "mean_predictive_std": round(float(blr_std.mean()), 4),
                "coverage_95": round(cover, 4),
                "sklearn_bayesianridge_r2": round(r2_score(yte, skb.predict(Zte)), 4)},
        "robustness": _rd(robust),
    }


# ---------------------------------------------------------------- classification
def classification(df: pd.DataFrame) -> dict:
    Xtr, Xte, ytr, yte, _, _ = split(df)
    models = classifiers()
    test_rows, probas = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        probas[name] = p
        test_rows.append({"model": name, **_clf_metrics(yte, p)})

    X, y = df[C.FEATURES].to_numpy(float), df[C.TARGET].to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    cv_rows = []
    for name, m in classifiers().items():
        s = cross_validate(m, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc", "recall"])
        cv_rows.append({"model": name, **{f"{k}_mean": s[f"test_{k}"].mean() for k in ("accuracy", "f1", "roc_auc", "recall")},
                        **{f"{k}_std": s[f"test_{k}"].std() for k in ("accuracy", "f1", "roc_auc")}})

    # kernel hyper-parameters by grid search (RBF)
    gs = GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel="rbf")),
                      {"svc__C": [0.1, 0.3, 1, 3, 10, 30], "svc__gamma": [0.003, 0.01, 0.03, 0.1, 0.3]},
                      cv=cv, scoring="roc_auc").fit(X, y)

    # early warning favours recall: Bayesian LR at a lower threshold
    blr_p = probas["Bayesian Logistic (Laplace)"]
    thresholds = [{"threshold": t, **_clf_metrics(yte, blr_p, t)} for t in (0.5, 0.4, 0.3, 0.2)]
    best = max(test_rows, key=lambda r: r["roc_auc"])["model"]
    cm = confusion_matrix(yte, (probas[best] >= 0.5).astype(int))

    blr = models["Bayesian Logistic (Laplace)"][-1]
    w_std = np.sqrt(np.diag(blr.S_N_))
    lr = models["Logistic Regression"][-1]

    # figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for i, name in enumerate(probas):
        fpr, tpr, _ = roc_curve(yte, probas[name])
        ax.plot(fpr, tpr, lw=1.6, color=(SERIES + ["#5b6372", "#9aa3b2"])[i % 7],
                label=f"{name}  AUC {roc_auc_score(yte, probas[name]):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="#c9ced6", lw=1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves (held-out 25 %)")
    ax.legend(fontsize=7)
    ax = axes[1]
    order = np.argsort(np.abs(blr.w_map_[1:]))
    ys = np.arange(len(order))
    ax.errorbar(blr.w_map_[1:][order], ys, xerr=2 * w_std[1:][order], fmt="o", color=SERIES[1],
                ecolor="#b7a8ef", capsize=2, ms=5)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(ys, [C.FEATURES[i].replace("_", " ") for i in order], fontsize=7.5)
    ax.set_title("Bayesian LR weights: MAP ± 2σ (Laplace posterior)")
    save(fig, "u2_classification")

    return {
        "test": _rd(test_rows), "cv5": _rd(cv_rows),
        "best_by_auc": best, "confusion_matrix_best": cm.tolist(),
        "svm_rbf_grid": {"best_params": {k.split("__")[1]: v for k, v in gs.best_params_.items()},
                         "best_cv_auc": round(gs.best_score_, 4)},
        "blr_thresholds": _rd(thresholds),
        "blr_weights": [{"feature": f, "w_map": round(float(blr.w_map_[i + 1]), 4),
                         "posterior_std": round(float(w_std[i + 1]), 4),
                         "sklearn_logreg_w": round(float(lr.coef_[0][i]), 4)} for i, f in enumerate(C.FEATURES)],
    }


def run(df: pd.DataFrame) -> dict:
    return {"regression": regression(df), "classification": classification(df)}


def _rd(rows):
    if isinstance(rows, dict):
        return {k: _rd(v) for k, v in rows.items()}
    if isinstance(rows, list):
        return [_rd(r) for r in rows]
    return round(float(rows), 4) if isinstance(rows, (float, np.floating)) else rows
