"""Unit 5: combining models.

From scratch (each checked against scikit-learn):
  CART (Gini / squared error, cost-complexity pruning path from sklearn), bagging, random forest,
  AdaBoost (Bishop §14.3), gradient boosting with Newton leaf values (Friedman 2001),
  Bayesian model averaging over 64 logistic models with Laplace evidence (Bishop §4.4, §14.1).
Library: extra trees, histogram GBM, generalized additive model (B-splines + penalised logistic),
  an MLP as an adaptive-basis model, soft voting and stacking.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit, logsumexp
from scipy.stats import t as student_t
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from .plots import MUTED, SERIES, plt, save
from .unit2_models import BayesianLogisticRegression, split


# ================================================================ CART from scratch
class CART(ClassifierMixin, BaseEstimator):
    """Binary-split decision tree (Breiman et al. 1984).

    task="clf": Gini impurity, leaf value = weighted P(y = 1).
    task="reg": squared error, leaf value = weighted mean (used inside gradient boosting).
    Thresholds are mid-points between consecutive distinct values, as in sklearn."""

    def __init__(self, task="clf", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None):
        self.task, self.max_depth, self.min_samples_split = task, max_depth, min_samples_split
        self.min_samples_leaf, self.max_features, self.random_state = min_samples_leaf, max_features, random_state

    # impurity of a node holding total weight W and weighted sums S1 = Σwy, S2 = Σwy²
    def _node_cost(self, W, S1, S2):
        if self.task == "clf":
            return 2 * S1 * (W - S1) / W          # W · Gini = W · 2p(1 − p), p = S1/W
        return S2 - S1 * S1 / W                   # W · variance

    def _k_features(self, d):
        mf = self.max_features
        if mf is None:
            return d
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        return max(1, int(mf * d)) if isinstance(mf, float) else int(mf)

    def _best_split(self, X, y, w, idx, feats):
        """Exhaustive search over every candidate feature and threshold at once: sort each column,
        cumulative sums give the left/right child impurities for every split position."""
        yi, wi = y[idx], w[idx]
        Wn, S1n, S2n = wi.sum(), (wi * yi).sum(), (wi * yi * yi).sum()
        parent = self._node_cost(Wn, S1n, S2n)
        msl, n = self.min_samples_leaf, len(idx)
        Xi = X[np.ix_(idx, feats)]
        o = np.argsort(Xi, axis=0, kind="stable")
        xs, ys, ws = np.take_along_axis(Xi, o, 0), yi[o], wi[o]
        cW, c1, c2 = np.cumsum(ws, 0)[:-1], np.cumsum(ws * ys, 0)[:-1], np.cumsum(ws * ys * ys, 0)[:-1]
        pos = np.arange(1, n)[:, None]            # left child = first `pos` samples
        ok = (xs[1:] > xs[:-1] + 1e-12) & (pos >= msl) & (n - pos >= msl) & (cW > 1e-15) & (Wn - cW > 1e-15)
        if not ok.any():
            return None, None, 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            child = self._node_cost(cW, c1, c2) + self._node_cost(Wn - cW, S1n - c1, S2n - c2)
        child = np.where(ok, child, np.inf)
        j = child.argmin(0)                       # best position per feature
        gains = parent - child[j, np.arange(len(feats))]
        k = int(np.argmax(gains))                 # first feature with the largest decrease
        if not gains[k] > 1e-12:
            return None, None, 0.0
        r = j[k]
        return feats[k], (xs[r, k] + xs[r + 1, k]) / 2, float(gains[k])

    def fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X, float), np.asarray(y, float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        rng = np.random.default_rng(self.random_state)
        d = X.shape[1]
        k = self._k_features(d)
        feat, thr, left, right, val, nsamp, imp, wsum = [], [], [], [], [], [], [], []
        self.feature_importances_ = np.zeros(d)

        def new_node(idx):
            W = w[idx].sum()
            feat.append(-1), thr.append(np.nan), left.append(-1), right.append(-1)
            val.append((w[idx] * y[idx]).sum() / W), nsamp.append(len(idx)), wsum.append(W)
            imp.append(self._node_cost(W, (w[idx] * y[idx]).sum(), (w[idx] * y[idx] ** 2).sum()) / W)
            return len(feat) - 1

        stack = [(new_node(np.arange(len(y))), np.arange(len(y)), 0)]
        while stack:
            node, idx, depth = stack.pop()
            if (self.max_depth is not None and depth >= self.max_depth) or len(idx) < self.min_samples_split \
                    or len(idx) < 2 * self.min_samples_leaf or imp[node] <= 1e-12:
                continue
            feats = np.arange(d) if k >= d else rng.choice(d, k, replace=False)
            f, t, gain = self._best_split(X, y, w, idx, feats)
            if f is None:
                continue
            go = X[idx, f] <= t
            feat[node], thr[node] = int(f), float(t)
            self.feature_importances_[f] += gain
            L, R = new_node(idx[go]), new_node(idx[~go])
            left[node], right[node] = L, R
            stack += [(R, idx[~go], depth + 1), (L, idx[go], depth + 1)]
        self.feature_, self.threshold_ = np.array(feat), np.array(thr)
        self.left_, self.right_, self.value_ = np.array(left), np.array(right), np.array(val, float)
        self.n_node_samples_, self.impurity_, self.weighted_n_ = np.array(nsamp), np.array(imp), np.array(wsum)
        tot = self.feature_importances_.sum()
        if tot > 0:
            self.feature_importances_ /= tot
        self.classes_ = np.array([0, 1])
        return self

    def apply(self, X):
        X = np.asarray(X, float)
        node = np.zeros(len(X), int)
        rows = np.arange(len(X))
        while True:
            f = self.feature_[node]
            inner = f >= 0
            if not inner.any():
                return node
            go_left = X[rows, np.where(inner, f, 0)] <= np.nan_to_num(self.threshold_[node])
            node = np.where(inner, np.where(go_left, self.left_[node], self.right_[node]), node)

    def predict_value(self, X):
        return self.value_[self.apply(X)]

    def predict_proba(self, X):
        p = self.predict_value(X)
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_value(X) > 0.5).astype(int)

    @property
    def n_leaves_(self):
        return int((self.feature_ < 0).sum())

    def depth_(self):
        def rec(i):
            return 0 if self.feature_[i] < 0 else 1 + max(rec(self.left_[i]), rec(self.right_[i]))
        return rec(0)

    def to_dict(self, names, node=0, depth=3):
        d = {"n": int(self.n_node_samples_[node]), "p": round(float(self.value_[node]), 3),
             "gini": round(float(self.impurity_[node]), 3)}
        if self.feature_[node] >= 0 and depth > 0:
            d.update(feature=names[self.feature_[node]], threshold=round(float(self.threshold_[node]), 2),
                     left=self.to_dict(names, self.left_[node], depth - 1),
                     right=self.to_dict(names, self.right_[node], depth - 1))
        return d


# ================================================================ bagging + random forest from scratch
class BaggedTrees(ClassifierMixin, BaseEstimator):
    """Bootstrap aggregation of full-depth CARTs (Breiman 1996). max_features="sqrt" gives a random
    forest (Breiman 2001). Out-of-bag samples give a free error estimate as trees are added."""

    def __init__(self, n_estimators=200, max_features=None, min_samples_leaf=1, random_state=C.SEED,
                 oob_checkpoints=(1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200)):
        self.n_estimators, self.max_features, self.min_samples_leaf = n_estimators, max_features, min_samples_leaf
        self.random_state, self.oob_checkpoints = random_state, oob_checkpoints

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        n = len(y)
        rng = np.random.default_rng(self.random_state)
        self.trees_, oob_sum, oob_cnt, self.oob_curve_ = [], np.zeros(n), np.zeros(n), []
        for b in range(1, self.n_estimators + 1):
            idx = rng.integers(0, n, n)
            t = CART(max_features=self.max_features, min_samples_leaf=self.min_samples_leaf,
                     random_state=int(rng.integers(1 << 31))).fit(X[idx], y[idx])
            self.trees_.append(t)
            oob = np.setdiff1d(np.arange(n), idx)
            oob_sum[oob] += t.predict_value(X[oob])
            oob_cnt[oob] += 1
            if b in self.oob_checkpoints:
                m = oob_cnt > 0
                self.oob_curve_.append({"n_trees": b, "oob_error": float(np.mean((oob_sum[m] / oob_cnt[m] > 0.5) != y[m])),
                                        "coverage": float(m.mean())})
        m = oob_cnt > 0
        self.oob_proba_ = np.where(m, oob_sum / np.maximum(oob_cnt, 1), np.nan)
        self.oob_score_ = float(np.mean((self.oob_proba_[m] > 0.5) == y[m]))
        self.feature_importances_ = np.mean([t.feature_importances_ for t in self.trees_], axis=0)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.mean([t.predict_value(X) for t in self.trees_], axis=0)
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ================================================================ AdaBoost from scratch
class AdaBoost(ClassifierMixin, BaseEstimator):
    """Discrete AdaBoost (Freund & Schapire 1997; Bishop §14.3) with decision stumps.

    ε_m = Σ w_n I(y_m(x_n) ≠ t_n) / Σ w_n,  α_m = ln((1 − ε_m)/ε_m),  w_n ← w_n exp(α_m I(miss)).
    Prediction sign(Σ α_m y_m(x)); probability via the exponential-loss link p = σ(Σ α_m y_m(x))."""

    def __init__(self, n_estimators=200, max_depth=1, learning_rate=1.0):
        self.n_estimators, self.max_depth, self.learning_rate = n_estimators, max_depth, learning_rate

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        t = np.where(y == 1, 1.0, -1.0)
        w = np.full(len(y), 1 / len(y))
        self.stumps_, self.alphas_, self.errors_ = [], [], []
        F = np.zeros(len(y))
        self.train_curve_ = []
        for _ in range(self.n_estimators):
            h = CART(max_depth=self.max_depth).fit(X, y, sample_weight=w)
            pred = np.where(h.predict(X) == 1, 1.0, -1.0)
            miss = pred != t
            eps = float(w[miss].sum() / w.sum())
            if eps <= 0 or eps >= 0.5:
                break
            a = self.learning_rate * np.log((1 - eps) / eps)
            w = w * np.exp(a * miss)
            w /= w.sum()
            self.stumps_.append(h), self.alphas_.append(a), self.errors_.append(eps)
            F += a * pred
            self.train_curve_.append({"train_error": float(np.mean(np.sign(F) != t)),
                                      "exp_loss": float(np.mean(np.exp(-t * F / 2)))})   # E = Σ exp(−t f), f = ½ΣαH
        self.alphas_ = np.array(self.alphas_)
        self.classes_ = np.array([0, 1])
        return self

    def staged_decision(self, X):
        F = np.zeros(len(X))
        for a, h in zip(self.alphas_, self.stumps_):
            F = F + a * np.where(h.predict(X) == 1, 1.0, -1.0)
            yield F

    def decision_function(self, X):
        F = np.zeros(len(X))
        for F in self.staged_decision(X):
            pass
        return F

    def predict_proba(self, X):
        p = expit(self.decision_function(X))
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)


# ================================================================ gradient boosting from scratch
class GradientBoosting(ClassifierMixin, BaseEstimator):
    """Gradient tree boosting for the log-loss (Friedman 2001, TreeBoost).

    F_0 = log-odds of the base rate. Each round fits a regression tree to the negative gradient
    r = y − σ(F), then replaces every leaf value with one Newton step γ = Σ r / Σ p(1 − p),
    and F ← F + ν γ (shrinkage ν = learning_rate)."""

    def __init__(self, n_estimators=150, learning_rate=0.1, max_depth=2, min_samples_leaf=1):
        self.n_estimators, self.learning_rate = n_estimators, learning_rate
        self.max_depth, self.min_samples_leaf = max_depth, min_samples_leaf

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        p0 = y.mean()
        self.F0_ = float(np.log(p0 / (1 - p0)))
        F = np.full(len(y), self.F0_)
        self.trees_ = []
        for _ in range(self.n_estimators):
            p = expit(F)
            r = y - p
            tree = CART(task="reg", max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf).fit(X, r)
            leaf = tree.apply(X)
            for j in np.unique(leaf):
                m = leaf == j
                den = float(np.sum(p[m] * (1 - p[m])))
                tree.value_[j] = float(r[m].sum()) / den if den > 1e-12 else 0.0
            F = F + self.learning_rate * tree.value_[leaf]
            self.trees_.append(tree)
        self.feature_importances_ = np.mean([t.feature_importances_ for t in self.trees_], axis=0)
        self.classes_ = np.array([0, 1])
        return self

    def staged_proba(self, X):
        F = np.full(len(X), self.F0_)
        for t in self.trees_:
            F = F + self.learning_rate * t.predict_value(X)
            yield expit(F)

    def predict_proba(self, X):
        p = None
        for p in self.staged_proba(X):
            pass
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ================================================================ Bayesian model averaging (scratch)
BLOCKS = {
    "grades": ["previous_sem_sgpa", "current_cgpa", "internal_assessment_average"],
    "attendance": ["attendance_percentage"],
    "backlogs": ["active_backlogs"],
    "study habits": ["daily_study_hours", "study_consistency", "revision_frequency",
                     "assignment_completion_consistency", "late_assignment_frequency"],
    "motivation": ["academic_motivation", "overall_academic_engagement", "programme_completion_confidence"],
    "stress": ["academic_workload_difficulty", "academic_performance_concern"],
}


def laplace_log_evidence(m: BayesianLogisticRegression, X, y) -> float:
    """Bishop eq. 4.137 with prior N(0, α⁻¹I):  ln p(D|M) ≈ ln p(D|w_MAP) − α/2‖w‖² + M/2 ln α − ½ ln|A|."""
    Phi = np.c_[np.ones(len(X)), X]
    a = Phi @ m.w_map_
    ll = float(np.sum(y * a - np.logaddexp(0, a)))
    M = len(m.w_map_)
    _, logdetS = np.linalg.slogdet(m.S_N_)                     # S_N = A⁻¹  →  −½ ln|A| = +½ ln|S_N|
    return ll - 0.5 * m.alpha * float(m.w_map_ @ m.w_map_) + 0.5 * M * np.log(m.alpha) + 0.5 * logdetS


class BayesianModelAverage(ClassifierMixin, BaseEstimator):
    """BMA over every subset of the 6 feature blocks (2⁶ = 64 Bayesian logistic models):
    p(y | x, D) = Σ_M p(y | x, M, D) p(M | D),  p(M | D) ∝ p(D | M) p(M), uniform p(M)."""

    def __init__(self, alpha=1.0, features=tuple(C.FEATURES)):
        self.alpha, self.features = alpha, features

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        pos = {f: i for i, f in enumerate(self.features)}
        names = list(BLOCKS)
        self.models_ = []
        for r in range(len(names) + 1):
            for combo in itertools.combinations(names, r):
                cols = [pos[f] for b in combo for f in BLOCKS[b]]
                m = BayesianLogisticRegression(alpha=self.alpha).fit(Z[:, cols], y)
                self.models_.append({"blocks": combo, "cols": cols, "model": m,
                                     "log_evidence": laplace_log_evidence(m, Z[:, cols], y)})
        le = np.array([d["log_evidence"] for d in self.models_])
        self.posterior_ = np.exp(le - logsumexp(le))
        self.inclusion_ = {b: float(sum(p for p, d in zip(self.posterior_, self.models_) if b in d["blocks"]))
                           for b in names}
        self.classes_ = np.array([0, 1])
        return self

    def _probs(self, X):
        Z = self.scaler_.transform(np.asarray(X, float))
        return np.array([d["model"].predict_proba(Z[:, d["cols"]])[:, 1] for d in self.models_])

    def predict_proba(self, X):
        p = self.posterior_ @ self._probs(X)
        return np.c_[1 - p, p]

    def predict_top(self, X):
        return self._probs(X)[int(np.argmax(self.posterior_))]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ================================================================ model zoo
def gam():
    """Generalized additive model: f(x) = β₀ + Σ_j f_j(x_j), each f_j a cubic B-spline (5 knots),
    fitted by L2-penalised logistic regression (the penalty smooths each f_j)."""
    return make_pipeline(SplineTransformer(n_knots=5, degree=3), LogisticRegression(C=0.3, max_iter=5000))


def models() -> dict:
    sc, S = StandardScaler, C.SEED
    lr = lambda: make_pipeline(sc(), LogisticRegression(max_iter=2000))  # noqa: E731
    svm = lambda: make_pipeline(sc(), SVC(kernel="rbf", probability=True, random_state=S))  # noqa: E731
    rf = lambda: RandomForestClassifier(300, min_samples_leaf=2, random_state=S)  # noqa: E731
    gb = lambda: GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=S)  # noqa: E731
    return {
        # ---- single models (baselines)
        "Logistic Regression (Unit 2)": ("single", lr()),
        "CART (scratch, full depth)": ("single", CART()),
        "CART (pruned, ccp α by CV)": ("single", "pruned"),
        "GAM (B-spline additive)": ("single", gam()),
        "MLP (adaptive basis)": ("single", make_pipeline(sc(), MLPClassifier((16,), alpha=1.0, max_iter=3000, random_state=S))),
        # ---- ensembles
        "Bagging (scratch, 100 trees)": ("bagging", BaggedTrees(100)),
        "Random Forest (scratch, 200 trees)": ("bagging", BaggedTrees(200, max_features="sqrt", min_samples_leaf=2)),
        "Random Forest (sklearn, 300)": ("bagging", rf()),
        "Extra Trees (sklearn, 300)": ("bagging", ExtraTreesClassifier(300, min_samples_leaf=2, random_state=S)),
        "AdaBoost (scratch, 200 stumps)": ("boosting", AdaBoost(200)),
        "AdaBoost (sklearn, 200 stumps)": ("boosting", AdaBoostClassifier(n_estimators=200, random_state=S)),
        "Gradient Boosting (scratch)": ("boosting", GradientBoosting(150, 0.05, max_depth=2)),
        "Gradient Boosting (sklearn)": ("boosting", gb()),
        "HistGradientBoosting (sklearn)": ("boosting", HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3,
                                                                                     max_iter=200, random_state=S)),
        "Bayesian Model Averaging (scratch)": ("averaging", BayesianModelAverage()),
        "Soft Voting (LR + SVM + RF + GB)": ("combining", VotingClassifier(
            [("lr", lr()), ("svm", svm()), ("rf", rf()), ("gb", gb())], voting="soft")),
        "Stacking (LR meta-learner)": ("combining", StackingClassifier(
            [("lr", lr()), ("svm", svm()), ("nb", make_pipeline(sc(), GaussianNB())), ("rf", rf()), ("gb", gb())],
            final_estimator=LogisticRegression(max_iter=2000), cv=5, stack_method="predict_proba")),
    }


def pruned_tree(X, y, cv):
    """Minimal cost-complexity pruning: R_α(T) = R(T) + α|T̃|, α chosen by 5-fold CV ROC-AUC."""
    path = DecisionTreeClassifier(random_state=C.SEED).cost_complexity_pruning_path(X, y)
    alphas = np.unique(path.ccp_alphas[:-1])
    grid = GridSearchCV(DecisionTreeClassifier(random_state=C.SEED), {"ccp_alpha": alphas}, cv=cv,
                        scoring="roc_auc").fit(X, y)
    return grid.best_estimator_, float(grid.best_params_["ccp_alpha"]), path


def _metrics(y, p, thr=0.5) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    yhat = (p >= thr).astype(int)
    return {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat), "f1": f1_score(y, yhat), "roc_auc": roc_auc_score(y, p),
            "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p)}


def _corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold scores."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = np.var(d, ddof=1) * (1 / k + n_test / n_train)
    tstat = d.mean() / np.sqrt(var) if var > 0 else 0.0
    return float(d.mean()), float(tstat), float(2 * student_t.sf(abs(tstat), k - 1))


def _par(fn, args):
    """Run independent fits on every core (folds, bootstrap replicates, repeats)."""
    return Parallel(n_jobs=-1)(delayed(fn)(*a) for a in args)


def _fit_predict(est, Xa, ya, Xb):
    t0 = time.perf_counter()
    if isinstance(est, str):                        # "pruned": choose ccp α by an inner 5-fold CV
        m = pruned_tree(Xa, ya, StratifiedKFold(5, shuffle=True, random_state=C.SEED))[0]
    else:
        m = clone(est).fit(Xa, ya)
    return m.predict_proba(Xb)[:, 1], time.perf_counter() - t0


def _shrink_run(lr_, Xtr, ytr, Xte, yte):
    g = GradientBoosting(300, lr_, max_depth=2).fit(Xtr, ytr)
    ll = [log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6)) for p in g.staged_proba(Xte)]
    tr = [log_loss(ytr, np.clip(p, 1e-6, 1 - 1e-6)) for p in g.staged_proba(Xtr)]
    steps = list(range(1, 301, 3)) + [300]
    return {"learning_rate": lr_, "best_iter": int(np.argmin(ll)) + 1, "best_test_logloss": float(min(ll)),
            "final_test_logloss": float(ll[-1]),
            "curve": [{"m": m, "test": float(ll[m - 1]), "train": float(tr[m - 1])} for m in steps]}


def _bma_top(Xs, ys):
    b = BayesianModelAverage().fit(Xs, ys)
    return float(b.posterior_.max()), float(1 / np.sum(b.posterior_ ** 2))


# ================================================================ main
def run(df: pd.DataFrame) -> dict:
    X, y = df[C.FEATURES].to_numpy(float), df[C.TARGET].to_numpy()
    Xtr, Xte, ytr, yte, _, _ = split(df)
    cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    names = [f.replace("_", " ") for f in C.FEATURES]
    out: dict = {"n": len(y), "n_train": len(ytr), "n_test": len(yte), "base_rate": float(y.mean()),
                 "cv": "stratified 5-fold, seed 42 (same folds as Unit 2)"}

    # ---------------------------------------------------------- CART
    sk_tree = DecisionTreeClassifier(max_depth=4, random_state=C.SEED).fit(Xtr, ytr)
    my_tree = CART(max_depth=4).fit(Xtr, ytr)
    sk_full = DecisionTreeClassifier(random_state=C.SEED).fit(Xtr, ytr)
    my_full = CART().fit(Xtr, ytr)
    pruned, ccp_alpha, path = pruned_tree(Xtr, ytr, cv)
    depth_rows = []
    for dep in range(1, 13):
        t = DecisionTreeClassifier(max_depth=dep, random_state=C.SEED)
        cvp = cross_val_predict(t, X, y, cv=cv, method="predict_proba")[:, 1]
        t.fit(Xtr, ytr)
        depth_rows.append({"depth": dep, "train_acc": accuracy_score(ytr, t.predict(Xtr)),
                           "test_acc": accuracy_score(yte, t.predict(Xte)),
                           "cv_acc": accuracy_score(y, cvp > 0.5), "cv_auc": roc_auc_score(y, cvp),
                           "leaves": int(t.get_n_leaves())})
    show = CART(max_depth=3).fit(Xtr, ytr)
    out["cart"] = {
        "tree_depth3": show.to_dict(names, depth=3),
        "scratch_vs_sklearn_depth4": {"test_agreement": float(np.mean(my_tree.predict(Xte) == sk_tree.predict(Xte))),
                                      "max_abs_proba_diff": float(np.max(np.abs(my_tree.predict_value(Xte)
                                                                                - sk_tree.predict_proba(Xte)[:, 1]))),
                                      "root_feature": names[my_tree.feature_[0]],
                                      "root_threshold": float(my_tree.threshold_[0]),
                                      "sklearn_root_feature": names[sk_tree.tree_.feature[0]],
                                      "sklearn_root_threshold": float(sk_tree.tree_.threshold[0])},
        "scratch_vs_sklearn_full": {"test_agreement": float(np.mean(my_full.predict(Xte) == sk_full.predict(Xte))),
                                    "leaves_scratch": my_full.n_leaves_, "leaves_sklearn": int(sk_full.get_n_leaves()),
                                    "depth_scratch": my_full.depth_(), "depth_sklearn": int(sk_full.get_depth())},
        "depth_curve": depth_rows,
        "pruning": {"ccp_alpha": ccp_alpha, "leaves": int(pruned.get_n_leaves()), "depth": int(pruned.get_depth()),
                    "full_leaves": int(sk_full.get_n_leaves()),
                    "path": [{"alpha": float(a), "impurity": float(i)} for a, i in
                             zip(path.ccp_alphas[::max(1, len(path.ccp_alphas) // 25)],
                                 path.impurities[::max(1, len(path.ccp_alphas) // 25)])],
                    "test_acc": accuracy_score(yte, pruned.predict(Xte)),
                    "full_test_acc": accuracy_score(yte, sk_full.predict(Xte))},
        "importances": dict(zip(C.FEATURES, np.round(my_full.feature_importances_, 4).tolist())),
    }

    # ---------------------------------------------------------- bagging / RF
    bag, rfs, rfk = _par(lambda m: m.fit(Xtr, ytr), [(BaggedTrees(200),),
                                                     (BaggedTrees(200, max_features="sqrt", min_samples_leaf=2),),
                                                     (RandomForestClassifier(300, min_samples_leaf=2, oob_score=True,
                                                                             random_state=C.SEED),)])
    out["forest"] = {
        "oob_curve_bagging": bag.oob_curve_, "oob_curve_rf": rfs.oob_curve_,
        "oob_acc_bagging": bag.oob_score_, "oob_acc_rf": rfs.oob_score_, "oob_acc_sklearn_rf": float(rfk.oob_score_),
        "test_auc_scratch_rf": roc_auc_score(yte, rfs.predict_proba(Xte)[:, 1]),
        "test_auc_sklearn_rf": roc_auc_score(yte, rfk.predict_proba(Xte)[:, 1]),
        "test_auc_bagging": roc_auc_score(yte, bag.predict_proba(Xte)[:, 1]),
        "test_acc_rf": accuracy_score(yte, rfs.predict(Xte)), "test_acc_bagging": accuracy_score(yte, bag.predict(Xte)),
        # why random features help: average correlation between two trees' test predictions
        "tree_correlation_bagging": _mean_pair_corr([t.predict_value(Xte) for t in bag.trees_[:40]]),
        "tree_correlation_rf": _mean_pair_corr([t.predict_value(Xte) for t in rfs.trees_[:40]]),
        "importances_scratch_rf": dict(zip(C.FEATURES, np.round(rfs.feature_importances_, 4).tolist())),
        "importances_sklearn_rf": dict(zip(C.FEATURES, np.round(rfk.feature_importances_, 4).tolist())),
    }

    # ---------------------------------------------------------- AdaBoost
    ada = AdaBoost(200).fit(Xtr, ytr)
    sk_ada = AdaBoostClassifier(n_estimators=200, random_state=C.SEED).fit(Xtr, ytr)
    te_curve = [float(np.mean((F > 0) != yte)) for F in ada.staged_decision(Xte)]
    rounds = len(ada.alphas_)
    keep = sorted(set(np.unique(np.geomspace(1, rounds, 40).astype(int)).tolist() + [rounds]))
    first = ada.stumps_[0]
    out["adaboost"] = {
        "rounds": rounds,
        "curve": [{"m": m, "train_error": ada.train_curve_[m - 1]["train_error"], "test_error": te_curve[m - 1],
                   "exp_loss": ada.train_curve_[m - 1]["exp_loss"], "eps": ada.errors_[m - 1],
                   "alpha": float(ada.alphas_[m - 1])} for m in keep],
        "first_stump": {"feature": names[first.feature_[0]], "threshold": float(first.threshold_[0]),
                        "eps": ada.errors_[0], "alpha": float(ada.alphas_[0])},
        "stump_features": pd.Series([names[s.feature_[0]] for s in ada.stumps_]).value_counts().head(8).to_dict(),
        "scratch_vs_sklearn": {"test_agreement": float(np.mean(ada.predict(Xte) == sk_ada.predict(Xte))),
                               "first_stump_same": bool(first.feature_[0] == sk_ada.estimators_[0].tree_.feature[0]),
                               "decision_corr": float(np.corrcoef(ada.decision_function(Xte),
                                                                  sk_ada.decision_function(Xte))[0, 1]),
                               "test_auc_scratch": roc_auc_score(yte, ada.decision_function(Xte)),
                               "test_auc_sklearn": roc_auc_score(yte, sk_ada.decision_function(Xte))},
    }

    # ---------------------------------------------------------- gradient boosting + shrinkage
    gbm = GradientBoosting(150, 0.05, max_depth=2).fit(Xtr, ytr)
    sk_gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2, criterion="squared_error",
                                       random_state=C.SEED).fit(Xtr, ytr)
    diff = np.abs(gbm.predict_proba(Xte)[:, 1] - sk_gb.predict_proba(Xte)[:, 1])
    shrink = _par(_shrink_run, [(lr_, Xtr, ytr, Xte, yte) for lr_ in (1.0, 0.3, 0.1, 0.03)])
    out["gboost"] = {
        "scratch_vs_sklearn": {"max_abs_proba_diff": float(diff.max()), "mean_abs_proba_diff": float(diff.mean()),
                               "test_auc_scratch": roc_auc_score(yte, gbm.predict_proba(Xte)[:, 1]),
                               "test_auc_sklearn": roc_auc_score(yte, sk_gb.predict_proba(Xte)[:, 1]),
                               "base_rate_scratch": float(expit(gbm.F0_)), "base_rate_sklearn": float(sk_gb.init_.class_prior_[1])},
        "shrinkage": shrink,
        "importances": dict(zip(C.FEATURES, np.round(gbm.feature_importances_, 4).tolist())),
    }

    # ---------------------------------------------------------- BMA
    bma = BayesianModelAverage().fit(Xtr, ytr)
    order = np.argsort(-bma.posterior_)
    full_idx = next(i for i, d in enumerate(bma.models_) if len(d["blocks"]) == len(BLOCKS))
    full_p = bma._probs(Xte)[full_idx]
    pb, pt = bma.predict_proba(Xte)[:, 1], bma.predict_top(Xte)
    rng = np.random.default_rng(C.SEED)
    jobs = []
    for n_sub in (25, 50, 100, 200, len(ytr)):
        for _ in range(1 if n_sub == len(ytr) else 5):
            sub = rng.choice(len(ytr), n_sub, replace=False) if n_sub < len(ytr) else np.arange(len(ytr))
            if ytr[sub].min() != ytr[sub].max():
                jobs.append((n_sub, sub))
    tops = _par(_bma_top, [(Xtr[sub], ytr[sub]) for _, sub in jobs])
    concentration = []
    for n_sub in dict.fromkeys(n for n, _ in jobs):
        r = [t for (n, _), t in zip(jobs, tops) if n == n_sub]
        concentration.append({"n": n_sub, "top_posterior": float(np.mean([a for a, _ in r])),
                              "effective_models": float(np.mean([e for _, e in r]))})
    out["bma"] = {
        "n_models": len(bma.models_), "blocks": {k: v for k, v in BLOCKS.items()},
        "top_models": [{"blocks": list(bma.models_[i]["blocks"]), "posterior": float(bma.posterior_[i]),
                        "log_evidence": float(bma.models_[i]["log_evidence"]),
                        "n_params": len(bma.models_[i]["cols"]) + 1} for i in order[:8]],
        "full_model_posterior": float(bma.posterior_[full_idx]),
        "full_model_rank": int(np.where(order == full_idx)[0][0]) + 1,
        "inclusion": bma.inclusion_,
        "effective_models": float(1 / np.sum(bma.posterior_ ** 2)),
        "test": {"BMA": _metrics(yte, pb), "Top model only": _metrics(yte, pt), "Full model (all blocks)": _metrics(yte, full_p)},
        "concentration": concentration,
    }

    # ---------------------------------------------------------- GAM shape functions
    g = gam().fit(Xtr, ytr)
    spl, lrm = g[0], g[-1]
    nb = spl.n_features_out_ // len(C.FEATURES)
    shapes = []
    for j, f in enumerate(C.FEATURES):
        grid = np.linspace(np.percentile(X[:, j], 1), np.percentile(X[:, j], 99), 40)
        Xg = np.tile(np.median(Xtr, 0), (len(grid), 1))
        Xg[:, j] = grid
        B = spl.transform(Xg)[:, j * nb:(j + 1) * nb]
        fx = B @ lrm.coef_[0][j * nb:(j + 1) * nb]
        base = spl.transform(Xtr)[:, j * nb:(j + 1) * nb] @ lrm.coef_[0][j * nb:(j + 1) * nb]
        fx = fx - base.mean()
        shapes.append({"feature": f, "x": np.round(grid, 3).tolist(), "f": np.round(fx, 4).tolist(),
                       "range": float(fx.max() - fx.min())})
    shapes.sort(key=lambda s: -s["range"])
    out["gam"] = {"n_knots": 5, "degree": 3, "basis_per_feature": int(nb), "C": 0.3, "shapes": shapes[:6],
                  "test": _metrics(yte, g.predict_proba(Xte)[:, 1])}

    # ---------------------------------------------------------- the big comparison (5-fold CV + held-out test)
    zoo = models()
    folds = list(cv.split(X, y))
    tasks = [(name, k) for name in zoo for k in range(len(folds) + 1)]        # k = 5 → train/test split
    res = _par(_fit_predict, [(zoo[n][1], *((X[folds[k][0]], y[folds[k][0]], X[folds[k][1]]) if k < len(folds)
                                           else (Xtr, ytr, Xte))) for n, k in tasks])
    cv_rows, test_rows, probas = [], [], {}
    for name, (family, _) in zoo.items():
        mine = [r for (n, _), r in zip(tasks, res) if n == name]
        oof = np.zeros(len(y))
        for (tr, te), (p, _) in zip(folds, mine[:-1]):
            oof[te] = p
        fold_auc = [roc_auc_score(y[te], oof[te]) for _, te in folds]
        cv_rows.append({"model": name, "family": family, **_metrics(y, oof),
                        "auc_fold_mean": float(np.mean(fold_auc)), "auc_fold_std": float(np.std(fold_auc)),
                        "fit_seconds": float(np.mean([s for _, s in mine[:-1]]))})
        probas[name] = mine[-1][0]
        test_rows.append({"model": name, "family": family, **_metrics(yte, probas[name])})
    cv_rows.sort(key=lambda r: -r["auc_fold_mean"])
    out["compare_cv"] = cv_rows
    out["compare_test"] = sorted(test_rows, key=lambda r: -r["roc_auc"])
    best_ens = next(r["model"] for r in cv_rows if r["family"] != "single")
    best_single = next(r["model"] for r in cv_rows if r["family"] == "single")
    out["best_ensemble"], out["best_single"] = best_ens, best_single

    roc_keep = ["Logistic Regression (Unit 2)", "CART (scratch, full depth)", "Random Forest (sklearn, 300)",
                "Gradient Boosting (sklearn)", "Stacking (LR meta-learner)", best_ens]
    out["roc"] = {}
    for name in dict.fromkeys(roc_keep):
        fpr, tpr, _ = roc_curve(yte, probas[name])
        pick = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 60)).astype(int))
        out["roc"][name] = {"fpr": np.round(fpr[pick], 4).tolist(), "tpr": np.round(tpr[pick], 4).tolist(),
                            "auc": roc_auc_score(yte, probas[name])}
    out["thresholds_best_ensemble"] = [{"threshold": t, **_metrics(yte, probas[best_ens], t)} for t in (0.5, 0.4, 0.3, 0.2)]

    # ---------------------------------------------------------- stacking internals: base diversity + meta weights
    st = zoo["Stacking (LR meta-learner)"][1]
    st_fit = clone(st).fit(Xtr, ytr)
    base_names = [n for n, _ in st.estimators]
    oof_base = np.column_stack([cross_val_predict(clone(e), Xtr, ytr, cv=5, method="predict_proba", n_jobs=-1)[:, 1]
                                for _, e in st.estimators])
    out["stacking"] = {"bases": base_names,
                       "meta_weights": dict(zip(base_names, np.round(st_fit.final_estimator_.coef_[0], 3).tolist())),
                       "base_corr": np.round(np.corrcoef(oof_base.T), 3).tolist(),
                       "base_auc_oof": dict(zip(base_names, [round(roc_auc_score(ytr, oof_base[:, i]), 4)
                                                             for i in range(len(base_names))]))}

    # ---------------------------------------------------------- significance: repeated 5x3 CV, corrected t-test
    rep = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=C.SEED)
    sig_models = {k: zoo[k][1] for k in ("Logistic Regression (Unit 2)", "Random Forest (sklearn, 300)",
                                          "Gradient Boosting (sklearn)", "AdaBoost (sklearn, 200 stumps)",
                                          "Stacking (LR meta-learner)", "Soft Voting (LR + SVM + RF + GB)")}
    splits = list(rep.split(X, y))
    res = _par(_fit_predict, [(m, X[tr], y[tr], X[te]) for m in sig_models.values() for tr, te in splits])
    scores = {k: [roc_auc_score(y[te], res[i * len(splits) + j][0]) for j, (_, te) in enumerate(splits)]
              for i, k in enumerate(sig_models)}
    base = scores["Logistic Regression (Unit 2)"]
    n_tr, n_te = len(splits[0][0]), len(splits[0][1])
    out["significance"] = [{"model": k, "auc_mean": float(np.mean(v)), "auc_std": float(np.std(v)),
                            **dict(zip(("diff_vs_lr", "t", "p_value"), _corrected_ttest(v, base, n_tr, n_te)))}
                           for k, v in scores.items()]

    # ---------------------------------------------------------- bias-variance (Brier = bias² + variance, exact)
    bv = []
    reps = 30
    rng = np.random.default_rng(C.SEED)
    boots = [rng.integers(0, len(ytr), len(ytr)) for _ in range(reps)]
    bv_models = (("CART (full depth)", DecisionTreeClassifier(random_state=C.SEED)),
                 ("CART (depth 2)", DecisionTreeClassifier(max_depth=2, random_state=C.SEED)),
                 ("Bagging (100 trees)", BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=C.SEED)),
                 ("Random Forest (300)", RandomForestClassifier(300, min_samples_leaf=2, random_state=C.SEED)),
                 ("AdaBoost (200 stumps)", AdaBoost(200)),
                 ("Gradient Boosting", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2,
                                                                  random_state=C.SEED)),
                 ("Logistic Regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))))
    res = _par(_fit_predict, [(est, Xtr[b], ytr[b], Xte) for _, est in bv_models for b in boots])
    for i, (name, _) in enumerate(bv_models):
        P = np.array([p for p, _ in res[i * reps:(i + 1) * reps]])
        pbar = P.mean(0)
        bias2 = float(np.mean((pbar - yte) ** 2))
        var = float(np.mean(P.var(0)))
        main = (P > 0.5).mean(0) > 0.5
        bv.append({"model": name, "bias2": bias2, "variance": var, "brier": float(np.mean((P - yte) ** 2)),
                   "error_01": float(np.mean((P > 0.5) != yte)),
                   "variance_01": float(np.mean((P > 0.5) != main))})
    out["bias_variance"] = {"replicates": reps, "rows": bv}

    # ---------------------------------------------------------- importances (permutation on test, best ensemble)
    perm = permutation_importance(zoo["Gradient Boosting (sklearn)"][1].fit(Xtr, ytr), Xte, yte, scoring="roc_auc",
                                  n_repeats=20, random_state=C.SEED)
    lr_fit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    out["importance"] = [{"feature": f, "rf_impurity": float(rfk.feature_importances_[i]),
                          "gb_permutation": float(perm.importances_mean[i]),
                          "gb_permutation_std": float(perm.importances_std[i]),
                          "lr_abs_coef": float(abs(lr_fit[-1].coef_[0][i]))} for i, f in enumerate(C.FEATURES)]
    out["importance"].sort(key=lambda r: -r["gb_permutation"])

    # ---------------------------------------------------------- decision surfaces on two features (for the deck)
    f2 = ["attendance_percentage", "internal_assessment_average"]
    j2 = [C.FEATURES.index(f) for f in f2]
    X2 = X[:, j2]
    lo, hi = np.percentile(X2, 1, 0), np.percentile(X2, 99, 0)
    gx, gy = np.linspace(lo[0], hi[0], 36), np.linspace(lo[1], hi[1], 28)
    G = np.array([[a, b] for b in gy for a in gx])
    surf = {}
    for name, est in (("CART (depth 4)", CART(max_depth=4)), ("Random Forest", BaggedTrees(150, max_features=1, min_samples_leaf=3)),
                      ("AdaBoost", AdaBoost(100)), ("Gradient Boosting", GradientBoosting(150, 0.1, max_depth=2))):
        m = est.fit(X2, y)
        surf[name] = {"p": np.round(m.predict_proba(G)[:, 1], 3).tolist(),
                      "train_acc": float(accuracy_score(y, m.predict(X2)))}
    out["surfaces"] = {"features": f2, "gx": np.round(gx, 2).tolist(), "gy": np.round(gy, 2).tolist(), "models": surf,
                       "points": [[round(float(a), 1), round(float(b), 1), int(c)] for (a, b), c in zip(X2, y)]}

    _figures(out)
    return _rd(out)


def _mean_pair_corr(preds):
    P = np.array(preds)
    Cm = np.corrcoef(P)
    iu = np.triu_indices(len(P), 1)
    return float(np.nanmean(Cm[iu]))


# ================================================================ report figures
def _figures(o):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    d = pd.DataFrame(o["cart"]["depth_curve"])
    ax[0].plot(d.depth, d.train_acc, label="train")
    ax[0].plot(d.depth, d.cv_acc, label="5-fold CV")
    ax[0].plot(d.depth, d.test_acc, label="held-out test", ls="--")
    ax[0].set_xlabel("max depth")
    ax[0].set_ylabel("accuracy")
    ax[0].set_title("CART: depth vs over-fitting")
    ax[0].legend(fontsize=8)
    for key, lab, c in (("oob_curve_bagging", "bagging", SERIES[0]), ("oob_curve_rf", "random forest", SERIES[1])):
        cc = pd.DataFrame(o["forest"][key])
        ax[1].plot(cc.n_trees, cc.oob_error, marker="o", ms=3, label=lab, color=c)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("number of trees")
    ax[1].set_ylabel("out-of-bag error")
    ax[1].set_title("Bagging and random forest: OOB error")
    ax[1].legend(fontsize=8)
    a = pd.DataFrame(o["adaboost"]["curve"])
    ax[2].plot(a.m, a.train_error, label="train error")
    ax[2].plot(a.m, a.test_error, label="test error")
    ax[2].plot(a.m, a.exp_loss, label="exponential loss (bound)", ls="--", color=MUTED)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("boosting rounds")
    ax[2].set_title("AdaBoost (scratch) with decision stumps")
    ax[2].legend(fontsize=8)
    save(fig, "u5_trees_boosting")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    r = pd.DataFrame(o["compare_cv"]).iloc[::-1]
    col = {"single": "#9aa3b2", "bagging": SERIES[1], "boosting": SERIES[0], "averaging": SERIES[3], "combining": SERIES[4]}
    ax[0].barh(r.model, r.auc_fold_mean, xerr=r.auc_fold_std, color=[col[f] for f in r.family], height=0.65)
    ax[0].set_xlim(0.7, 1.0)
    ax[0].set_xlabel("5-fold CV ROC-AUC (mean ± sd)")
    ax[0].set_title("Unit 5: single models vs ensembles")
    ax[0].tick_params(axis="y", labelsize=8)
    for s in o["gboost"]["shrinkage"]:
        cc = pd.DataFrame(s["curve"])
        ax[1].plot(cc.m, cc.test, label=f"ν = {s['learning_rate']}  (best {s['best_test_logloss']:.3f} @ {s['best_iter']})")
    ax[1].set_xlabel("boosting iterations")
    ax[1].set_ylabel("test log-loss")
    ax[1].set_title("Gradient boosting: shrinkage")
    ax[1].legend(fontsize=8)
    save(fig, "u5_comparison")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    inc = o["bma"]["inclusion"]
    ax[0].barh(list(inc)[::-1], list(inc.values())[::-1], color=SERIES[3])
    ax[0].set_xlim(0, 1)
    ax[0].set_title("BMA: posterior inclusion probability")
    cc = pd.DataFrame(o["bma"]["concentration"])
    ax[1].plot(cc.n, cc.top_posterior, marker="o")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("training students")
    ax[1].set_ylabel("posterior of the top model")
    ax[1].set_title("BMA concentrates as data grows")
    for s, c in zip(o["gam"]["shapes"][:4], SERIES):
        xs = np.array(s["x"])
        ax[2].plot((xs - xs.min()) / (np.ptp(xs) or 1), s["f"], color=c, label=s["feature"].replace("_", " "))
    ax[2].set_xlabel("feature value (rescaled 0-1)")
    ax[2].set_ylabel("partial log-odds f_j(x_j)")
    ax[2].set_title("GAM shape functions")
    ax[2].legend(fontsize=7)
    save(fig, "u5_bma_gam")

    fig, ax = plt.subplots(figsize=(8, 4))
    b = pd.DataFrame(o["bias_variance"]["rows"])
    ax.barh(b.model, b.bias2, color=SERIES[0], label="bias²")
    ax.barh(b.model, b.variance, left=b.bias2, color=SERIES[2], label="variance")
    ax.set_xlabel("Brier score = bias² + variance (30 bootstrap training sets)")
    ax.set_title("Why ensembles work: variance reduction")
    ax.legend(fontsize=8)
    save(fig, "u5_bias_variance")


def _rd(x):
    if isinstance(x, dict):
        return {k: _rd(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_rd(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return round(float(x), 4) if isinstance(x, (float, np.floating)) else x
