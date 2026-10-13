"""Unit 4: Markov chains, hidden Markov models (from scratch), Viterbi, and a Kalman filter (LDS)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .plots import MUTED, SERIES, STATE, plt, save


class DiscreteHMM:
    """Categorical-emission HMM trained by Baum–Welch with scaling factors (Bishop §13.2)."""

    def __init__(self, K=3, M=3, n_restarts=10, max_iter=500, tol=1e-7, seed=C.SEED):
        self.K, self.M, self.n_restarts, self.max_iter, self.tol, self.seed = K, M, n_restarts, max_iter, tol, seed

    # ---- inference -------------------------------------------------------------------
    def forward(self, o):
        T = len(o)
        alpha, c = np.zeros((T, self.K)), np.zeros(T)
        a = self.pi * self.B[:, o[0]]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.A) * self.B[:, o[t]]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        return alpha, c          # alpha[t] = p(z_t | o_1..t); log p(O) = sum(log c)

    def backward(self, o, c):
        T = len(o)
        beta = np.ones((T, self.K))
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (self.B[:, o[t + 1]] * beta[t + 1])) / c[t + 1]
        return beta

    def loglik(self, seqs):
        return float(sum(np.log(self._forward_batch(O)[1]).sum() for O in _batches(seqs)))

    # batched versions: all sequences of the same length at once, O has shape (n, T)
    def _forward_batch(self, O):
        n, T = O.shape
        alpha, c = np.zeros((n, T, self.K)), np.zeros((n, T))
        a = self.pi[None] * self.B[:, O[:, 0]].T
        for t in range(T):
            if t:
                a = (alpha[:, t - 1] @ self.A) * self.B[:, O[:, t]].T
            c[:, t] = a.sum(1)
            alpha[:, t] = a / c[:, t, None]
        return alpha, c

    def _backward_batch(self, O, c):
        n, T = O.shape
        beta = np.ones((n, T, self.K))
        for t in range(T - 2, -1, -1):
            beta[:, t] = ((self.B[:, O[:, t + 1]].T * beta[:, t + 1]) @ self.A.T) / c[:, t + 1, None]
        return beta

    def viterbi(self, o):
        logA, logB = np.log(self.A + 1e-300), np.log(self.B + 1e-300)
        T = len(o)
        omega = np.log(self.pi + 1e-300) + logB[:, o[0]]
        back = np.zeros((T, self.K), int)
        for t in range(1, T):
            s = omega[:, None] + logA
            back[t] = s.argmax(0)
            omega = s.max(0) + logB[:, o[t]]
        path = [int(omega.argmax())]
        for t in range(T - 1, 0, -1):
            path.append(int(back[t, path[-1]]))
        return np.array(path[::-1])

    def predict_next(self, o):
        """p(o_{T+1} | o_1..T) and p(z_{T+1} | o_1..T)."""
        alpha, _ = self.forward(o)
        z_next = alpha[-1] @ self.A
        return z_next @ self.B, z_next

    # ---- learning --------------------------------------------------------------------
    def _em(self, seqs, rng):
        K, M = self.K, self.M
        self.pi = rng.dirichlet(np.ones(K))
        self.A = rng.dirichlet(np.ones(K) * 2, size=K)
        self.B = rng.dirichlet(np.ones(M) * 2, size=K)
        lls = []
        batches = _batches(seqs)
        for _ in range(self.max_iter):
            pi_acc, A_num, B_num, ll = np.zeros(K), np.zeros((K, K)), np.zeros((K, M)), 0.0
            for O in batches:
                alpha, c = self._forward_batch(O)
                beta = self._backward_batch(O, c)
                gamma = alpha * beta                                   # p(z_t | O), shape (n, T, K)
                ll += np.log(c).sum()
                pi_acc += gamma[:, 0].sum(0)
                for t in range(O.shape[1] - 1):                        # sum over n of xi(z_t, z_t+1)
                    right = self.B[:, O[:, t + 1]].T * beta[:, t + 1] / c[:, t + 1, None]
                    A_num += self.A * (alpha[:, t].T @ right)
                for m_ in range(M):
                    B_num[:, m_] += gamma[O == m_].sum(0)
            lls.append(ll)
            self.pi = pi_acc / pi_acc.sum()
            self.A = A_num / A_num.sum(1, keepdims=True)
            self.B = (B_num + 1e-10) / (B_num + 1e-10).sum(1, keepdims=True)
            if len(lls) > 1 and abs(lls[-1] - lls[-2]) < self.tol * abs(lls[-1]):
                break
        return self.loglik(seqs), lls

    def fit(self, seqs):
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_restarts):
            ll, curve = self._em(seqs, rng)
            if best is None or ll > best[0]:
                best = (ll, curve, self.pi.copy(), self.A.copy(), self.B.copy())
        self.ll_, self.curve_, self.pi, self.A, self.B = best
        order = np.argsort(self.B @ np.arange(self.M))   # Struggling -> Stable -> Strong
        self.pi, self.A, self.B = self.pi[order], self.A[np.ix_(order, order)], self.B[order]
        return self

    def n_params(self):
        return (self.K - 1) + self.K * (self.K - 1) + self.K * (self.M - 1)


class LocalLevelKalman:
    """Linear dynamical system x_t = x_{t-1} + w, y_t = x_t + v (Bishop §13.3)."""

    def __init__(self, q=0.1, r=0.3):
        self.q, self.r = q, r

    def filter(self, y, m0, p0):
        m, p, ll, means = m0, p0, 0.0, []
        for obs in y:
            p_pred = p + self.q
            s = p_pred + self.r
            ll += -0.5 * (np.log(2 * np.pi * s) + (obs - m) ** 2 / s)
            k = p_pred / s                                  # Kalman gain
            m, p = m + k * (obs - m), (1 - k) * p_pred
            means.append(m)
        return np.array(means), m, p + self.q, ll           # last: one-step-ahead mean / var

    def fit(self, seqs, m0, p0):
        grid = [(q, r) for q in np.geomspace(0.005, 2, 25) for r in np.geomspace(0.01, 2, 25)]
        scores = [(sum(LocalLevelKalman(q, r).filter(s, m0, p0)[3] for s in seqs), q, r) for q, r in grid]
        _, self.q, self.r = max(scores)
        return self


# ---------------------------------------------------------------- helpers
def _batches(seqs):
    """Group sequences by length into (n, T) integer arrays for vectorised forward-backward."""
    by_len = {}
    for o in seqs:
        by_len.setdefault(len(o), []).append(o)
    return [np.array(v) for _, v in sorted(by_len.items())]


def sequences(history: pd.DataFrame):
    g = history.sort_values(["student_id", "semester"]).groupby("student_id", sort=False)
    ids = list(g.groups)
    obs = [g.get_group(i)["band"].to_numpy() for i in ids]
    sg = [g.get_group(i)["sgpa"].to_numpy() for i in ids]
    st = [g.get_group(i)["true_state"].to_numpy() for i in ids]
    return ids, obs, sg, st


def _markov_chain(seqs, M=3):
    counts = np.ones((M, M))                                # Laplace smoothing
    for o in seqs:
        for a, b in zip(o[:-1], o[1:]):
            counts[a, b] += 1
    return counts / counts.sum(1, keepdims=True)


def _logloss(p_true):
    return float(-np.mean(np.log(np.clip(p_true, 1e-12, 1))))


def run(df: pd.DataFrame, history: pd.DataFrame) -> dict:
    truth = json.loads(C.GROUND_TRUTH_JSON.read_text())
    A_true = np.array(truth["A_true"])
    ids, obs, sg, st = sequences(history)
    N_obs = sum(len(o) for o in obs)

    # ---- model selection over the number of hidden states -------------------------------
    sel = []
    models = {}
    for K in range(2, 6):
        m = DiscreteHMM(K=K, n_restarts=8).fit(obs)
        models[K] = m
        sel.append({"K": K, "loglik": m.ll_, "n_params": m.n_params(),
                    "bic": -2 * m.ll_ + m.n_params() * np.log(N_obs), "aic": -2 * m.ll_ + 2 * m.n_params()})
    model = models[3]

    # ---- cross-check against hmmlearn ---------------------------------------------------
    X = np.concatenate(obs).reshape(-1, 1)
    lengths = [len(o) for o in obs]
    ref = hmm.CategoricalHMM(n_components=3, init_params="", params="")
    ref.startprob_, ref.transmat_, ref.emissionprob_ = model.pi, model.A, model.B
    best_ref = None
    for seed in range(3):  # hmmlearn loops over sequences in Python (~0.1 s/iter): a cross-check, kept short
        r = hmm.CategoricalHMM(n_components=3, n_iter=150, tol=1e-3, random_state=seed).fit(X, lengths)
        if best_ref is None or r.score(X, lengths) > best_ref.score(X, lengths):
            best_ref = r
    ref_order = np.argsort(best_ref.emissionprob_ @ np.arange(3))
    viterbi_ref = ref.decode(X, lengths)[1]

    # ---- Viterbi decoding vs the true hidden states ------------------------------------
    paths = [model.viterbi(o) for o in obs]
    decode_acc = float(np.mean(np.concatenate(paths) == np.concatenate(st)))
    obs_as_state_acc = float(np.mean(np.concatenate(obs) == np.concatenate(st)))
    true_counts = np.zeros((3, 3))
    for s in st:
        for a, b in zip(s[:-1], s[1:]):
            true_counts[a, b] += 1
    A_emp = true_counts / true_counts.sum(1, keepdims=True)
    B_emp = np.array([[np.mean(np.concatenate(obs)[np.concatenate(st) == k] == b) for b in range(3)] for k in range(3)])

    # ---- next-semester prediction (hold out each student's last completed semester) ------
    long = [i for i, o in enumerate(obs) if len(o) >= 4]
    train_seqs = [o[:-1] if i in set(long) else o for i, o in enumerate(obs)]
    m_tr = DiscreteHMM(K=3, n_restarts=8).fit(train_seqs)
    mc = _markov_chain(train_seqs)
    majority = int(np.bincount(np.concatenate(train_seqs)).argmax())
    y_next = np.array([obs[i][-1] for i in long])
    p_hmm = np.array([m_tr.predict_next(obs[i][:-1])[0] for i in long])
    p_mc = np.array([mc[obs[i][-2]] for i in long])
    persist = np.array([obs[i][-2] for i in long])
    rows = [
        {"model": "Majority class", "accuracy": float(np.mean(y_next == majority)), "log_loss": None},
        {"model": "Persistence (same as last)", "accuracy": float(np.mean(y_next == persist)), "log_loss": None},
        {"model": "Markov chain on grades", "accuracy": float(np.mean(y_next == p_mc.argmax(1))),
         "log_loss": _logloss(p_mc[np.arange(len(long)), y_next])},
        {"model": "Hidden Markov Model", "accuracy": float(np.mean(y_next == p_hmm.argmax(1))),
         "log_loss": _logloss(p_hmm[np.arange(len(long)), y_next])},
    ]

    # continuous next-SGPA: Kalman filter vs Gaussian HMM vs persistence
    sg_tr = [s[:-1] if i in set(long) else s for i, s in enumerate(sg)]
    m0, p0 = float(np.mean([s[0] for s in sg_tr])), float(np.var([s[0] for s in sg_tr]))
    kf = LocalLevelKalman().fit(sg_tr, m0, p0)
    y_true = np.array([sg[i][-1] for i in long])
    kf_pred = np.array([kf.filter(sg[i][:-1], m0, p0)[1] for i in long])
    g = hmm.GaussianHMM(3, n_iter=100, tol=1e-2, random_state=C.SEED, covariance_type="diag")
    g.fit(np.concatenate(sg_tr).reshape(-1, 1), [len(s) for s in sg_tr])
    g_pred = []
    for i in long:
        post = g.predict_proba(sg[i][:-1].reshape(-1, 1))[-1] @ g.transmat_
        g_pred.append(float(post @ g.means_.ravel()))
    rmse = lambda p: float(np.sqrt(np.mean((np.asarray(p) - y_true) ** 2)))  # noqa: E731
    cont = [{"model": "Persistence", "rmse": rmse([sg[i][-2] for i in long])},
            {"model": "Student's running mean (CGPA)", "rmse": rmse([sg[i][:-1].mean() for i in long])},
            {"model": "Gaussian HMM (hmmlearn)", "rmse": rmse(g_pred)},
            {"model": "Kalman filter (LDS)", "rmse": rmse(kf_pred)}]

    # ---- early warning: HMM forecast of the in-progress semester -> at_risk -------------
    pos = {sid: k for k, sid in enumerate(ids)}
    p_poor = np.array([model.predict_next(obs[pos[s]])[0][0] for s in df["student_id"]])
    p_strug = np.array([model.predict_next(obs[pos[s]])[1][0] for s in df["student_id"]])
    yr = df[C.TARGET].to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    survey = df[C.FEATURES].to_numpy(float)
    p_survey = cross_val_predict(lr, survey, yr, cv=cv, method="predict_proba")[:, 1]
    p_both = cross_val_predict(lr, np.c_[survey, p_poor, p_strug], yr, cv=cv, method="predict_proba")[:, 1]
    early = [{"source": "HMM on grade history only", "roc_auc": roc_auc_score(yr, p_poor)},
             {"source": "Survey features only (logistic)", "roc_auc": roc_auc_score(yr, p_survey)},
             {"source": "Survey + HMM forecast", "roc_auc": roc_auc_score(yr, p_both)}]

    # ---- example trajectories for the deck -----------------------------------------------
    six = [i for i, o in enumerate(obs) if len(o) == 6]
    picks = {}
    for i in six:
        s = sg[i]
        kind = ("declining" if s[-1] - s[0] < -1.5 else "recovering" if s[-1] - s[0] > 1.5 else
                "steady strong" if s.min() > 8 else "steady struggling" if s.max() < 6.5 else None)
        if kind and kind not in picks:
            picks[kind] = i
    examples = []
    for kind, i in picks.items():
        alpha, _ = model.forward(obs[i])
        nxt, znext = model.predict_next(obs[i])
        examples.append({"kind": kind, "student_id": ids[i], "sgpa": sg[i].tolist(), "band": obs[i].tolist(),
                         "viterbi": paths[i].tolist(), "true_state": st[i].tolist(),
                         "p_struggling_filtered": alpha[:, 0].round(4).tolist(),
                         "next_band_dist": nxt.round(4).tolist(), "next_state_dist": znext.round(4).tolist()})

    # ---- figures -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, M, title in ((axes[0], model.A, "Learned transition matrix A"),
                         (axes[1], A_true, "True transition matrix (generator)"),
                         (axes[2], model.B, "Learned emission matrix B")):
        ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for (r_, c_), v in np.ndenumerate(M):
            ax.text(c_, r_, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.55 else "#14181f", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_yticks(range(3), C.STATE_NAMES)
        ax.set_xticks(range(3), C.BAND_NAMES if M is model.B else C.STATE_NAMES)
    save(fig, "u4_matrices")

    fig, axes = plt.subplots(1, len(examples), figsize=(3.6 * len(examples), 3.4), sharey=True)
    for ax, ex in zip(np.atleast_1d(axes), examples):
        t = np.arange(1, len(ex["sgpa"]) + 1)
        ax.plot(t, ex["sgpa"], color=MUTED, lw=1.5, zorder=1)
        ax.scatter(t, ex["sgpa"], c=[STATE[s] for s in ex["viterbi"]], s=60, zorder=2)
        for e in C.BAND_EDGES:
            ax.axhline(e, ls=":", color="#c9ced6", lw=1)
        ax.set_title(ex["kind"])
        ax.set_xlabel("semester")
    np.atleast_1d(axes)[0].set_ylabel("SGPA (colour = Viterbi state)")
    save(fig, "u4_viterbi_examples")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    axes[0].plot([r_["K"] for r_ in sel], [r_["bic"] for r_ in sel], marker="o")
    axes[0].set_title("BIC vs number of hidden states")
    axes[0].set_xlabel("K")
    axes[1].barh([r_["model"] for r_ in rows], [r_["accuracy"] for r_ in rows], color=SERIES[0])
    axes[1].set_title("Next-semester band accuracy")
    axes[2].barh([r_["model"] for r_ in cont], [r_["rmse"] for r_ in cont], color=SERIES[1])
    axes[2].set_title("Next-semester SGPA RMSE")
    save(fig, "u4_comparison")

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(model.curve_, marker="o", ms=3, color=SERIES[3])
    ax.set_title("Baum–Welch log-likelihood per iteration")
    ax.set_xlabel("iteration")
    save(fig, "u4_baum_welch")

    return {
        "n_students": len(obs), "n_observations": N_obs, "lengths": {int(k): int(v) for k, v in zip(*np.unique(lengths, return_counts=True))},
        "selection": _rd(sel), "K_best_bic": int(min(sel, key=lambda r_: r_["bic"])["K"]),
        "pi": _rd(model.pi), "A": _rd(model.A), "B": _rd(model.B), "loglik": round(model.ll_, 3),
        "baum_welch_iterations": len(model.curve_),
        "hmmlearn": {"loglik_same_params": round(float(ref.score(X, lengths)), 3),
                     "loglik_own_fit": round(float(best_ref.score(X, lengths)), 3),
                     "A_own_fit": _rd(best_ref.transmat_[np.ix_(ref_order, ref_order)]),
                     "viterbi_agreement": float(np.mean(viterbi_ref == np.concatenate(paths)))},
        "truth": {"A_true": _rd(A_true), "A_empirical": _rd(A_emp), "B_empirical": _rd(B_emp),
                  "A_max_abs_error": round(float(np.abs(model.A - A_true).max()), 4),
                  "viterbi_state_accuracy": round(decode_acc, 4),
                  "band_equals_state_accuracy": round(obs_as_state_acc, 4)},
        "next_semester": {"n_students": len(long), "categorical": _rd(rows), "continuous": _rd(cont),
                          "kalman": {"q": round(kf.q, 4), "r": round(kf.r, 4)}},
        "early_warning": _rd(early),
        "examples": examples,
    }


def _rd(x, n=4):
    if isinstance(x, dict):
        return {(int(k) if isinstance(k, np.integer) else k): _rd(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_rd(v, n) for v in x]
    if isinstance(x, np.ndarray):
        return np.round(x, n).tolist()
    if isinstance(x, (float, np.floating)):
        return round(float(x), n)
    if isinstance(x, np.integer):
        return int(x)
    return x
