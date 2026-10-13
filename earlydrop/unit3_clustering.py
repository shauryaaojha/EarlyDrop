"""Unit 3: K-Means, Gaussian mixtures (EM), hierarchical clustering, PCA, factor analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from . import config as C
from .plots import MUTED, SERIES, STATE, plt, save

K_FINAL = 3


# ---------------------------------------------------------------- from-scratch algorithms
def kmeans_scratch(Z, k, seed=C.SEED, max_iter=100):
    """Lloyd's algorithm with k-means++ seeding. Returns labels, centroids and the full trace."""
    rng = np.random.default_rng(seed)
    cent = [Z[rng.integers(len(Z))]]
    for _ in range(1, k):
        d2 = np.min(((Z[:, None, :] - np.array(cent)[None]) ** 2).sum(-1), axis=1)
        cent.append(Z[rng.choice(len(Z), p=d2 / d2.sum())])
    cent = np.array(cent)
    trace = []
    for _ in range(max_iter):
        d = ((Z[:, None, :] - cent[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        J = float(d[np.arange(len(Z)), lab].sum())                     # distortion measure
        trace.append({"centroids": cent.copy(), "labels": lab.copy(), "J": J})
        new = np.array([Z[lab == j].mean(0) for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    return lab, cent, trace


def gmm_em_scratch(Z, k, seed=C.SEED, max_iter=300, tol=1e-6, reg=1e-4):
    """EM for a full-covariance Gaussian mixture (Bishop §9.2.2), initialised from K-Means."""
    n, d = Z.shape
    lab, cent, _ = kmeans_scratch(Z, k, seed)
    pi = np.bincount(lab, minlength=k) / n
    mu = cent.copy()
    cov = np.array([np.cov(Z[lab == j].T) + reg * np.eye(d) for j in range(k)])
    lls = []
    for _ in range(max_iter):
        # E-step: responsibilities γ(z_nk)
        logp = np.column_stack([np.log(pi[j]) + multivariate_normal.logpdf(Z, mu[j], cov[j]) for j in range(k)])
        ll = float(logsumexp(logp, axis=1).sum())
        lls.append(ll)
        R = np.exp(logp - logsumexp(logp, axis=1, keepdims=True))
        # M-step
        Nk = R.sum(0)
        pi = Nk / n
        mu = (R.T @ Z) / Nk[:, None]
        cov = np.array([((R[:, j, None] * (Z - mu[j])).T @ (Z - mu[j])) / Nk[j] + reg * np.eye(d)
                        for j in range(k)])
        if len(lls) > 1 and abs(lls[-1] - lls[-2]) < tol * abs(lls[-1]):
            break
    return R.argmax(1), {"pi": pi, "mu": mu, "cov": cov, "loglik": lls}


def pca_scratch(Z):
    """Eigen-decomposition of the sample covariance matrix."""
    S = np.cov(Z.T)
    vals, vecs = np.linalg.eigh(S)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


# ---------------------------------------------------------------- helpers
def _scores(Z, lab, y, state):
    return {"silhouette": silhouette_score(Z, lab), "davies_bouldin": davies_bouldin_score(Z, lab),
            "calinski_harabasz": calinski_harabasz_score(Z, lab),
            "ari_at_risk": adjusted_rand_score(y, lab), "nmi_at_risk": normalized_mutual_info_score(y, lab),
            "ari_true_state": adjusted_rand_score(state, lab)}


def _persona_rank(lab, y, eng):
    """Old cluster ids in persona order: highest at-risk rate first, the rest by engagement
    ascending -> High risk, Coasting, Engaged achievers."""
    ks = list(np.unique(lab))
    first = max(ks, key=lambda k: y[lab == k].mean())
    rest = sorted((k for k in ks if k != first), key=lambda k: eng[lab == k].mean())
    return np.array([first] + rest)


def _order_by_risk(lab, y, eng):
    rank = _persona_rank(lab, y, eng)
    remap = np.empty(len(rank), int)
    remap[rank] = np.arange(len(rank))
    return remap[lab]


def k_selection(Z, y):
    rows = []
    for k in range(2, 9):
        km = KMeans(k, n_init=20, random_state=C.SEED).fit(Z)
        hc = AgglomerativeClustering(k, linkage="ward").fit_predict(Z)
        rows.append({"k": k, "kmeans_inertia": km.inertia_, "kmeans_silhouette": silhouette_score(Z, km.labels_),
                     "kmeans_davies_bouldin": davies_bouldin_score(Z, km.labels_),
                     "kmeans_calinski_harabasz": calinski_harabasz_score(Z, km.labels_),
                     "ward_silhouette": silhouette_score(Z, hc)})
    gmm_rows = []
    for k in range(1, 9):
        g = GaussianMixture(k, covariance_type="full", n_init=3, random_state=C.SEED, reg_covar=1e-4).fit(Z)
        gd = GaussianMixture(k, covariance_type="diag", n_init=3, random_state=C.SEED).fit(Z)
        gmm_rows.append({"k": k, "bic_full": g.bic(Z), "aic_full": g.aic(Z), "bic_diag": gd.bic(Z)})
    return rows, gmm_rows


def run(df: pd.DataFrame, outcomes: pd.DataFrame) -> dict:
    X = df[C.FEATURES].to_numpy(float)
    Z = StandardScaler().fit_transform(X)
    y = df[C.TARGET].to_numpy()
    state = df[["student_id"]].merge(outcomes, on="student_id")["true_state_current"].to_numpy()

    ksel, gsel = k_selection(Z, y)

    # ---- the three algorithms at k = 3 ------------------------------------------------
    km_lab, km_cent, trace = kmeans_scratch(Z, K_FINAL)
    sk_km = KMeans(K_FINAL, n_init=20, random_state=C.SEED).fit(Z)
    gm_lab, gm = gmm_em_scratch(Z, K_FINAL)
    sk_gm = GaussianMixture(K_FINAL, covariance_type="full", n_init=5, random_state=C.SEED, reg_covar=1e-4).fit(Z)
    methods = {
        "K-Means (scratch)": km_lab,
        "K-Means (sklearn)": sk_km.labels_,
        "GMM / EM (scratch)": gm_lab,
        "GMM (sklearn)": sk_gm.predict(Z),
        "Hierarchical: Ward": AgglomerativeClustering(K_FINAL, linkage="ward").fit_predict(Z),
        "Hierarchical: complete": AgglomerativeClustering(K_FINAL, linkage="complete").fit_predict(Z),
        "Hierarchical: average": AgglomerativeClustering(K_FINAL, linkage="average").fit_predict(Z),
        "Hierarchical: single": AgglomerativeClustering(K_FINAL, linkage="single").fit_predict(Z),
    }
    compare = [{"method": m, **_scores(Z, lab, y, state), "sizes": np.bincount(lab).tolist()}
               for m, lab in methods.items()]
    dissim = []
    for metric in ("euclidean", "manhattan", "cosine"):
        lab = AgglomerativeClustering(K_FINAL, linkage="average", metric=metric).fit_predict(Z)
        dissim.append({"metric": metric, "silhouette": silhouette_score(Z, lab, metric=metric),
                       "ari_at_risk": adjusted_rand_score(y, lab), "sizes": sorted(np.bincount(lab).tolist())})

    # ---- personas from K-Means ---------------------------------------------------------
    eng = df["overall_academic_engagement"].to_numpy()
    km_lab = _order_by_risk(km_lab, y, eng)
    personas = []
    names = ["High risk", "Coasting", "Engaged achievers"]
    for j in range(K_FINAL):
        m = km_lab == j
        personas.append({"cluster": j, "name": names[j], "size": int(m.sum()),
                         "at_risk_rate": float(y[m].mean()),
                         **{f: float(df.loc[m, f].mean()) for f in C.FEATURES}})

    # ---- PCA -----------------------------------------------------------------------
    vals, vecs = pca_scratch(Z)
    evr = vals / vals.sum()
    cum = np.cumsum(evr)
    sk_pca = PCA().fit(Z)
    P2 = Z @ vecs[:, :2]
    if np.corrcoef(P2[:, 0], X[:, 1])[0, 1] < 0:  # orient PC1 so that "right = better CGPA"
        vecs[:, 0] *= -1
        P2[:, 0] *= -1
    cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    acc_by_k, recon = [], []
    for k in range(1, len(C.FEATURES) + 1):
        Pk = Z @ vecs[:, :k]
        acc_by_k.append(float(cross_val_score(LogisticRegression(max_iter=2000), Pk, y, cv=cv).mean()))
        recon.append(float(np.mean((Z - Pk @ vecs[:, :k].T) ** 2)))

    # ---- factor analysis on the Likert block -------------------------------------------
    Lz = StandardScaler().fit_transform(df[C.LIKERT].to_numpy(float))
    fa_ll = [{"n_factors": k, "cv_loglik": float(np.mean([FactorAnalysis(k, random_state=C.SEED)
                                                           .fit(Lz[tr]).score(Lz[te])
                                                           for tr, te in cv.split(Lz, y)]))}
             for k in range(1, 9)]
    fa = FactorAnalysis(4, rotation="varimax", random_state=C.SEED).fit(Lz)
    load = fa.components_.T
    load *= np.sign(load[np.abs(load).argmax(0), np.arange(load.shape[1])])  # dominant loading positive
    factors = []
    for j in range(load.shape[1]):
        idx = np.argsort(-np.abs(load[:, j]))[:5]
        factors.append({"factor": j + 1, "top_items": [{"item": C.LIKERT[i], "loading": float(load[i, j])}
                                                       for i in idx]})

    # ---- figures -----------------------------------------------------------------------
    ks = [r["k"] for r in ksel]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    axes[0].plot(ks, [r["kmeans_inertia"] for r in ksel], marker="o")
    axes[0].set_title("Elbow: K-Means distortion J")
    axes[1].plot(ks, [r["kmeans_silhouette"] for r in ksel], marker="o", label="K-Means")
    axes[1].plot(ks, [r["ward_silhouette"] for r in ksel], marker="o", label="Ward")
    axes[1].set_title("Silhouette")
    axes[1].legend()
    axes[2].plot([r["k"] for r in gsel], [r["bic_full"] for r in gsel], marker="o", label="full covariance")
    axes[2].plot([r["k"] for r in gsel], [r["bic_diag"] for r in gsel], marker="o", label="diagonal")
    axes[2].set_title("GMM BIC (lower is better)")
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel("k")
    save(fig, "u3_k_selection")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (title, lab) in zip(axes, (("K-Means clusters", km_lab),
                                       ("GMM clusters", _order_by_risk(gm_lab, y, eng)), ("True outcome", y))):
        cols = STATE if title != "True outcome" else [STATE[2], STATE[0]]
        labels = names if title != "True outcome" else ["Not at risk", "At risk"]
        for j in np.unique(lab):
            ax.scatter(P2[lab == j, 0], P2[lab == j, 1], s=10, color=cols[j], label=labels[j], alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({evr[0]:.0%})")
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel(f"PC2 ({evr[1]:.0%})")
    save(fig, "u3_clusters_pca")

    Lk = linkage(Z, "ward")
    fig, ax = plt.subplots(figsize=(12, 3.8))
    dendrogram(Lk, truncate_mode="lastp", p=30, ax=ax, color_threshold=Lk[-K_FINAL + 1, 2],
               above_threshold_color=MUTED)
    ax.set_title("Ward dendrogram (last 30 merges)")
    ax.set_ylabel("merge distance")
    save(fig, "u3_dendrogram")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5))
    comp = np.arange(1, len(evr) + 1)
    axes[0].bar(comp, evr, color=SERIES[0])
    axes[0].plot(comp, cum, marker="o", color=SERIES[2])
    axes[0].axhline(0.9, ls="--", color=MUTED, lw=1)
    axes[0].set_title("Scree + cumulative variance")
    axes[1].plot(comp, acc_by_k, marker="o")
    axes[1].set_title("CV accuracy using first k PCs")
    axes[2].plot([r["n_factors"] for r in fa_ll], [r["cv_loglik"] for r in fa_ll], marker="o", color=SERIES[1])
    axes[2].set_title("Factor analysis: CV log-likelihood")
    for ax in axes:
        ax.set_xlabel("number of latent dimensions")
    save(fig, "u3_pca_fa")

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(gm["loglik"], marker="o", ms=3, color=SERIES[1])
    ax.set_title("EM never decreases the log-likelihood")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("log p(X | θ)")
    save(fig, "u3_em_convergence")

    # ---- deck exports: 2-D projection, k-means animation, dendrogram geometry ----------
    proj = lambda c: (c @ vecs[:, :2]).round(3).tolist()  # noqa: E731
    rank = _persona_rank(trace[-1]["labels"], y, eng)  # rank[i] = old id of persona i
    remap = np.empty(K_FINAL, int)
    remap[rank] = np.arange(K_FINAL)
    km_trace = [{"centroids": proj(t["centroids"][rank]), "J": round(t["J"], 2),
                 "labels": remap[t["labels"]].tolist()} for t in trace]
    dg = dendrogram(Lk, truncate_mode="lastp", p=24, no_plot=True)

    return {
        "k_final": K_FINAL, "k_selection": _rd(ksel), "gmm_selection": _rd(gsel),
        "compare": _rd(compare), "dissimilarity": _rd(dissim), "personas": _rd(personas),
        "scratch_vs_sklearn": {
            "kmeans_ari": round(adjusted_rand_score(methods["K-Means (scratch)"], sk_km.labels_), 4),
            "gmm_ari": round(adjusted_rand_score(gm_lab, sk_gm.predict(Z)), 4),
            "gmm_scratch_loglik": round(gm["loglik"][-1], 2),
            "gmm_sklearn_loglik": round(float(sk_gm.score(Z) * len(Z)), 2),
            "em_iterations": len(gm["loglik"]), "kmeans_iterations": len(trace),
            "pca_eigvals_match": bool(np.allclose(vals, sk_pca.explained_variance_, rtol=1e-6)),
        },
        "pca": {"eigenvalues": _rd(vals.tolist()), "explained_ratio": _rd(evr.tolist()), "cumulative": _rd(cum.tolist()),
                "k90": int(np.searchsorted(cum, 0.90) + 1), "k95": int(np.searchsorted(cum, 0.95) + 1),
                "acc_by_k": _rd(acc_by_k), "recon_mse_by_k": _rd(recon),
                "pc1_loadings": dict(zip(C.FEATURES, np.round(vecs[:, 0], 4).tolist())),
                "pc2_loadings": dict(zip(C.FEATURES, np.round(vecs[:, 1], 4).tolist()))},
        "factor_analysis": {"cv_loglik": _rd(fa_ll), "n_factors_best": int(max(fa_ll, key=lambda r: r["cv_loglik"])["n_factors"]),
                            "factors": _rd(factors)},
        "deck": {"points": [[round(float(a), 3), round(float(b), 3), int(c), int(r)]
                            for a, b, c, r in zip(P2[:, 0], P2[:, 1], km_lab, y)],
                 "kmeans_trace": km_trace,
                 "dendrogram": {"icoord": _rd(dg["icoord"], 2), "dcoord": _rd(dg["dcoord"], 2),
                                "cut": round(float(Lk[-K_FINAL + 1, 2]), 2)}},
    }


def _rd(x, n=4):
    if isinstance(x, dict):
        return {k: _rd(v, n) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_rd(v, n) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), n)
    if isinstance(x, np.integer):
        return int(x)
    return x
