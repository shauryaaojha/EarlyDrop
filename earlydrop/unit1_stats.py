"""Unit 1: EDA, probability, Bayes' rule, (conditional) independence, polynomial curve fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from .plots import MUTED, SERIES, STATE, plt, save


def describe(df: pd.DataFrame) -> list[dict]:
    rows = []
    for f in C.FEATURES:
        x = df[f]
        rows.append({"feature": f, "mean": x.mean(), "std": x.std(), "var": x.var(), "min": x.min(),
                     "q1": x.quantile(.25), "median": x.median(), "q3": x.quantile(.75),
                     "max": x.max(), "skew": stats.skew(x)})
    return [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]


def covariance_pairs(df: pd.DataFrame) -> list[dict]:
    pairs = [("attendance_percentage", "internal_assessment_average"),
             ("daily_study_hours", "current_cgpa"),
             ("academic_motivation", "overall_academic_engagement"),
             ("active_backlogs", "current_cgpa")]
    out = []
    for a, b in pairs:
        x, y = df[a], df[b]
        cov = float(((x - x.mean()) * (y - y.mean())).sum() / (len(x) - 1))  # Cov = E[(X-mx)(Y-my)]
        r = cov / (x.std() * y.std())
        out.append({"x": a, "y": b, "cov": round(cov, 3), "r": round(r, 3),
                    "check_numpy": round(float(np.corrcoef(x, y)[0, 1]), 3)})
    return out


def probability(df: pd.DataFrame) -> dict:
    # discrete random variable: active backlogs
    pmf = df["active_backlogs"].value_counts(normalize=True).sort_index()
    ex = float((pmf.index * pmf).sum())
    var = float(((pmf.index - ex) ** 2 * pmf).sum())
    # continuous random variable: CGPA
    g = df["current_cgpa"]
    mu, sd = float(g.mean()), float(g.std())
    return {
        "backlogs_pmf": {str(int(k)): round(float(v), 4) for k, v in pmf.items()},
        "backlogs_expectation": round(ex, 4), "backlogs_variance": round(var, 4),
        "cgpa_mean": round(mu, 3), "cgpa_var": round(sd ** 2, 3),
        "cgpa_quantiles": {q: round(float(g.quantile(q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "p_cgpa_below_6_empirical": round(float((g < 6).mean()), 4),
        "p_cgpa_below_6_normal": round(float(stats.norm.cdf(6, mu, sd)), 4),
    }


def bayes(df: pd.DataFrame) -> dict:
    """H = at risk. E1 = attendance below the 75 % rule. E2 = weak internal marks (< 60 %)."""
    H = df[C.TARGET] == 1
    E1 = df["attendance_percentage"] < C.ATTENDANCE_RULE
    E2 = df["internal_assessment_average"] < 60
    p = lambda m: float(m.mean())  # noqa: E731
    cond = lambda a, b: float((a & b).sum() / b.sum()) if b.sum() else float("nan")  # noqa: E731

    prior = p(H)
    # Bayes' rule with both pieces of evidence: P(H|E) = P(E|H)P(H)/P(E)
    E = E1 & E2
    lik, ev = cond(E, H), p(E)
    posterior = lik * prior / ev
    # Sequential update assuming E1 and E2 are conditionally independent given H (naive)
    post1 = cond(E1, H) * prior / p(E1)
    l2h, l2n = cond(E2, H), cond(E2, ~H)
    post2_naive = l2h * post1 / (l2h * post1 + l2n * (1 - post1))
    return {
        "prior": round(prior, 4),
        "p_E1_given_H": round(cond(E1, H), 4), "p_E2_given_H": round(l2h, 4),
        "p_E_given_H": round(lik, 4), "p_E": round(ev, 4), "n_E": int(E.sum()),
        "posterior": round(posterior, 4), "posterior_check_direct": round(cond(H, E), 4),
        "chain": [{"stage": "Prior P(H)", "p": round(prior, 4)},
                  {"stage": "+ attendance < 75%", "p": round(post1, 4)},
                  {"stage": "+ internal marks < 60%", "p": round(post2_naive, 4)}],
        "naive_vs_exact_gap": round(post2_naive - posterior, 4),
    }


def independence(df: pd.DataFrame) -> dict:
    """Low attendance (A) and low study hours (B): marginally dependent? Given risk status?"""
    A = df["attendance_percentage"] < C.ATTENDANCE_RULE
    B = df["daily_study_hours"] <= 1.5
    H = df[C.TARGET] == 1
    res = {"p_A": A.mean(), "p_B": B.mean(), "p_AB": (A & B).mean(), "p_A_p_B": A.mean() * B.mean()}
    for name, m in (("given_at_risk", H), ("given_not_at_risk", ~H)):
        a, b = A[m], B[m]
        res[name] = {"p_AB": (a & b).mean(), "p_A_p_B": a.mean() * b.mean()}
    table = pd.crosstab(A, B)
    res["chi2_p_value"] = stats.chi2_contingency(table)[1]
    return _round(res)


def poly_fit(df: pd.DataFrame, seed: int = C.SEED) -> dict:
    """Bishop-style curve fitting: attendance -> internal marks on a small training sample."""
    rng = np.random.default_rng(seed)
    x_all = (df["attendance_percentage"].to_numpy() - 35) / 65   # scale to ~[0, 1]
    y_all = df["internal_assessment_average"].to_numpy() / 100
    idx = rng.permutation(len(df))
    tr, te = idx[:12], idx[12:]
    degrees = list(range(10))
    tr_rmse, te_rmse, coefs = [], [], {}
    for M in degrees:
        w = np.polyfit(x_all[tr], y_all[tr], M)
        tr_rmse.append(float(np.sqrt(np.mean((np.polyval(w, x_all[tr]) - y_all[tr]) ** 2))))
        te_rmse.append(float(np.sqrt(np.mean((np.polyval(w, x_all[te]) - y_all[te]) ** 2))))
        coefs[M] = w
    # regularised M = 9 (ridge on polynomial features), ln(lambda) = -8
    Phi = np.vander(x_all[tr], 10)
    lam = np.exp(-8)
    w_reg = np.linalg.solve(Phi.T @ Phi + lam * np.eye(10), Phi.T @ y_all[tr])
    reg_te = float(np.sqrt(np.mean((np.vander(x_all[te], 10) @ w_reg - y_all[te]) ** 2)))
    best = int(np.argmin(te_rmse))

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    grid = np.linspace(x_all.min(), x_all.max(), 300)
    for ax, M in zip(axes[:3], (1, 3, 9)):
        ax.scatter(x_all[te], y_all[te], s=8, color="#c9ced6", label="unseen students")
        ax.scatter(x_all[tr], y_all[tr], s=28, color=SERIES[0], zorder=3, label="12 training points")
        ax.plot(grid, np.clip(np.polyval(coefs[M], grid), -0.2, 1.2), color=SERIES[2])
        ax.set_ylim(0.2, 1.05)
        ax.set_title(f"M = {M}")
        ax.set_xlabel("attendance (scaled)")
    axes[0].set_ylabel("internal marks (scaled)")
    axes[0].legend(loc="lower right", fontsize=7.5)
    ax = axes[3]
    ax.plot(degrees, tr_rmse, marker="o", label="training")
    ax.plot(degrees, te_rmse, marker="o", label="test")
    ax.set_yscale("log")
    ax.set_xlabel("polynomial degree M")
    ax.set_title("RMSE vs degree")
    ax.legend()
    fig.suptitle("Polynomial curve fitting: under-fit, good fit, over-fit", x=0.01, ha="left", weight="bold")
    save(fig, "u1_polyfit")
    return {"degrees": degrees, "train_rmse": _r(tr_rmse), "test_rmse": _r(te_rmse),
            "best_degree": best, "m9_test_rmse": round(te_rmse[9], 4),
            "m9_regularised_test_rmse": round(reg_te, 4), "n_train": 12}


def trends(df: pd.DataFrame, history: pd.DataFrame) -> dict:
    """Polynomial (degree 2) trend of mean SGPA per semester, split by outcome."""
    h = history.merge(df[["student_id", C.TARGET]], on="student_id")
    out = {}
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for lab, name, col in ((0, "Not at risk", STATE[2]), (1, "At risk", STATE[0])):
        g = h[h[C.TARGET] == lab].groupby("semester")["sgpa"].mean()
        w = np.polyfit(g.index, g.values, 2)
        out[name] = {"semester": g.index.tolist(), "mean_sgpa": _r(g.values),
                     "coef_a2_a1_a0": _r(w), "slope_at_last": round(float(2 * w[0] * g.index[-1] + w[1]), 4)}
        ax.scatter(g.index, g.values, color=col, s=30, zorder=3)
        xs = np.linspace(1, g.index.max(), 100)
        ax.plot(xs, np.polyval(w, xs), color=col)
        ax.annotate(name, (g.index[-1], g.values[-1]), xytext=(6, 0), textcoords="offset points",
                    color=MUTED, va="center")
    ax.set_xlabel("semester")
    ax.set_ylabel("mean SGPA")
    ax.set_title("Semester trend, degree-2 fit")
    save(fig, "u1_trend")
    return out


def figures(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    for ax, f in zip(axes, ["current_cgpa", "attendance_percentage", "internal_assessment_average",
                            "daily_study_hours"]):
        for lab, name, col in ((0, "Not at risk", STATE[2]), (1, "At risk", STATE[0])):
            ax.hist(df.loc[df[C.TARGET] == lab, f], bins=18, alpha=0.65, color=col, label=name,
                    edgecolor="white", linewidth=0.6)
        ax.set_title(f.replace("_", " "))
    axes[0].legend()
    save(fig, "u1_distributions")

    cols = C.FEATURES + [C.TARGET]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), [c.replace("_", " ") for c in cols], rotation=60, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(cols)), [c.replace("_", " ") for c in cols], fontsize=7.5)
    ax.grid(False)
    fig.colorbar(im, fraction=0.04)
    ax.set_title("Pearson correlation (15 features + target)")
    save(fig, "u1_correlation")


def run(df: pd.DataFrame, history: pd.DataFrame) -> dict:
    figures(df)
    corr = df[C.FEATURES + [C.TARGET]].corr()
    return {
        "n_rows": len(df), "n_features": len(C.FEATURES),
        "class_balance": {"not_at_risk": int((df[C.TARGET] == 0).sum()), "at_risk": int(df[C.TARGET].sum())},
        "describe": describe(df),
        "covariance_pairs": covariance_pairs(df),
        "corr_with_target": {k: round(float(v), 3) for k, v in
                             corr[C.TARGET].drop(C.TARGET).sort_values().items()},
        "corr_matrix": {"labels": list(corr.columns), "values": _r(corr.to_numpy())},
        "probability": probability(df),
        "bayes": bayes(df),
        "independence": independence(df),
        "poly_fit": poly_fit(df),
        "trends": trends(df, history),
    }


def _r(x, n=4):
    return np.round(np.asarray(x, float), n).tolist()


def _round(d):
    if isinstance(d, dict):
        return {k: _round(v) for k, v in d.items()}
    return round(float(d), 4)
