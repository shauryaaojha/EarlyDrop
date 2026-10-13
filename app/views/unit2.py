import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve

from common import C, MUTED, SERIES, clean, metrics, nice, header, table
from earlydrop import unit2_models as U2

header("Linear models for regression & classification", "Unit 2 · comparative evaluation")
M = metrics()["unit2"]
df = clean()


@st.cache_resource
def fitted():
    Xtr, Xte, ytr, yte, _, _ = U2.split(df)
    models = U2.classifiers()
    probs = {n: m.fit(Xtr, ytr).predict_proba(Xte)[:, 1] for n, m in models.items()}
    return yte, probs, models


t1, t2, t3 = st.tabs(["Regression", "Classification", "Bayesian logistic regression"])
with t1:
    R = M["regression"]
    st.markdown(f"Target: **end-semester SGPA** (the outcome of the semester in progress). "
                f"Train {R['n_train']} / test {R['n_test']}, features standardised.")
    st.dataframe(table(R["table"]), width="stretch")
    c = st.columns(4)
    b = R["blr"]
    c[0].metric("evidence-optimal α", f"{b['alpha']:.3f}")
    c[1].metric("noise σ = 1/√β", f"{b['noise_std']:.3f}")
    c[2].metric("effective parameters γ", f"{b['gamma_eff_params']:.2f}")
    c[3].metric("95 % interval coverage", f"{b['coverage_95']:.0%}")
    rb = R["robustness"]
    fig = go.Figure()
    for i, (k, lab) in enumerate((("clean_rmse", "clean training data"), ("corrupted_rmse", "8 % corrupted targets"))):
        fig.add_bar(x=list(rb), y=[rb[m][k] for m in rb], name=lab, marker_color=SERIES[i])
    fig.update_layout(barmode="group", height=340, title="Robust regression: Huber vs OLS under data-entry errors",
                      yaxis_title="test RMSE")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Ridge α chosen by CV: {R['ridge_alpha']}. OLS by normal equations matches sklearn: {R['ols_matches_sklearn']}.")

with t2:
    Cl = M["classification"]
    st.markdown("**Held-out test set (25 %)**")
    st.dataframe(table(Cl["test"]), width="stretch")
    st.markdown("**Stratified 5-fold cross-validation**")
    st.dataframe(table(Cl["cv5"]), width="stretch")
    yte, probs, _ = fitted()
    fig = go.Figure()
    for i, (n, p) in enumerate(probs.items()):
        fpr, tpr, _ = roc_curve(yte, p)
        fig.add_scatter(x=fpr, y=tpr, name=n, line=dict(width=2, color=(SERIES + [MUTED, "#c9cfc6"])[i % 7]))
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color=MUTED, width=1), showlegend=False)
    fig.update_layout(height=460, title="ROC curves", xaxis_title="false positive rate", yaxis_title="true positive rate")
    st.plotly_chart(fig, width="stretch")
    g = Cl["svm_rbf_grid"]
    st.caption(f"Kernel trick: RBF SVM grid search best C = {g['best_params']['C']}, γ = {g['best_params']['gamma']} "
               f"(CV AUC {g['best_cv_auc']:.3f}).")

with t3:
    yte, probs, models = fitted()
    st.markdown("""Implemented from scratch: Gaussian prior **p(w) = N(0, α⁻¹I)**, MAP by Newton–Raphson (IRLS),
**Laplace** posterior N(w_MAP, H⁻¹), predictive probability by the **probit approximation**
σ(κ(σₐ²) μₐ), κ = (1 + πσₐ²/8)^(−½).""")
    thr = st.slider("decision threshold (lower = catch more at-risk students)", 0.05, 0.95, 0.5, 0.05)
    p = probs["Bayesian Logistic (Laplace)"]
    cm = confusion_matrix(yte, (p >= thr).astype(int))
    tn, fp, fn, tp = cm.ravel()
    c = st.columns(4)
    c[0].metric("recall (at-risk caught)", f"{tp / (tp + fn):.0%}", f"{tp} of {tp + fn}")
    c[1].metric("precision", f"{tp / max(tp + fp, 1):.0%}")
    c[2].metric("false alarms", int(fp))
    c[3].metric("missed students", int(fn))
    w = pd.DataFrame(M["classification"]["blr_weights"]).sort_values("w_map")
    fig = go.Figure(go.Scatter(x=w["w_map"], y=[nice(f) for f in w["feature"]], mode="markers",
                               error_x=dict(array=2 * w["posterior_std"], color="#b9a6e3"),
                               marker=dict(size=10, color=SERIES[2])))
    fig.add_vline(x=0, line_color=MUTED)
    fig.update_layout(height=520, title="Posterior weights: MAP ± 2σ (standardised features)")
    st.plotly_chart(fig, width="stretch")
