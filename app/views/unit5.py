import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from common import C, MUTED, SERIES, clean, header, kpi, metrics, nice, table

header("Combining models", "PBL-III · Unit 5 · CART · bagging · boosting · BMA · GAM · stacking",
       "Can a committee of models flag at-risk students better than the best single model? "
       "Every ensemble below is scored on the same 5 folds as Unit 2.")
M = metrics()["unit5"]
df = clean()
FAM = {"single": "#9aa3b2", "bagging": SERIES[1], "boosting": SERIES[0], "averaging": SERIES[3], "combining": SERIES[4]}

cv = pd.DataFrame(M["compare_cv"])
tree = cv[cv.model.str.startswith("CART (scratch")].iloc[0]
lr = cv[cv.model.str.startswith("Logistic")].iloc[0]
best = cv.iloc[0]
c = st.columns(4)
kpi(c[0], tree.auc_fold_mean, "single tree (full depth)", "CV ROC-AUC", dec=3, tag="CART")
kpi(c[1], cv[cv.model.str.startswith("Random Forest (scratch")].iloc[0].auc_fold_mean, "random forest (scratch)",
    "same trees, averaged", dec=3, tag="BAGGING")
kpi(c[2], lr.auc_fold_mean, "logistic regression", "Unit 2 baseline", dec=3, tag="BASELINE")
kpi(c[3], best.auc_fold_mean, best.model.split(" (")[0], "best overall", dec=3, tag="BEST")
st.write("")

t1, t2, t3, t4, t5 = st.tabs(["Comparison", "Trees & forests", "Boosting", "BMA & GAM", "Stacking & bias–variance"])

with t1:
    r = cv.iloc[::-1]
    f = go.Figure(go.Bar(x=r.auc_fold_mean, y=r.model, orientation="h", marker_color=[FAM[x] for x in r.family],
                         error_x=dict(type="data", array=r.auc_fold_std, color="#c9cfc6", thickness=1.2),
                         text=[f"{v:.3f}" for v in r.auc_fold_mean], textposition="outside"))
    f.update_layout(height=620, xaxis_range=[0.75, 1.0], title="5-fold CV ROC-AUC (mean ± sd) · colour = family",
                    margin=dict(l=10, r=40, t=48, b=10))
    st.plotly_chart(f, width="stretch")
    st.caption("grey single model · blue bagging · ochre boosting · violet Bayesian model averaging · green voting / stacking")
    show = ["model", "family", "auc_fold_mean", "auc_fold_std", "accuracy", "precision", "recall", "f1", "brier",
            "log_loss", "fit_seconds"]
    st.markdown("**Out-of-fold metrics (threshold 0.5)**")
    st.dataframe(table(cv[show].to_dict("records")), width="stretch", hide_index=True)
    st.markdown("**Is any ensemble significantly better than logistic regression?** "
                "Repeated 5 × 3 CV, Nadeau–Bengio corrected t-test.")
    st.dataframe(table(M["significance"]), width="stretch", hide_index=True)
    st.markdown("**Held-out 25 % test set**")
    st.dataframe(table(M["compare_test"]), width="stretch", hide_index=True)

with t2:
    a, b = st.columns(2)
    d = pd.DataFrame(M["cart"]["depth_curve"])
    f = go.Figure()
    for col, name, color in (("train_acc", "train", SERIES[0]), ("cv_acc", "5-fold CV", SERIES[1]),
                             ("test_acc", "held-out test", SERIES[2])):
        f.add_scatter(x=d.depth, y=d[col], name=name, mode="lines+markers", line=dict(color=color))
    f.update_layout(height=360, title="CART: deeper trees memorise the training set", xaxis_title="max depth",
                    yaxis_title="accuracy")
    a.plotly_chart(f, width="stretch")
    f = go.Figure()
    for key, name, color in (("oob_curve_bagging", "bagging (all 15 features)", SERIES[0]),
                             ("oob_curve_rf", "random forest (√15 ≈ 3 per split)", SERIES[1])):
        o = pd.DataFrame(M["forest"][key])
        f.add_scatter(x=o.n_trees, y=o.oob_error, name=name, mode="lines+markers", line=dict(color=color))
    f.update_layout(height=360, title="Out-of-bag error as trees are added", xaxis_type="log",
                    xaxis_title="number of trees", yaxis_title="OOB error")
    b.plotly_chart(f, width="stretch")
    p, fr = M["cart"]["pruning"], M["forest"]
    st.caption(f"Cost-complexity pruning picks α = {p['ccp_alpha']:.4f}: {p['full_leaves']} leaves → {p['leaves']} "
               f"(test accuracy {p['full_test_acc']:.2f} → {p['test_acc']:.2f}). Average correlation between two trees' "
               f"predictions: bagging {fr['tree_correlation_bagging']:.2f}, random forest {fr['tree_correlation_rf']:.2f}. "
               "Lower correlation is why random feature subsets help.")

    st.markdown("#### Try it live")
    x1, x2, x3 = st.columns(3)
    kind = x1.selectbox("Model", ["Decision tree", "Random forest", "AdaBoost", "Gradient boosting"])
    n_est = x2.slider("Trees / rounds", 1, 300, 100, disabled=kind == "Decision tree")
    depth = x3.slider("Max depth", 1, 12, 3 if kind != "Random forest" else 12)
    X, y = df[C.FEATURES].to_numpy(float), df[C.TARGET].to_numpy()
    model = {"Decision tree": lambda: DecisionTreeClassifier(max_depth=depth, random_state=C.SEED),
             "Random forest": lambda: RandomForestClassifier(n_est, max_depth=depth, random_state=C.SEED, n_jobs=-1),
             "AdaBoost": lambda: AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth), n_estimators=n_est,
                                                    random_state=C.SEED),
             "Gradient boosting": lambda: GradientBoostingClassifier(n_estimators=n_est, max_depth=depth,
                                                                     learning_rate=0.05, random_state=C.SEED)}[kind]()
    with st.spinner("5-fold CV ..."):
        p_ = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=C.SEED),
                               method="predict_proba")[:, 1]
    m = st.columns(3)
    m[0].metric("CV ROC-AUC", f"{roc_auc_score(y, p_):.3f}", f"{roc_auc_score(y, p_) - lr.auc_fold_mean:+.3f} vs logistic")
    m[1].metric("CV accuracy", f"{np.mean((p_ > 0.5) == y):.3f}")
    m[2].metric("recall (at risk caught)", f"{np.mean(p_[y == 1] > 0.5):.2f}")

with t3:
    a, b = st.columns(2)
    ad = pd.DataFrame(M["adaboost"]["curve"])
    f = go.Figure()
    f.add_scatter(x=ad.m, y=ad.train_error, name="train error", line=dict(color=SERIES[0]))
    f.add_scatter(x=ad.m, y=ad.test_error, name="test error", line=dict(color=SERIES[1]))
    f.add_scatter(x=ad.m, y=ad.exp_loss, name="exponential loss (upper bound)", line=dict(color=MUTED, dash="dash"))
    f.update_layout(height=380, title="AdaBoost from scratch (decision stumps)", xaxis_type="log", xaxis_title="round m")
    a.plotly_chart(f, width="stretch")
    f = go.Figure()
    for s, color in zip(M["gboost"]["shrinkage"], SERIES):
        cc = pd.DataFrame(s["curve"])
        f.add_scatter(x=cc.m, y=cc.test, name=f"ν = {s['learning_rate']}", line=dict(color=color))
    f.update_layout(height=380, title="Gradient boosting: smaller steps need more trees, over-fit less",
                    xaxis_title="iterations", yaxis_title="test log-loss", yaxis_range=[0.2, 0.7])
    b.plotly_chart(f, width="stretch")
    s1, s2, fs = M["adaboost"]["scratch_vs_sklearn"], M["gboost"]["scratch_vs_sklearn"], M["adaboost"]["first_stump"]
    st.caption(f"First stump: {fs['feature']} ≤ {fs['threshold']} with weighted error ε = {fs['eps']:.3f}, so "
               f"α = ln((1 − ε)/ε) = {fs['alpha']:.3f}. Scratch AdaBoost agrees with sklearn on "
               f"{s1['test_agreement']:.0%} of test students (decision correlation {s1['decision_corr']:.3f}); scratch "
               f"gradient boosting differs from sklearn by at most {s2['max_abs_proba_diff']:.4f} in probability.")
    st.dataframe(table([{"learning_rate": s["learning_rate"], "best_iteration": s["best_iter"],
                         "best_test_logloss": s["best_test_logloss"], "after_300": s["final_test_logloss"]}
                        for s in M["gboost"]["shrinkage"]]), width="stretch", hide_index=True)

with t4:
    bm = M["bma"]
    a, b = st.columns(2)
    inc = bm["inclusion"]
    f = go.Figure(go.Bar(x=list(inc.values()), y=list(inc), orientation="h", marker_color=SERIES[3],
                         text=[f"{v:.2f}" for v in inc.values()], textposition="outside"))
    f.update_layout(height=340, xaxis_range=[0, 1.1], title="Posterior inclusion probability of each feature block")
    a.plotly_chart(f, width="stretch")
    cc = pd.DataFrame(bm["concentration"])
    f = go.Figure(go.Scatter(x=cc.n, y=cc.top_posterior, mode="lines+markers", line=dict(color=SERIES[3])))
    f.update_layout(height=340, title="BMA picks one model as data grows (Bishop §14.1)", xaxis_type="log",
                    xaxis_title="training students", yaxis_title="posterior of the top model", yaxis_range=[0, 1])
    b.plotly_chart(f, width="stretch")
    st.markdown(f"**Top models out of {bm['n_models']}** (Laplace-approximated evidence, uniform prior over models)")
    st.dataframe(table([{"blocks": " + ".join(m["blocks"]) or "(intercept only)", "posterior": m["posterior"],
                         "log_evidence": m["log_evidence"], "parameters": m["n_params"]} for m in bm["top_models"]]),
                 width="stretch", hide_index=True)
    st.dataframe(table([{"model": k, **v} for k, v in bm["test"].items()]), width="stretch", hide_index=True)
    st.markdown("#### Generalized additive model: one smooth curve per feature")
    cols = st.columns(3)
    for i, s in enumerate(M["gam"]["shapes"]):
        f = go.Figure(go.Scatter(x=s["x"], y=s["f"], mode="lines", line=dict(color=SERIES[i % 5], width=3)))
        f.add_hline(y=0, line_color=MUTED, line_dash="dot")
        f.update_layout(height=250, title=nice(s["feature"]), yaxis_title="log-odds", margin=dict(l=10, r=10, t=40, b=10))
        cols[i % 3].plotly_chart(f, width="stretch")

with t5:
    a, b = st.columns(2)
    bv = pd.DataFrame(M["bias_variance"]["rows"])
    f = go.Figure()
    f.add_bar(y=bv.model, x=bv.bias2, name="bias²", orientation="h", marker_color=SERIES[0])
    f.add_bar(y=bv.model, x=bv.variance, name="variance", orientation="h", marker_color=SERIES[2])
    f.update_layout(barmode="stack", height=380, title="Brier score = bias² + variance (30 bootstrap training sets)")
    a.plotly_chart(f, width="stretch")
    stk = M["stacking"]
    f = go.Figure(go.Heatmap(z=stk["base_corr"], x=stk["bases"], y=stk["bases"], colorscale="Cividis", zmin=0.5, zmax=1,
                             text=np.round(stk["base_corr"], 2), texttemplate="%{text}"))
    f.update_layout(height=380, title="Stacking: correlation of base-model predictions (out-of-fold)")
    b.plotly_chart(f, width="stretch")
    st.markdown("**Meta-learner weights** (logistic regression on base-model probabilities)")
    st.dataframe(table([{"base": k, "weight": v, "base_oof_auc": stk["base_auc_oof"][k]}
                        for k, v in stk["meta_weights"].items()]), width="stretch", hide_index=True)
    st.markdown("**Feature importance**: random-forest impurity vs gradient-boosting permutation vs |logistic weight|")
    st.dataframe(table(M["importance"]), width="stretch", hide_index=True)
