import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from common import C, MUTED, SERIES, STATE, clean, metrics, nice, header, table

header("Mixture models & clustering", "Unit 3 · K-Means · EM · hierarchical · PCA · FA")
M = metrics()["unit3"]
df = clean()

t1, t2, t3, t4 = st.tabs(["Live clustering", "Comparison & personas", "PCA", "Factor analysis"])
with t1:
    a, b, c = st.columns([1, 1, 2])
    algo = a.selectbox("Algorithm", ["K-Means", "Gaussian mixture (EM)", "Hierarchical"])
    k = b.slider("k", 2, 8, 3)
    link = c.selectbox("Linkage (hierarchical)", ["ward", "complete", "average", "single"],
                       disabled=algo != "Hierarchical")
    feats = st.multiselect("Features", C.FEATURES, default=C.FEATURES, format_func=nice)
    if len(feats) < 2:
        st.stop()
    Z = StandardScaler().fit_transform(df[feats])
    if algo == "K-Means":
        lab = KMeans(k, n_init=20, random_state=C.SEED).fit_predict(Z)
    elif algo.startswith("Gaussian"):
        lab = GaussianMixture(k, covariance_type="full", n_init=3, random_state=C.SEED, reg_covar=1e-4).fit(Z).predict(Z)
    else:
        lab = AgglomerativeClustering(k, linkage=link).fit_predict(Z)
    P = PCA(2).fit_transform(Z)
    y = df[C.TARGET].to_numpy()
    fig = go.Figure()
    for j in np.unique(lab):
        m = lab == j
        fig.add_scatter(x=P[m, 0], y=P[m, 1], mode="markers", name=f"cluster {j} · {y[m].mean():.0%} at risk",
                        marker=dict(size=7, color=(SERIES + [MUTED, "#c9cfc6", "#5a6e64"])[j % 8],
                                    line=dict(width=1, color="#0f1a16")))
    fig.update_layout(height=520, title=f"{algo}, k = {k}, shown on the first two principal components",
                      xaxis_title="PC1", yaxis_title="PC2")
    st.plotly_chart(fig, width="stretch")
    m = st.columns(3)
    m[0].metric("silhouette", f"{silhouette_score(Z, lab):.3f}")
    m[1].metric("Davies–Bouldin (lower better)", f"{davies_bouldin_score(Z, lab):.3f}")
    m[2].metric("ARI vs at_risk", f"{adjusted_rand_score(y, lab):.3f}")

with t2:
    ks = pd.DataFrame(M["k_selection"])
    gs = pd.DataFrame(M["gmm_selection"])
    c = st.columns(3)
    for col, (y_, data, title) in zip(c, (("kmeans_inertia", ks, "Elbow (distortion J)"),
                                          ("kmeans_silhouette", ks, "Silhouette"),
                                          ("bic_full", gs, "GMM BIC (lower better)"))):
        f = go.Figure(go.Scatter(x=data["k"], y=data[y_], mode="lines+markers", line=dict(color=SERIES[0])))
        f.update_layout(height=280, title=title, xaxis_title="k")
        col.plotly_chart(f, width="stretch")
    st.markdown("**All methods at k = 3**")
    st.dataframe(table(M["compare"]), width="stretch")
    st.markdown("**Dissimilarity measures** (average linkage)")
    st.dataframe(table(M["dissimilarity"]), width="stretch")
    st.markdown("**K-Means personas**")
    per = pd.DataFrame(M["personas"])
    show = ["name", "size", "at_risk_rate", "current_cgpa", "attendance_percentage", "active_backlogs",
            "daily_study_hours", "study_consistency", "late_assignment_frequency", "overall_academic_engagement"]
    st.dataframe(table(per[show].to_dict("records"), "{:.2f}"), width="stretch")
    s = M["scratch_vs_sklearn"]
    st.caption(f"From-scratch K-Means vs sklearn ARI {s['kmeans_ari']}, from-scratch EM vs sklearn ARI {s['gmm_ari']} "
               f"(log-lik {s['gmm_scratch_loglik']} vs {s['gmm_sklearn_loglik']}, {s['em_iterations']} EM iterations).")

with t3:
    p = M["pca"]
    comp = list(range(1, len(p["explained_ratio"]) + 1))
    f = go.Figure()
    f.add_bar(x=comp, y=p["explained_ratio"], name="explained variance", marker_color=SERIES[0])
    f.add_scatter(x=comp, y=p["cumulative"], name="cumulative", line=dict(color=SERIES[1]))
    f.add_hline(y=0.9, line_dash="dash", line_color=MUTED)
    f.update_layout(height=360, title=f"Scree plot: {p['k90']} components reach 90 %, {p['k95']} reach 95 %")
    st.plotly_chart(f, width="stretch")
    f = go.Figure(go.Scatter(x=comp, y=p["acc_by_k"], mode="lines+markers", line=dict(color=SERIES[3])))
    f.update_layout(height=300, title="Choosing latent dimensions: CV accuracy using the first k PCs", xaxis_title="k")
    st.plotly_chart(f, width="stretch")
    lo = pd.DataFrame({"PC1": p["pc1_loadings"], "PC2": p["pc2_loadings"]})
    st.dataframe(lo.style.format("{:.3f}"), width="stretch")

with t4:
    fa = M["factor_analysis"]
    st.markdown(f"Factor analysis on the 28 Likert items. Cross-validated log-likelihood peaks at "
                f"**{fa['n_factors_best']} factors**. The 4-factor varimax solution:")
    cols = st.columns(len(fa["factors"]))
    for col, f in zip(cols, fa["factors"]):
        col.markdown(f"**Factor {f['factor']}**")
        for it in f["top_items"]:
            col.markdown(f"<small>{nice(it['item'])} · <code>{it['loading']:+.2f}</code></small>",
                         unsafe_allow_html=True)
