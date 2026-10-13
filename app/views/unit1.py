import numpy as np
import plotly.graph_objects as go
import streamlit as st

from common import C, GRID, MUTED, SERIES, STATE, clean, metrics, nice, header, table

header("Statistics, probability & Bayes", "Unit 1 · exploratory data analysis")
M = metrics()["unit1"]
df = clean()

t1, t2, t3, t4 = st.tabs(["Distributions", "Covariance & correlation", "Bayes' rule (live)", "Curve fitting (live)"])

with t1:
    f = st.selectbox("Feature", C.FEATURES, index=1, format_func=nice)
    fig = go.Figure()
    for lab, name, col in ((0, "Not at risk", STATE[2]), (1, "At risk", STATE[0])):
        fig.add_histogram(x=df.loc[df[C.TARGET] == lab, f], name=name, marker_color=col, opacity=0.75,
                          marker_line=dict(width=1, color="#0f1a16"))
    fig.update_layout(barmode="overlay", title=nice(f), height=380, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")
    p = M["probability"]
    c = st.columns(4)
    c[0].metric("E[backlogs] = Σ x·P(x)", f"{p['backlogs_expectation']:.3f}")
    c[1].metric("Var[backlogs]", f"{p['backlogs_variance']:.3f}")
    c[2].metric("CGPA median", f"{p['cgpa_quantiles']['0.5']:.2f}")
    c[3].metric("P(CGPA < 6): data vs normal", f"{p['p_cgpa_below_6_empirical']:.3f} / {p['p_cgpa_below_6_normal']:.3f}")
    st.dataframe(table(M["describe"]), width="stretch")

with t2:
    cm = M["corr_matrix"]
    fig = go.Figure(go.Heatmap(z=cm["values"], x=[nice(l) for l in cm["labels"]], y=[nice(l) for l in cm["labels"]],
                               zmin=-1, zmax=1, colorscale=[[0, "#3a8ad0"], [0.5, "#26392f"], [1, "#cf4b44"]],
                               hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>"))
    fig.update_layout(height=640, title="Pearson correlation")
    st.plotly_chart(fig, width="stretch")
    st.markdown("**Covariance pairs**: Cov(X,Y) = E[(X − μx)(Y − μy)], r = Cov / (σx σy)")
    st.dataframe(table(M["covariance_pairs"]), width="stretch")

with t3:
    st.markdown("**H** = student is at risk. Pick the evidence thresholds; the posterior is recomputed from the data.")
    a, b = st.columns(2)
    att = a.slider("E1: attendance below (%)", 60, 90, int(C.ATTENDANCE_RULE))
    inte = b.slider("E2: internal marks below (%)", 40, 80, 60)
    H = df[C.TARGET] == 1
    E1, E2 = df["attendance_percentage"] < att, df["internal_assessment_average"] < inte
    prior = H.mean()
    post1 = (E1 & H).sum() / H.sum() * prior / max(E1.mean(), 1e-9)
    l2h, l2n = (E2 & H).sum() / H.sum(), (E2 & ~H).sum() / (~H).sum()
    post2 = l2h * post1 / (l2h * post1 + l2n * (1 - post1))
    exact = H[E1 & E2].mean() if (E1 & E2).any() else float("nan")
    fig = go.Figure(go.Bar(x=["Prior P(H)", f"+ attendance < {att}%", f"+ internals < {inte}%"],
                           y=[prior, post1, post2], marker_color=[MUTED, SERIES[0], STATE[0]],
                           text=[f"{v:.2f}" for v in (prior, post1, post2)], textposition="outside"))
    fig.update_layout(yaxis_range=[0, 1.1], height=380, title="Bayesian updating: P(H | E) = P(E | H) P(H) / P(E)")
    st.plotly_chart(fig, width="stretch")
    st.markdown(f"Exact P(H | E1, E2) from the data: **{exact:.3f}** on {(E1 & E2).sum()} students. "
                f"The sequential update assumes E1 ⟂ E2 | H (conditional independence); the gap is "
                f"**{post2 - exact:+.3f}**.")
    ind = M["independence"]
    st.markdown(f"""**Independence check** (A = attendance < 75 %, B = study ≤ 1.5 h/day):
P(A∩B) = {ind['p_AB']:.3f} vs P(A)P(B) = {ind['p_A_p_B']:.3f} → *dependent* (χ² p = {ind['chi2_p_value']:.2g}).
Given at risk: {ind['given_at_risk']['p_AB']:.3f} vs {ind['given_at_risk']['p_A_p_B']:.3f}; much closer, so A and B
are nearly **conditionally independent** given risk status.""")

with t4:
    st.markdown("Fit an M-th order polynomial to a few students (attendance → internal marks) and watch it over-fit.")
    a, b, c = st.columns(3)
    M_deg = a.slider("degree M", 0, 9, 3)
    n_tr = b.slider("training points", 6, 60, 12)
    lnlam = c.slider("ln λ (regularisation, −30 = none)", -30, 0, -30)
    x = (df["attendance_percentage"].to_numpy() - 35) / 65
    y = df["internal_assessment_average"].to_numpy() / 100
    idx = np.random.default_rng(C.SEED).permutation(len(df))
    tr, te = idx[:n_tr], idx[n_tr:]
    Phi = np.vander(x[tr], M_deg + 1)
    w = np.linalg.solve(Phi.T @ Phi + np.exp(lnlam) * np.eye(M_deg + 1), Phi.T @ y[tr])
    grid = np.linspace(x.min(), x.max(), 300)
    rm = lambda i: float(np.sqrt(np.mean((np.vander(x[i], M_deg + 1) @ w - y[i]) ** 2)))  # noqa: E731
    fig = go.Figure()
    fig.add_scatter(x=x[te], y=y[te], mode="markers", name="unseen", marker=dict(color=MUTED, size=5, opacity=.5))
    fig.add_scatter(x=x[tr], y=y[tr], mode="markers", name="training", marker=dict(color=SERIES[0], size=10))
    fig.add_scatter(x=grid, y=np.clip(np.vander(grid, M_deg + 1) @ w, -0.2, 1.3), name=f"M = {M_deg}",
                    line=dict(color=SERIES[1], width=3))
    fig.update_layout(height=420, yaxis_range=[0.1, 1.1], xaxis_title="attendance (scaled)",
                      yaxis_title="internal marks (scaled)", yaxis_gridcolor=GRID)
    st.plotly_chart(fig, width="stretch")
    k = st.columns(2)
    k[0].metric("training RMSE", f"{rm(tr):.4f}")
    k[1].metric("test RMSE", f"{rm(te):.4f}")
    st.subheader("Semester trend (degree-2 fit)")
    st.json(M["trends"], expanded=False)
