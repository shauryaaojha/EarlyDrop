import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from common import C, MUTED, SERIES, clean, header, metrics, nice, real
from earlydrop.generate import band_of
from earlydrop.unit2_models import BayesianLogisticRegression
from earlydrop.unit4_hmm import DiscreteHMM

header("Predict a student", "Early warning · live inference",
       "Move the sliders. The Bayesian logistic model returns a risk probability with a 90 % credible band, "
       "K-Means assigns a persona, and the HMM forecasts next semester from the grade history.")
df = clean()
M = metrics()


@st.cache_resource
def models():
    sc = StandardScaler().fit(df[C.FEATURES].to_numpy(float))
    Z = sc.transform(df[C.FEATURES].to_numpy(float))
    y = df[C.TARGET].to_numpy()
    blr = BayesianLogisticRegression().fit(Z, y)
    km = KMeans(3, n_init=20, random_state=C.SEED).fit(Z)
    eng = df["overall_academic_engagement"].to_numpy()
    first = max(range(3), key=lambda k: y[km.labels_ == k].mean())
    rest = sorted((k for k in range(3) if k != first), key=lambda k: eng[km.labels_ == k].mean())
    names = {first: "High risk", rest[0]: "Coasting", rest[1]: "Engaged achievers"}
    hmm = DiscreteHMM(K=3)
    u4 = M["unit4"]
    hmm.pi, hmm.A, hmm.B = np.array(u4["pi"]), np.array(u4["A"]), np.array(u4["B"])
    return sc, blr, km, names, hmm


sc, blr, km, names, hmm = models()

SPEC = {  # feature: (min, max, step)
    "previous_sem_sgpa": (3.0, 10.0, 0.05), "current_cgpa": (3.0, 10.0, 0.05),
    "internal_assessment_average": (15.0, 100.0, 1.0), "attendance_percentage": (35.0, 100.0, 0.5),
    "active_backlogs": (0, 4, 1), "daily_study_hours": (0.5, 5.0, 0.5),
}
q = df[C.FEATURES]
presets = {
    "Median student": q.median(),
    "Typical high-risk": q[df[C.TARGET] == 1].median(),
    "Typical not-at-risk": q[df[C.TARGET] == 0].median(),
}
r = real()
if r is not None:
    for i in range(min(3, len(r))):
        presets[f"Real respondent #{i + 1}"] = r[C.FEATURES].iloc[i]
choice = st.segmented_control("Start from", list(presets), default="Median student")
if choice and st.session_state.get("_preset") != choice:
    st.session_state["_preset"] = choice
    for f in C.FEATURES:
        v = presets[choice][f]
        st.session_state[f"in_{f}"] = int(round(v)) if f in C.LIKERT or f == "active_backlogs" else float(v)

groups = {
    "Academic record": C.FEATURES[:5],
    "Study behaviour": ["daily_study_hours", "study_consistency", "revision_frequency",
                        "assignment_completion_consistency", "late_assignment_frequency"],
    "Mindset": ["academic_motivation", "overall_academic_engagement", "academic_workload_difficulty",
                "academic_performance_concern", "programme_completion_confidence"],
}
x = {}
cols = st.columns(3, gap="large")
for col, (g, feats) in zip(cols, groups.items()):
    col.markdown(f"**{g}**")
    for f in feats:
        lo, hi, step = SPEC.get(f, (1, 5, 1))
        key = f"in_{f}"
        if key not in st.session_state:
            v = presets["Median student"][f]
            st.session_state[key] = int(round(v)) if isinstance(step, int) else float(v)
        x[f] = col.slider(nice(f), lo, hi, step=step, key=key)

row = np.array([[x[f] for f in C.FEATURES]], float)
z = sc.transform(row)
mean, lo90, hi90 = blr.predict_with_uncertainty(z)
p, l, h = float(mean[0]), float(lo90[0]), float(hi90[0])
persona = names[int(km.predict(z)[0])]
level = "High" if p >= 0.5 else "Elevated" if p >= 0.25 else "Low"
colr = {"High": "#cf4b44", "Elevated": "#b3861a", "Low": "#3a8ad0"}[level]

st.write("")
a, b = st.columns([1, 1.2], gap="large")
with a:
    g = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * p, number=dict(suffix="%", font=dict(size=56, family="Space Grotesk")),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=MUTED), bar=dict(color=colr, thickness=0.28),
                   bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[100 * l, 100 * h], color="rgba(255,255,255,.08)")],
                   threshold=dict(line=dict(color="#fff", width=3), thickness=0.8, value=50))))
    g.update_layout(height=330, title=f"Risk of SGPA < 6.5 this semester · {level}")
    st.plotly_chart(g, width="stretch")
    st.markdown(f'<div class="ed-card"><b>{level} risk · {persona}</b><div style="color:#c9cfc6;margin-top:.4rem">'
                f'90 % credible band: <b>{l:.0%} – {h:.0%}</b> (Laplace posterior, 2000 samples).</div></div>',
                unsafe_allow_html=True)
with b:
    contrib = blr.w_map_[1:] * z[0]
    order = np.argsort(np.abs(contrib))[::-1][:8][::-1]
    f = go.Figure(go.Bar(x=contrib[order], y=[nice(C.FEATURES[i]) for i in order], orientation="h",
                         marker_color=["#cf4b44" if v > 0 else "#3a8ad0" for v in contrib[order]]))
    f.update_layout(height=330, title="What drives this score (log-odds contribution)", xaxis_title="← lowers risk · raises risk →")
    st.plotly_chart(f, width="stretch")

st.markdown("#### What if attendance changed?")
grid = np.linspace(40, 100, 61)
rows = np.repeat(row, len(grid), 0)
rows[:, C.FEATURES.index("attendance_percentage")] = grid
mm, ll, hh = blr.predict_with_uncertainty(sc.transform(rows), n_samples=800)
f = go.Figure()
f.add_scatter(x=np.r_[grid, grid[::-1]], y=np.r_[hh, ll[::-1]], fill="toself", fillcolor="rgba(227,195,90,.14)",
              line=dict(width=0), hoverinfo="skip", name="90 % band")
f.add_scatter(x=grid, y=mm, line=dict(color=SERIES[0], width=3), name="risk")
f.add_vline(x=C.ATTENDANCE_RULE, line_dash="dash", line_color=MUTED, annotation_text="75 % rule")
f.add_scatter(x=[x["attendance_percentage"]], y=[p], mode="markers", marker=dict(size=14, color=colr), name="now")
f.update_layout(height=320, yaxis_range=[0, 1], xaxis_title="attendance %", yaxis_title="P(at risk)")
st.plotly_chart(f, width="stretch")

st.markdown("#### Grade history → HMM forecast")
hist_txt = st.text_input("SGPA of completed semesters (comma-separated)", "7.9, 7.2, 6.6, 6.1")
try:
    sg = np.array([float(s) for s in hist_txt.replace(";", ",").split(",") if s.strip()])
    obs = band_of(sg)
    path = hmm.viterbi(obs)
    nxt, znext = hmm.predict_next(obs)
    c = st.columns(4)
    c[0].metric("decoded current state", C.STATE_NAMES[path[-1]])
    c[1].metric("P(Struggling next sem)", f"{znext[0]:.0%}")
    c[2].metric("P(Poor band next sem)", f"{nxt[0]:.0%}")
    c[3].metric("Viterbi path", " → ".join(C.STATE_NAMES[s][:4] for s in path))
except ValueError:
    st.warning("Enter numbers like 7.9, 7.2, 6.6")
st.caption("Decision support only: an early-warning score should trigger an offer of help, never a label.")
