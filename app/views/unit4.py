import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import C, GRID, MUTED, SERIES, STATE, header, history, metrics, table

header("Hidden Markov models", "Unit 4 · sequential data",
       "Each student's semester SGPA band (Poor / Average / Good) is an observation emitted by a hidden academic "
       "state. Baum–Welch learns the dynamics; Viterbi decodes the most likely path of states.")
M = metrics()["unit4"]
H = history()
NODE = ["#cf4b44", "#b3861a", "#3a8ad0"]


def state_machine(A, pi) -> str:
    pos = [(150, 300), (450, 110), (750, 300)]
    defs, paths, nodes = [], [], []
    for i in range(3):
        for j in range(3):
            p = A[i][j]
            if p < 0.005:
                continue
            w = 1.2 + 14 * p
            dur = max(0.6, 3.2 - 3 * p)
            if i == j:
                x, y = pos[i]
                d = f"M {x - 28} {y - 52} C {x - 90} {y - 190}, {x + 90} {y - 190}, {x + 28} {y - 52}"
                lx, ly = x, y - 168
            else:
                (x1, y1), (x2, y2) = pos[i], pos[j]
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                nx, ny = -(y2 - y1), (x2 - x1)
                n = np.hypot(nx, ny)
                cx, cy = mx + 55 * nx / n, my + 55 * ny / n
                ux, uy = (x2 - x1) / np.hypot(x2 - x1, y2 - y1), (y2 - y1) / np.hypot(x2 - x1, y2 - y1)
                d = f"M {x1 + 58 * ux} {y1 + 58 * uy} Q {cx} {cy} {x2 - 62 * ux} {y2 - 62 * uy}"
                lx, ly = 0.25 * x1 + 0.5 * cx + 0.25 * x2, 0.25 * y1 + 0.5 * cy + 0.25 * y2
            col = NODE[i]
            paths.append(f'<path d="{d}" stroke="{col}" stroke-opacity=".22" stroke-width="{w:.1f}" fill="none"/>'
                         f'<path class="flow" d="{d}" stroke="{col}" stroke-width="{max(1.2, w / 2.4):.1f}" fill="none" '
                         f'stroke-dasharray="2 14" style="animation-duration:{dur:.2f}s" marker-end="url(#ar{i})"/>'
                         f'<text x="{lx:.0f}" y="{ly:.0f}" class="pl">{p:.2f}</text>')
        defs.append(f'<marker id="ar{i}" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="5" markerHeight="5" '
                    f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{NODE[i]}"/></marker>')
        x, y = pos[i]
        nodes.append(f'<g class="node" style="animation-delay:{i * .25}s"><circle cx="{x}" cy="{y}" r="56" fill="#15231e" '
                     f'stroke="{NODE[i]}" stroke-width="2"/><circle cx="{x}" cy="{y}" r="56" fill="none" stroke="{NODE[i]}" '
                     f'class="halo"/><text x="{x}" y="{y - 4}" class="nl">{C.STATE_NAMES[i]}</text>'
                     f'<text x="{x}" y="{y + 18}" class="np">π = {pi[i]:.2f}</text></g>')
    return f"""<style>@font-face{{font-family:'Space Grotesk';font-weight:700;src:url('/app/static/fonts/SpaceGrotesk-700.ttf')}}
@font-face{{font-family:'IBM Plex Mono';font-weight:500;src:url('/app/static/fonts/IBMPlexMono-500.ttf')}}
html,body{{margin:0;background:transparent}}</style>
<div style="border:1px solid #26392f;border-radius:20px;background:rgba(21,35,30,.7);padding:.5rem">
<svg viewBox="0 0 900 400" width="100%" role="img" aria-label="HMM state transition diagram">
<style>
.flow{{animation:dash linear infinite}} @keyframes dash{{to{{stroke-dashoffset:-32}}}}
.pl{{fill:#c9cfc6;font:500 15px 'IBM Plex Mono',monospace;text-anchor:middle;paint-order:stroke;stroke:#0f1a16;stroke-width:5px}}
.nl{{fill:#fff;font:700 17px 'Space Grotesk',sans-serif;text-anchor:middle}} .np{{fill:#8fa197;font:400 12px 'IBM Plex Mono',monospace;text-anchor:middle}}
.halo{{animation:halo 2.6s ease-out infinite;transform-box:fill-box;transform-origin:center}}
@keyframes halo{{0%{{opacity:.7;transform:scale(1)}}100%{{opacity:0;transform:scale(1.5)}}}}
.node{{animation:pop .8s cubic-bezier(.2,1.4,.4,1) both}} @keyframes pop{{from{{opacity:0;transform:translateY(20px)}}}}
</style><defs>{''.join(defs)}</defs>{''.join(paths)}{''.join(nodes)}</svg></div>"""


t1, t2, t3, t4 = st.tabs(["State machine", "Student explorer", "Prediction vs baselines", "Model selection & checks"])
with t1:
    st.iframe(state_machine(M["A"], M["pi"]), height=560)
    c = st.columns(3)
    for col, (mat, title, xl) in zip(c, ((M["A"], "Learned A", C.STATE_NAMES),
                                         (M["truth"]["A_true"], "True A (generator)", C.STATE_NAMES),
                                         (M["B"], "Learned emissions B", C.BAND_NAMES))):
        f = go.Figure(go.Heatmap(z=mat, x=xl, y=C.STATE_NAMES, zmin=0, zmax=1, text=np.round(mat, 2),
                                 texttemplate="%{text}", colorscale=[[0, "#15231e"], [1, "#b3861a"]], showscale=False))
        f.update_layout(title=title, height=300, yaxis_autorange="reversed")
        col.plotly_chart(f, width="stretch")
    st.caption(f"Max |Â − A| = {M['truth']['A_max_abs_error']:.3f}. Baum–Welch converged in "
               f"{M['baum_welch_iterations']} iterations, log-likelihood {M['loglik']}.")

with t2:
    ex = {e["kind"]: e for e in M["examples"]}
    pick = st.radio("Example trajectory", list(ex), horizontal=True)
    sid = st.selectbox("…or any student", sorted(H.student_id.unique()), index=None, placeholder="Search a student id")
    from earlydrop.unit4_hmm import DiscreteHMM

    @st.cache_resource
    def model():
        m = DiscreteHMM(K=3)
        m.pi, m.A, m.B = np.array(M["pi"]), np.array(M["A"]), np.array(M["B"])
        return m

    m = model()
    if sid:
        g = H[H.student_id == sid].sort_values("semester")
        sg, band, true = g.sgpa.to_numpy(), g.band.to_numpy(), g.true_state.to_numpy()
    else:
        e = ex[pick]
        sg, band, true = np.array(e["sgpa"]), np.array(e["band"]), np.array(e["true_state"])
    path = m.viterbi(band)
    alpha, _ = m.forward(band)
    nxt, znext = m.predict_next(band)
    t = np.arange(1, len(sg) + 1)
    f = go.Figure()
    for lo, hi, name in ((0, C.BAND_EDGES[0], "Poor"), (C.BAND_EDGES[0], C.BAND_EDGES[1], "Average"),
                         (C.BAND_EDGES[1], 10, "Good")):
        f.add_hrect(y0=lo, y1=hi, line_width=0, fillcolor=["#cf4b44", "#b3861a", "#3a8ad0"][C.BAND_NAMES.index(name)],
                    opacity=0.05, annotation_text=name, annotation_position="right", annotation_font_color=MUTED)
    f.add_scatter(x=t, y=sg, mode="lines+markers", line=dict(color="#c9cfc6", width=2, shape="spline"),
                  marker=dict(size=18, color=[NODE[s] for s in path], line=dict(width=3, color="#0f1a16")),
                  text=[f"Viterbi: {C.STATE_NAMES[s]}<br>true: {C.STATE_NAMES[q]}" for s, q in zip(path, true)],
                  hovertemplate="semester %{x}<br>SGPA %{y}<br>%{text}<extra></extra>", name="SGPA")
    f.update_layout(height=380, title="SGPA trajectory · dot colour = decoded hidden state", yaxis_range=[3.5, 10],
                    xaxis_title="semester", showlegend=False)
    st.plotly_chart(f, width="stretch")
    a, b = st.columns(2)
    f = go.Figure(go.Scatter(x=t, y=alpha[:, 0], fill="tozeroy", line=dict(color=STATE[0], width=2),
                             fillcolor="rgba(207,75,68,.18)"))
    f.update_layout(height=280, title="Filtered P(Struggling | grades so far)", yaxis_range=[0, 1], xaxis_title="semester")
    a.plotly_chart(f, width="stretch")
    f = go.Figure(go.Bar(x=C.BAND_NAMES, y=nxt, marker_color=NODE, text=[f"{v:.0%}" for v in nxt],
                         textposition="outside"))
    f.update_layout(height=280, title="Forecast for the next semester", yaxis_range=[0, 1.15])
    b.plotly_chart(f, width="stretch")

with t3:
    ns = M["next_semester"]
    st.markdown(f"Each 3rd/4th-year student's **last completed semester** is held out ({ns['n_students']} students) "
                f"and predicted from the semesters before it.")
    a, b = st.columns(2)
    cat = pd.DataFrame(ns["categorical"])
    f = go.Figure(go.Bar(y=cat.model, x=cat.accuracy, orientation="h", marker_color=SERIES[0],
                         text=[f"{v:.1%}" for v in cat.accuracy], textposition="outside"))
    f.update_layout(height=320, title="Next-semester band accuracy", xaxis_range=[0, 1])
    a.plotly_chart(f, width="stretch")
    con = pd.DataFrame(ns["continuous"])
    f = go.Figure(go.Bar(y=con.model, x=con.rmse, orientation="h", marker_color=SERIES[2],
                         text=[f"{v:.3f}" for v in con.rmse], textposition="outside"))
    f.update_layout(height=320, title="Next-semester SGPA RMSE (lower is better)")
    b.plotly_chart(f, width="stretch")
    st.markdown(f"""**Reading it honestly.** The emission matrix is close to diagonal (bands are a reliable proxy for
the state), so the HMM's hard predictions match a plain Markov chain on grades; its edge is a lower log-loss
({cat.log_loss.iloc[3]:.3f} vs {cat.log_loss.iloc[2]:.3f}). For the continuous SGPA, the **Kalman filter** (local-level
linear dynamical system, q = {ns['kalman']['q']}, r = {ns['kalman']['r']}) beats persistence.""")
    ew = pd.DataFrame(M["early_warning"])
    f = go.Figure(go.Bar(x=ew.source, y=ew.roc_auc, marker_color=[SERIES[3], SERIES[0], SERIES[2]],
                         text=[f"{v:.3f}" for v in ew.roc_auc], textposition="outside"))
    f.update_layout(height=340, title="Early warning: predicting this semester's at-risk outcome (ROC-AUC)",
                    yaxis_range=[0.5, 1.05])
    st.plotly_chart(f, width="stretch")
    st.caption("Grade history alone already gives a strong signal; the survey adds attendance, internals and behaviour "
               "on top. The survey includes previous SGPA and CGPA, which is why adding the HMM forecast does not move it further.")

with t4:
    sel = pd.DataFrame(M["selection"])
    f = go.Figure()
    f.add_scatter(x=sel.K, y=sel.bic, mode="lines+markers", name="BIC", line=dict(color=SERIES[0]))
    f.add_scatter(x=sel.K, y=sel.aic, mode="lines+markers", name="AIC", line=dict(color=SERIES[1]))
    f.update_layout(height=320, title=f"Number of hidden states: BIC picks K = {M['K_best_bic']}", xaxis_title="K")
    st.plotly_chart(f, width="stretch")
    st.dataframe(table(M["selection"]), width="stretch")
    hl = M["hmmlearn"]
    st.markdown(f"""- From-scratch log-likelihood **{M['loglik']}** vs hmmlearn with the same parameters **{hl['loglik_same_params']}**
- hmmlearn's own best fit: **{hl['loglik_own_fit']}**; Viterbi paths agree on **{hl['viterbi_agreement']:.0%}** of semesters
- Viterbi recovers the true hidden state **{M['truth']['viterbi_state_accuracy']:.1%}** of the time
  (reading the band as the state directly: {M['truth']['band_equals_state_accuracy']:.1%})
- {M['n_observations']} observations from {M['n_students']} students, sequence lengths {M['lengths']}""")
