import streamlit as st

from common import kpi, metrics

M = metrics()
u1, u2, u3, u4, u5 = M["unit1"], M["unit2"], M["unit3"], M["unit4"], M["unit5"]
best = max(u2["classification"]["test"], key=lambda r: r["roc_auc"])
u5auc = lambda prefix: next(r["auc_fold_mean"] for r in u5["compare_cv"] if r["model"].startswith(prefix))  # noqa: E731
ew = {r["source"]: r["roc_auc"] for r in u4["early_warning"]}

st.markdown(f"""
<section class="ed-hero">
  <div class="ed-eyebrow">21CSC305P · Machine Learning</div>
  <h1 class="ed-hero-title">EarlyDrop</h1>
  <div class="ed-hero-sub">Spotting academic risk <em>before</em> the grades fall. Survey signals, grade
  trajectories and five units of machine learning, rolled into one early-warning system.</div>
  <div style="margin-top:1.4rem"><span class="ed-badge"><i></i>SYNTHETIC COHORT · n = {u1['n_rows']}</span></div>
</section>""", unsafe_allow_html=True)

c = st.columns(4)
kpi(c[0], u1["n_rows"], "students in the cohort", f"{M['meta']['n_real']} real responses for calibration", tag="DATA")
kpi(c[1], 100 * u1["class_balance"]["at_risk"] / u1["n_rows"], "at-risk rate", "end-semester SGPA < 6.5",
    dec=1, suf="%", tag="TARGET")
kpi(c[2], best["roc_auc"], "best ROC-AUC", best["model"], dec=3, tag="UNIT 2")
kpi(c[3], 100 * u4["A"][0][0], "stay in Struggling state", "HMM self-transition", dec=0, suf="%", tag="UNIT 4")

st.write("")
left, right = st.columns(2, gap="large")
with left:
    st.markdown("### The pipeline")
    steps = [
        ("Google Form → raw CSV", "the live form's exact 50-column export"),
        ("Cleaning", "duplicates · <code>80.50%</code> strings · mid-survey questions · typos like <code>7%</code>"),
        ("Unit 1 · statistics", "EDA · probability · Bayes' rule · curve fitting"),
        ("Unit 2 · linear models", "OLS · Ridge · Huber · Bayesian LR · LDA · SVM kernels"),
        ("Unit 3 · mixture models", "K-Means · EM for GMM · hierarchical · PCA · factor analysis"),
        ("Unit 4 · sequences", "Markov chain · HMM (Baum–Welch, Viterbi) · Kalman filter"),
        ("Unit 5 · combining models", "CART · bagging · random forest · AdaBoost · gradient boosting · BMA · GAM · stacking"),
        ("Early warning", "risk with an uncertainty band, persona and trajectory"),
    ]
    st.markdown('<div class="ed-steps">' + "".join(
        f'<div class="ed-step"><div><b>{a}</b><br><small>{b}</small></div></div>' for a, b in steps) + "</div>",
        unsafe_allow_html=True)
with right:
    st.markdown("### What the models found")
    personas = "<br>".join(f"<span style='color:{['#ec7a6c', '#e3c35a', '#7db4e6'][i]}'>●</span> {p['name']}: "
                           f"{p['size']} students, {p['at_risk_rate']:.0%} at risk" for i, p in enumerate(u3["personas"]))
    blocks = [
        ("Three personas, no labels needed", personas),
        ("Early-warning signal (ROC-AUC)", "<br>".join(f"{k}: <b>{v:.3f}</b>" for k, v in ew.items())),
        ("Ground truth recovered",
         f"HMM transition error max |Â − A| = <b>{u4['truth']['A_max_abs_error']:.3f}</b><br>"
         f"Factor analysis finds <b>{u3['factor_analysis']['n_factors_best']}</b> latent factors<br>"
         f"Scratch EM = sklearn (ARI <b>{u3['scratch_vs_sklearn']['gmm_ari']:.2f}</b>)"),
        ("Ensembles (Unit 5)",
         f"one tree <b>{u5auc('CART (scratch'):.3f}</b> → "
         f"random forest <b>{u5auc('Random Forest (scratch'):.3f}</b> CV AUC<br>"
         f"best: {u5['best_ensemble']} <b>{u5['compare_cv'][0]['auc_fold_mean']:.3f}</b> vs logistic "
         f"<b>{u5auc('Logistic'):.3f}</b>"),
    ]
    st.markdown('<div style="display:grid;gap:.55rem">' + "".join(
        f'<div class="ed-card"><b>{t}</b><div style="margin-top:.5rem;color:#c9cfc6;line-height:1.7">{b}</div></div>'
        for t, b in blocks) + "</div>", unsafe_allow_html=True)

st.write("")
checks = M["meta"]["checks"]
st.caption(f"{sum(checks.values())}/{len(checks)} self-checks pass · pipeline run {M['meta']['generated_at']} · "
           f"{sum(M['meta']['timings_sec'].values()):.0f}s on a laptop CPU · synthetic cohort calibrated on the real form responses")
