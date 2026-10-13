import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import C, SERIES, clean, metrics, nice, header, real, table

header("Data & cleaning", "Problem identification · dataset selection")
M = metrics()
df = clean()

st.markdown("""**Problem.** Academic difficulty is usually noticed only after grades have already dropped.
EarlyDrop asks whether survey indicators collected mid-semester (attendance, internal marks, study behaviour,
engagement, motivation, support) can flag students whose **end-of-semester SGPA will fall below 6.5**.""")

t1, t2, t3 = st.tabs(["Cleaning log", "Clean dataset", "Real responses vs synthetic"])
with t1:
    st.markdown("The raw export is cleaned by the **same code** for real and synthetic data.")
    for item in M["cleaning"]:
        extra = ""
        if item.get("values"):
            extra = f" → values `{item['values']}`"
        if item.get("columns"):
            extra = " → " + ", ".join(f"`{k}` ({v})" for k, v in item["columns"].items())
        st.markdown(f"- **{item['step']}**: {item['count']}{extra}")
    raw = pd.read_csv(C.SYN_FORM_CSV, dtype=str, keep_default_na=False)
    st.caption("Raw synthetic export (first rows, Google Form headers):")
    st.dataframe(raw.head(8), width="stretch")
with t2:
    st.markdown(f"**{len(df)} rows × {len(C.SNAKE) - 2 + 1} modelling columns** (48 answers + target), "
                f"0 missing after cleaning. The 15 selected features:")
    st.dataframe(table(M["unit1"]["describe"]), width="stretch")
    st.dataframe(df.head(20), width="stretch")
with t3:
    R = M["real"]
    if not R.get("available"):
        st.info("Real responses file not found.")
    else:
        p = R["profile"]
        st.markdown(f"""**{R['n']} real responses** collected through the live form. CGPA ranges from
**{p['cgpa_min']} to {p['cgpa_max']}** and total active backlogs = **{p['backlogs_total']}**, so every respondent
is a high performer and there is no outcome label. This is why the model is trained on a synthetic cohort, and why
the real answers are used to **calibrate** it.""")
        cal = pd.DataFrame(R["calibration"]).sort_values("real_mean")
        fig = go.Figure()
        fig.add_bar(y=[nice(i) for i in cal["item"]], x=cal["real_mean"], orientation="h", name="Real (15)",
                    marker_color=SERIES[0])
        fig.add_bar(y=[nice(i) for i in cal["item"]], x=cal["synthetic_high_mean"], orientation="h",
                    name="Synthetic, CGPA ≥ 8.2", marker_color=SERIES[2])
        fig.update_layout(barmode="group", height=900, title="Likert means: real vs calibrated synthetic",
                          xaxis_range=[1, 5], legend=dict(orientation="h", y=1.03))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Mean absolute gap: {R['calibration_mean_abs_gap']:.3f} Likert points.")
        st.subheader("Real responses scored by the trained Bayesian logistic model")
        st.dataframe(table(R["scores"]), width="stretch")
        r = real()
        if r is not None:
            with st.expander("Real cleaned responses"):
                st.dataframe(r.drop(columns=C.TEXT_FIELDS, errors="ignore"), width="stretch")
