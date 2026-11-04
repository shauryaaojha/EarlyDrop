"""EarlyDrop dashboard entry point:  streamlit run app/streamlit_app.py"""
import streamlit as st

st.set_page_config(page_title="EarlyDrop · Dropout-risk ML", page_icon="📉", layout="wide",
                   initial_sidebar_state="expanded")

import fx  # noqa: E402

PAGES = {
    "": [
        st.Page("views/home.py", title="Overview", icon=":material/blur_on:", default=True),
        st.Page("views/data.py", title="Data & cleaning", icon=":material/dataset:", url_path="data"),
    ],
    "PBL-I · Units 1–2": [
        st.Page("views/unit1.py", title="Statistics & Bayes", icon=":material/query_stats:", url_path="unit-1"),
        st.Page("views/unit2.py", title="Linear models", icon=":material/trending_up:", url_path="unit-2"),
    ],
    "PBL-II · Units 3–4": [
        st.Page("views/unit3.py", title="Clustering & PCA", icon=":material/bubble_chart:", url_path="unit-3"),
        st.Page("views/unit4.py", title="Hidden Markov", icon=":material/timeline:", url_path="unit-4"),
    ],
    "PBL-III · Unit 5": [
        st.Page("views/unit5.py", title="Ensembles", icon=":material/forest:", url_path="unit-5"),
    ],
    "Early warning": [
        st.Page("views/predict.py", title="Predict a student", icon=":material/person_search:", url_path="predict"),
    ],
}
pg = st.navigation(PAGES)
with st.sidebar:
    st.markdown('<div class="ed-sidefoot"><div class="k">Next review · PBL-III</div><div class="d">18 Nov 2026</div>'
                '<div class="team"><span title="Yash Arun Mulajkar">YM</span><span title="Bignesh Sahu">BS</span>'
                '<span title="Shaurya Ojha">SO</span><span title="Aditya Bhutada">AB</span></div></div>',
                unsafe_allow_html=True)
# motion stays on; add ?static to the URL to turn it off (slow projector PCs)
fx.inject("home" if pg.url_path == "" else pg.url_path, "static" not in st.query_params)
pg.run()
