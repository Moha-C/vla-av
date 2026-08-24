from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
SNAPSHOT = APP_DIR / "dashboard_snapshot.html"

st.set_page_config(
    page_title="VLA-AV SimLingo World",
    page_icon="VLA",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      header[data-testid="stHeader"], footer { display: none; }
      .stApp { background: #05070a; }
      .block-container {
        max-width: 100%;
        padding: 0;
      }
      iframe[title="streamlit.components.v1.html"] {
        border: 0;
        background: #05070a;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

if not SNAPSHOT.is_file():
    st.error(
        "Dashboard snapshot missing. Run: "
        "python scripts/export_streamlit_dashboard.py"
    )
    st.stop()

st.iframe(SNAPSHOT, width="stretch", height="content", tab_index=0)
