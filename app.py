import streamlit as st

st.set_page_config(page_title="Research Dashboard", layout="wide")

PAPERS = {
    "LSFTL — Low-Resource MT with LoRA": "lsftl",
}

st.sidebar.title("Research Dashboard")
selected_paper_label = st.sidebar.selectbox("Select a paper:", list(PAPERS.keys()))
selected_paper_key = PAPERS[selected_paper_label]

if selected_paper_key == "lsftl":
    from papers.lsftl.dashboard import render
    render()