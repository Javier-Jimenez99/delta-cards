"""Delta Cards evaluation app."""

import os
import sys

import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evaluation_app.tab_dataset import render_dataset_tab
from evaluation_app.tab_inspector import render_inspector_tab

st.set_page_config(
    page_title="Delta Cards",
    page_icon="DC",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Delta Cards")

tab_inspector, tab_dataset = st.tabs(["Pipeline inspector", "Dataset"])

with tab_inspector:
    render_inspector_tab()

with tab_dataset:
    render_dataset_tab()
