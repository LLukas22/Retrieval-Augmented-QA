import os
import sys
from pathlib import Path

root = str(Path(__file__).parent.parent)

if root not in sys.path:
    sys.path.insert(0, root)

import streamlit as st
from ui.utils import sidebar_footer

st.set_page_config(
    page_title="Introduction",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")


st.sidebar.success("Select a demo above.")
sidebar_footer()
st.markdown(
    """
    **👈 Select a demo from the sidebar** 
    """
)
