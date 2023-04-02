import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(),"src"))

import streamlit as st
from implementation.ui.utils import sidebar_footer
st.set_page_config(
    page_title="Welcome",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")


st.sidebar.success("Select a demo above.")
sidebar_footer()
st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    """
)