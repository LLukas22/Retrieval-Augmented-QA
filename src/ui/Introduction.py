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
message = os.getenv("WELCOME_MESSAGE", "This demo was initialized with about 500.000 english wikipedia articles from April, 1st, 2023. Fell free to ask the system about any topic you like.\n\n ⚠️CAUTION: If offline models are used no safety layers are in place. If you ask the system about an offensive topic it will answer you, even if the answer is immoral!⚠️")
st.write("# Welcome the Retrieval Augmented QA-Demo! 👋")
st.write("This is a demo of a transformer augmented document retrieval and qestion-answering pipeline.")
st.write(message)

st.sidebar.success("Select a demo above.")
sidebar_footer()
st.markdown(
    """
    **👈 Select a demo from the sidebar** 
    """
)
