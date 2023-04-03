import streamlit as st
import time
from ui.utils import update_embeddings,sidebar_footer

st.set_page_config(page_title="Embeddings", page_icon="✳️",layout="wide")

st.markdown("# Embeddings")
sidebar_footer()
st.markdown("## Update Embeddings")
st.text("Recalculate all embeddings in the database.⚠️ If the node is not GPU accelerated this could take some time!")
should_update_embeddings = st.button("↻ Update Embeddings")

result=None
with st.spinner("⌛️ &nbsp;&nbsp; Embeddings are updating..."):
    if should_update_embeddings:
        result = update_embeddings()

if result:
    st.write("Updated Embeddings sucessfully!")