import os
import sys
from pathlib import Path
#Ensure the  modules are in the path
root = str(Path(__file__).parent.parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)
    
from typing import List
import logging
from json import JSONDecodeError

import streamlit as st

from ui.utils import show_answer,sidebar_footer, set_state_if_absent
from ui.api_connector import get_api_connector
from haystack.schema import Answer,Document

def render():
    st.set_page_config(page_title="❓Extractive QA",layout="wide")
    
    connector = get_api_connector()
    set_state_if_absent("qa_query","")
    set_state_if_absent("qa_results", None)
 
    # Small callback to reset the interface in case the text of the question changes
    def reset_results(*args):
        st.session_state.qa_answer = None
        st.session_state.qa_results = None

    # Title
    st.title("❓Extractive QA")
    st.sidebar.header("❓Extractive QA")
    st.markdown(
        """
    This demo searches the documents in the document store.
    
    Ask any question on this topic and see if Haystack can find the correct answer to your query!
    """,
        unsafe_allow_html=True,
    )
    
    # Sidebar
    st.sidebar.header("Options")
    top_k_reader = st.sidebar.slider(
        "Max. number of answers",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        on_change=reset_results,
    )
    top_k_retrievers = st.sidebar.slider(
        "Max. number of documents from retriever",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        on_change=reset_results,
    )
    
    sidebar_footer()
    
    with st.spinner("⌛️ &nbsp;&nbsp; Api is starting..."):
        if not connector.healtcheck():
            st.error("🚫 &nbsp;&nbsp; Connection Error. Is Haystack running?")
            run_query = False
            reset_results() 
            
    # Search bar
    form = st.form(key="qa_form")
    form.write("### 📝 Enter your query")
    query = form.text_input(
        value=st.session_state["qa_query"],
        max_chars=100,
        label="qa_query",
        label_visibility="hidden",
    )
    # Run button
    run_pressed = form.form_submit_button("Run")
    
    # Get results for query
    if run_pressed and query != "" and query != st.session_state.qa_query:
        reset_results()
        st.session_state.qa_query = query
        with st.spinner("🧠 &nbsp;&nbsp; Performing neural search on documents..."):
            try:
                st.session_state.qa_results = connector.qa(
                    query, top_k_reader=top_k_reader,  top_k_retrievers= top_k_retrievers
                )
            except JSONDecodeError:
                st.error("👓 &nbsp;&nbsp; An error occurred reading the results. Is the document store working?")
                return
            except Exception as e:
                logging.exception(e)
                if "The server is busy processing requests" in str(e) or "503" in str(e):
                    st.error("🧑‍🌾 &nbsp;&nbsp; All our workers are busy! Try again later.")
                else:
                    st.error("🐞 &nbsp;&nbsp; An error occurred during the request.\n")
                    st.exception(e)
                return


    if st.session_state.qa_results:
        answers = st.session_state.qa_results.answers
        documents = st.session_state.qa_results.documents
        if len(answers) > 0:
            st.write("## Results:")
            for answer in answers:
                show_answer(answer,documents)
        else:
            st.write("###🤔 Found no Answers. Is your question related to documents in the database?")
                                
render()