import os

import logging
from json import JSONDecodeError

import streamlit as st
from annotated_text import annotation,annotated_text
from markdown import markdown
import time

from utils import search,sidebar_footer,haystack_status,get_backlink

def render():
    st.set_page_config(page_title="🔎Semantic Search",layout="wide")
    
    def set_state_if_absent(key, value):
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Title
    st.title("🔎Semantic Search")
    st.sidebar.header("Semantic Search - Game of thrones")
    st.markdown(
        """
    This demo takes its data from a scraped game of thrones wiki.

    Define a querry and Haystack will find the most relevant documents for you!
    
    """,
        unsafe_allow_html=True,
    )
    
    set_state_if_absent("query","John Snow family")
    set_state_if_absent("results", None)
    set_state_if_absent("raw_json", None)
    
    def reset_results(*args):
        st.session_state.answer = None
        st.session_state.results = None
        st.session_state.raw_json = None
        
    st.sidebar.header("Options")
    top_k = st.sidebar.slider(
        "Max. number of answers",
        min_value=2,
        max_value=20,
        value=6,
        step=2,
        on_change=reset_results,
    )
    
    debug = st.sidebar.checkbox("Show debug info")

    sidebar_footer()
    
    query = st.text_input(
        value=st.session_state["query"],
        max_chars=100,
        on_change=reset_results,
        label="query",
        label_visibility="hidden",
    )
    
    # Run button
    run_pressed = st.button("Run")
    run_query = (
        run_pressed or query != st.session_state["query"]
    )
    
    # Check the connection
    with st.spinner("⌛️ &nbsp;&nbsp; Haystack is starting..."):
        if not haystack_status():
            st.error("🚫 &nbsp;&nbsp; Connection Error. Is Haystack running?")
            run_query = False
            reset_results()
            
    # Get results for query
    if run_query and query:
        reset_results()
        st.session_state["query"] = query
        with st.spinner("🧠 &nbsp;&nbsp; Performing semantic search on documents..."):
            try:
                st.session_state.results, st.session_state.raw_json = search(
                    query, top_k=top_k
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
            
        if st.session_state.results:
            if len(st.session_state.results.documents) > 0:
                st.write("## Results:")
                for document in st.session_state.results.documents:
                    text = document.content
                    st.write(text)
                    
                    source = ""
                    url, title = get_backlink(document)
                    if url and title:
                        source = f"[{title}]({url})"
                    else:
                        source = f"{document.meta['name']}"
                    st.markdown(f"**Relevance:** {round(document.score*100,2)}% - **Source:** {source} - **Retriever:** {document.meta['retriever']}") 
                    st.markdown('----')
            else:
                st.write("###🤔 Found no Answers. Is your question related to documents in the database?")
             
        if debug:
            st.subheader("REST API JSON response")
            st.write(st.session_state.raw_json)
                    
render()
    
    