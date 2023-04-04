import os
import sys
from pathlib import Path
#Ensure the  modules are in the path
root = str(Path(__file__).parent.parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)

import logging
from json import JSONDecodeError

import streamlit as st
from utils import sidebar_footer, set_state_if_absent, get_wikipedia_url_from_title
from ui.api_connector import get_api_connector
from haystack.schema import Document

def write_document(document:Document,column):
    title=None
    source=None
    split=None
    wiki_id=None
    score = round(document.score*100,2)
    
    if "title" in document.meta:
        title = document.meta["title"]
    if "source" in document.meta:
        source = document.meta["source"]
    if "_split_id" in document.meta:
        split = document.meta["_split_id"]
    if "wiki_id" in document.meta:
        wiki_id = document.meta["wiki_id"]
        
    source_display="Unknown"
    if title:
        source_display=title
        if split:
            source_display+=f" (Split: {split+1})"
    
    if source:
        source_display=f"[{source_display}]({source})"
    
    if wiki_id  and title:    
        wikipedia_url = get_wikipedia_url_from_title(title)
        source_display=f"[{source_display}]({wikipedia_url})"
        
    if title:
        column.markdown(f"### Title: {title}")
            
    column.write(document.content)
    
    column.markdown(f"**Relevance:** {score}% - **Source:** {source_display}") 
    column.markdown('----')
    
def render():
    
    connetor = get_api_connector()
    
    st.set_page_config(page_title="🔎Semantic Search",layout="wide")
    
    # Title
    st.title("🔎Semantic Search")
    st.sidebar.header("🔎Semantic Search")
    st.markdown("""
    This demo searches the documents in the document store.

    Define a query and the most relevant documents will be returned and shown below.
    
    The results will be divided into Lexical(BM25) and Semantic(Embedding) results.
    """)
    
    set_state_if_absent("search_query","")
    set_state_if_absent("search_results", None)
    
    def reset_results(*args):
        st.session_state.search_results = None
        
    st.sidebar.header("Options")
    top_k = st.sidebar.slider(
        "Max. number of answers",
        min_value=2,
        max_value=20,
        value=6,
        step=2,
        on_change=reset_results,
    )
    
    sidebar_footer()
    
    query = st.text_input(
        value=st.session_state.search_query,
        max_chars=150,
        on_change=reset_results,
        label="search_query",
        label_visibility="hidden",
    )
    
    # Run button
    run_pressed = st.button("Run")
    run_query = (
        run_pressed or query != st.session_state.search_query
    )
    
    # Check the connection
    with st.spinner("⌛️ &nbsp;&nbsp; API is starting..."):
        if not connetor.healtcheck():
            st.error("🚫 &nbsp;&nbsp; Connection Error. Is the API running?")
            run_query = False
            reset_results()
            
    # Get results for query
    if run_query and query != "":
        reset_results()
        st.session_state["query"] = query
        with st.spinner("🧠 &nbsp;&nbsp; Performing semantic search on documents..."):
            try:
                st.session_state.search_results = connetor.search(
                    query, top_k=top_k
                )
            except JSONDecodeError:
                st.error("👁️ 👄 👁️ &nbsp;&nbsp; An error occurred reading the results. Is the document store working?")
                return
            except Exception as e:
                logging.exception(e)
                if "The server is busy processing requests" in str(e) or "503" in str(e):
                    st.error("🧑‍🌾 &nbsp;&nbsp; All our workers are busy! Try again later.")
                else:
                    st.error("🐞 &nbsp;&nbsp; An error occurred during the request.\n")
                    st.exception(e)
                return
            
        if st.session_state.search_results:
            if len(st.session_state.search_results.documents) > 0:
                docs = st.session_state.search_results.documents
                if 'retriever' in docs[0].meta:
                    bm_25_docs = [doc for doc in docs if doc.meta['retriever'] == 'BM25']
                    semantic_docs = [doc for doc in docs if doc.meta['retriever'] != 'BM25']
                    semantic_column, lexical_column = st.columns(2,gap="medium")
                    semantic_column.header("Semantic Results")
                    lexical_column.header("Lexical Results")
                    
                    for doc in semantic_docs:
                        write_document(doc,semantic_column)
                    for doc in bm_25_docs:
                        write_document(doc,lexical_column)
                    
                else:   
                    results_column = st.columns(1,gap="small")
                    results_column.header("Results")
                    for document in docs:
                        write_document(document,results_column)
            else:
                st.write("### 🤔 Found no Answers. Is your question related to documents in the database?")                    
render()
    
    