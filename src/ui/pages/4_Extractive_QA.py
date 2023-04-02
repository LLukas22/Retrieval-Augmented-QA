import os

import logging
from json import JSONDecodeError

import streamlit as st
from annotated_text import annotation,annotated_text
from markdown import markdown
import time

from implementation.ui.utils import haystack_status,query,get_backlink,sidebar_footer

# Sliders
QUESTION_TAG = "DEFAULT_QUESTION_AT_STARTUP"
DEFAULT_DOCS_TAG = "DEFAULT_DOCS_FROM_RETRIEVER"
DEFAULT_NUMBER_TAG= "DEFAULT_NUMBER_OF_ANSWERS"

def render():
    st.set_page_config(page_title="➡️❓Extractive QA",layout="wide")
    
    def set_state_if_absent(key, value):
        if key not in st.session_state:
            st.session_state[key] = value
    
    set_state_if_absent("question","Who is the mother of dragons?")
    set_state_if_absent("results", None)
    set_state_if_absent("raw_json", None)
 
    # Small callback to reset the interface in case the text of the question changes
    def reset_results(*args):
        st.session_state.answer = None
        st.session_state.results = None
        st.session_state.raw_json = None

    # Title
    st.title("➡️❓Extractive QA")
    st.sidebar.header("Extractive QA - Game of thrones")
    st.markdown(
        """
    This demo takes its data from a scraped game of thrones wiki.

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
    top_k_retriever = st.sidebar.slider(
        "Max. number of documents from retriever",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        on_change=reset_results,
    )
    debug = st.sidebar.checkbox("Show debug info")

    sidebar_footer()
    
    # Search bar
    question = st.text_input(
        value=st.session_state["question"],
        max_chars=100,
        on_change=reset_results,
        label="question",
        label_visibility="hidden",
    )
    # Run button
    run_pressed = st.button("Run")
    run_query = (
        run_pressed or question != st.session_state["question"]
    )

    # Check the connection
    with st.spinner("⌛️ &nbsp;&nbsp; Haystack is starting..."):
        if not haystack_status():
            st.error("🚫 &nbsp;&nbsp; Connection Error. Is Haystack running?")
            run_query = False
            reset_results()

    # Get results for query
    if run_query and question:
        reset_results()
        st.session_state["question"] = question
        with st.spinner("🧠 &nbsp;&nbsp; Performing neural search on documents..."):
            try:
                st.session_state.results, st.session_state.raw_json = query(
                    question, top_k_reader=top_k_reader, top_k_retriever=top_k_retriever
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
        if len(st.session_state.results.answers) > 0:
            highlight_color =  "#023020"
            st.write("## Results:")
            for count, answer in enumerate(st.session_state.results.answers):
                document = next(doc for doc in st.session_state.results.documents if doc.id == answer.document_id)
                text = document.content
                offsets_in_document = answer.offsets_in_document[0]
                
                annotated_text(
                    text[:offsets_in_document.start],
                    (answer.answer, "ANSWER", highlight_color),
                    text[offsets_in_document.end:]
                )
                
                source = ""
                url, title = get_backlink(document)
                if url and title:
                    source = f"[{title}]({url})"
                else:
                    source = f"{document.meta['name']}"
                st.markdown(f"**Relevance:** {round(answer.score*100,2)} -  **Source:** {source} - **Retriever:** {document.meta['retriever']}") 
                st.markdown('----')
        else:
            st.write("###🤔 Found no Answers. Is your question related to documents in the database?")
             
        if debug:
            st.subheader("REST API JSON response")
            st.write(st.session_state.raw_json)
                      
render()