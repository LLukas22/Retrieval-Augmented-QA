from typing import List, Dict, Any, Tuple, Optional
import streamlit as st
from ui.api_connector import get_api_connector,ApiConnector
from annotated_text import annotation,annotated_text
from haystack.schema import Answer,Document

def sidebar_footer()->None:
    connector = get_api_connector()
    versions = connector.versions()
    
    haystack = f"<small> (unknown)</small>"
    transformers = f"<small> (unknown)</small>"
    
    if "haystack" in versions:
        haystack = f"<small> (v{versions['haystack']})</small>"
        
    if "transformers" in versions:
        transformers = f"<small> (v{versions['transformers']})</small>"
         
    st.sidebar.markdown(
        f"""
    <style>
        a {{
            text-decoration: none;
        }}
        .haystack-footer {{
            text-align: center;
        }}
        .haystack-footer h4 {{
            margin: 0.1rem;
            padding:0;
        }}
        footer {{
            opacity: 0;
        }}
    </style>
    <div class="haystack-footer">
        <p>View on <a href="https://github.com/LLukas22/Retrieval-Augmented-QA">GitHub</a></p>
        <h4>Built with <a href="https://www.deepset.ai/haystack">Haystack</a>{haystack}</h4>
        <h4>and <a href="https://huggingface.co/">Transformers</a>{transformers}</h4>
    </div>
    """,
        unsafe_allow_html=True,
    )
       
def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value
        
        
def get_wikipedia_url_from_title(title:str)->str:
    url = "https://en.wikipedia.org/wiki/"
    return  f"{url}{title.replace(' ', '_')}"


def show_answer(answer:Answer,documents:List[Document]):
    highlight_color =  "#023020"
    score =  round(answer.score*100,2)
    st.write(f"### Answer:\"{answer.answer}\" (Score: {score}%)")
    
    offsets_in_context = answer.offsets_in_context[0]
    
    st.write("**Context:**")
    annotated_text(answer.context[:offsets_in_context.start],
        (answer.answer, f"ANSWER ({score}%)", highlight_color),
        answer.context[offsets_in_context.end:])
    
    title=None
    source=None
    split=None
    wiki_id=None
    
    if "title" in answer.meta:
        title = answer.meta["title"]
    if "source" in answer.meta:
        source = answer.meta["source"]
    if "_split_id" in answer.meta:
        split = answer.meta["_split_id"]
    if "wiki_id" in answer.meta:
        wiki_id = answer.meta["wiki_id"]
        
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
        
    with st.expander("Show full context"):
        document = next(doc for doc in documents if doc.id == answer.document_ids[0])
        offsets_in_document = answer.offsets_in_document[0]
        
        if title:
            st.markdown(f"### Title: {title}")
        
        annotated_text(document.content[:offsets_in_document.start],
        (answer.answer, f"ANSWER ({score}%)", highlight_color),
            document.content[offsets_in_document.end:])
        
    st.markdown(f"#### Source: {source_display}") 
    st.markdown('----')