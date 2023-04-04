from typing import List, Dict, Any, Tuple, Optional
import streamlit as st
from ui.api_connector import get_api_connector,ApiConnector


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