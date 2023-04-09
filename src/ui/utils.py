from typing import List, Dict, Any, Tuple, Optional
import streamlit as st
from ui.api_connector import get_api_connector
from annotated_text import annotation,annotated_text
from haystack.schema import Answer,Document
from schemas.chat import DefaultConfigResponse

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
    # highlight_color =  "#024d10#"#"#023020"
    # background_color = "#0f1212"
    score =  round(answer.score*100,2)
    st.write(f"### Answer:\"{answer.answer}\" (Score: {score}%)")
    
    offsets_in_context = answer.offsets_in_context[0]
    
    st.write("**Context:**")
    annotated_text(answer.context[:offsets_in_context.start],
        (answer.answer, f"ANSWER ({score}%)"),
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
        (answer.answer, f"ANSWER ({score}%)"),
            document.content[offsets_in_document.end:])
        
    st.markdown(f"#### Source: {source_display}") 
    st.markdown('----')
    
    
def set_default_generation_config(config:Optional[DefaultConfigResponse]):
    generation_config = config.config if config else {}
    if "temperature" in generation_config:
        set_state_if_absent("temperature",float(generation_config["temperature"]))
    else:
        set_state_if_absent("temperature",1.0)
        
    if "top_p" in generation_config:
        set_state_if_absent("top_p",float(generation_config["top_p"]))
    else:
        set_state_if_absent("top_p",1.0)
        
    if "max_new_tokens" in generation_config:
        set_state_if_absent("max_new_tokens",generation_config["max_new_tokens"])
    else:
         set_state_if_absent("max_new_tokens",256)
    
    if "top_k" in generation_config:
        set_state_if_absent("top_k",generation_config["top_k"])
    else:
        set_state_if_absent("top_k",50)
        
    if "repetition_penalty" in generation_config:
        set_state_if_absent("repetition_penalty",float(generation_config["repetition_penalty"]))
    else:
        set_state_if_absent("repetition_penalty",1.0)
        
        
def get_generation_config(temperature,top_p,max_new_tokens,repetition_penalty)->Dict[str,Any]:
    return {
             "temperature":temperature,
             "top_p":top_p,
             "max_new_tokens":max_new_tokens,
             "top_k":st.session_state.top_k,
             "repetition_penalty":repetition_penalty
        }