from typing import List, Dict, Any, Tuple, Optional

import logging
from time import sleep
from ui.config import API_ENDPOINT
import httpx
from httpx._client import UseClientDefault
import streamlit as st
import pydantic
from schemas.health import HealthResponse
from schemas.query import SearchResponse,QueryRequest,QAResponse
from haystack.schema import Document,Answer,Span

HEALTH = "health"
QUERY = "qa"
UPDATE_EMBEDDINGS = "update_embeddings"

httpx_client:httpx.Client=None

def sidebar_footer()->None:
    hs_version = f"<small> (unknown)</small>"
    try:
        hs_version = f"<small> (v{haystack_version()})</small>"
    except Exception:
        pass
    
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
        <h4>Built with <a href="https://www.deepset.ai/haystack">Haystack</a>{hs_version}</h4>
    </div>
    """,
        unsafe_allow_html=True,
    )
    


def _get_client()->httpx.Client:
    global httpx_client
    if httpx_client:
        return httpx_client
    timeout = httpx.Timeout(5, read=30, write=30)
    httpx_client = httpx.Client(base_url=API_ENDPOINT(),timeout=timeout)
    return httpx_client
    
def get(url:str,timeout:int|UseClientDefault=UseClientDefault())->Optional[httpx.Response]:
    try:
        client=_get_client()
        response = client.get(url,timeout=timeout)
        response.raise_for_status()
        return response
    except Exception as e:
        logging.exception(e)
    return None

def post(url:str,json:Any|None=None,timeout:int|UseClientDefault=UseClientDefault())->Optional[httpx.Response]:
    try:
        client=_get_client()
        response = client.post(url,json=json,timeout=timeout)
        response.raise_for_status()
        return response
    except Exception as e:
        logging.exception(e)
    return None


def system_info()->Optional[HealthResponse]:
    url= "/health"
    try:
        result = get(url)
        if result:
            return pydantic.parse_obj_as(type_=HealthResponse,obj=result.json()) 
    except Exception as e:
        logging.exception(e)
    return None    
    

def haystack_status()->Optional[dict[str,bool]]:
    url = f"/initialized"
    try:
        result = get(url)
        if result:
            return result.json()
    except Exception as e:
        logging.exception(e)
        sleep(1)  # To avoid spamming a non-existing endpoint at startup
    return None

def update_embeddings()->bool:
    url = "/update_embeddings"
    try:
        result = post(url,timeout=None)
        if result:
            return True    
    except Exception as e:
        logging.exception(e)
    return False

@st.cache
def haystack_version():
    """
    Get the Haystack version from the REST API
    """
    url = f"/hs_version"
    try:
        result = get(url)
        if result:
            return result.json()["hs_version"]
    except Exception as e:
        logging.exception(e)
    return None

def query(query, filters={}, top_k_reader=5, top_k_retriever=5) -> Optional[Tuple[QAResponse,Dict[str,Any]]]:
    """
    Send a query to the REST API and parse the answer.
    Returns both a ready-to-use representation of the results and the raw JSON.
    """
    url = f"/qa"
    
    request = QueryRequest(query=query)
    request.params = {
        "filters": filters,
        "DPR":{ "top_k": top_k_retriever},
        "BM25":{ "top_k": top_k_retriever},
        "Reader":{ "top_k": top_k_reader}
        }
    response_raw = post(url,json=request.dict())
    
    if response_raw.status_code >= 400 and response_raw.status_code != 503:
        raise Exception(f"{vars(response_raw)}")

    response = response_raw.json()
    if "errors" in response:
        raise Exception(", ".join(response["errors"]))

    parsed_response = pydantic.parse_obj_as(type_=QAResponse,obj=response)
    return parsed_response,response

def search(query, filters={}, top_k=5)->Optional[Tuple[SearchResponse,Dict[str,Any]]]:
    url="/search"
    request=QueryRequest(query=query)
    request.params = {
        "filters": filters,
        "DPR":{ "top_k": top_k},
        "BM25":{ "top_k": top_k},
        }
    response_raw = post(url,json=request.dict())
    if response_raw.status_code >= 400 and response_raw.status_code != 503:
        raise Exception(f"{vars(response_raw)}")

    response = response_raw.json()
    if "errors" in response:
        raise Exception(", ".join(response["errors"]))

    parsed_response = pydantic.parse_obj_as(type_=SearchResponse,obj=response)
    return parsed_response,response
    
    
def get_backlink(document:Document) -> Tuple[Optional[str], Optional[str]]:
    meta = document.meta
    if meta.get("url", None) and meta.get("title", None):
        return meta.get("url"), meta.get("title")
    return None, None