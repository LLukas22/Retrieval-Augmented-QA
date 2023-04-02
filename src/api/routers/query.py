from typing import List, Dict, Any

from collections.abc import Mapping
import logging
import time
import json

from fastapi import FastAPI, APIRouter, BackgroundTasks
from haystack.document_stores import BaseDocumentStore
from ..schemas.base import Document

from ..schemas.query import QueryRequest, QueryResponse, SearchResponse, SearchRequest
from ..utils import App,DocumentStore,Limiter
from ..pipelines import DPR_Retriever,Search_Pipeline,Extractive_QA_Pipeline

logger = logging.getLogger("haystack")

router = APIRouter()
app: FastAPI = App()
concurrency_limiter=Limiter()

@router.post("/qa", response_model=QueryResponse, response_model_exclude_none=True)
def query(request: QueryRequest):
    """
    This endpoint receives the question as a string and allows the requester to set
    additional parameters that will be passed on to the Haystack pipeline.
    """
    with concurrency_limiter.run():
        result = _process_request(Extractive_QA_Pipeline(), request)
        # Ensure answers and documents exist, even if they're empty lists
        if not "documents" in result:
            result["documents"] = []
        if not "answers" in result:
            result["answers"] = []
        return result

@router.post("/search", response_model=SearchResponse, response_model_exclude_none=True)
def search(request: SearchRequest):
    with concurrency_limiter.run():
        result = _process_request(Search_Pipeline(), request)
        # Ensure answers and documents exist, even if they're empty lists
        if not "documents" in result:
            result["documents"] = []
        return result

@router.post("/update_embeddings")
def index()->bool:
    document_store = DocumentStore()
    document_store.update_embeddings(DPR_Retriever())
    return True

def _process_request(pipeline, request) -> Dict[str, Any]:
    start_time = time.time()

    params = request.params or {}

    # format global, top-level filters (e.g. "params": {"filters": {"name": ["some"]}})
    if "filters" in params.keys():
        params["filters"] = _format_filters(params["filters"])

    # format targeted node filters (e.g. "params": {"Retriever": {"filters": {"value"}}})
    for key in params.keys():
        if isinstance(params[key], Mapping) and "filters" in params[key].keys():
            params[key]["filters"] = _format_filters(params[key]["filters"])

    result = pipeline.run(query=request.query, params=params, debug=request.debug)
    
    logger.info(
        json.dumps({"request": request, "response": result, "time": f"{(time.time() - start_time):.2f}"}, default=str)
    )
    return result


def _format_filters(filters):
    """
    Adjust filters to compliant format:
    Put filter values into a list and remove filters with null value.
    """
    new_filters = {}
    if filters is None:
        logger.warning(
            f"Request with deprecated filter format ('\"filters\": null'). "
            f"Remove empty filters from params to be compliant with future versions"
        )
    else:
        for key, values in filters.items():
            if values is None:
                logger.warning(
                    f"Request with deprecated filter format ('{key}: null'). "
                    f"Remove null values from filters to be compliant with future versions"
                )
                continue

            if not isinstance(values, list):
                logger.warning(
                    f"Request with deprecated filter format ('{key}': {values}). "
                    f"Change to '{key}':[{values}]' to be compliant with future versions"
                )
                values = [values]

            new_filters[key] = values
    return new_filters
