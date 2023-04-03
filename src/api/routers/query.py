from typing import List, Dict, Any

from collections.abc import Mapping
import time
import json
from haystack.document_stores import ElasticsearchDocumentStore
from ..pipelines import SearchPipeline, ExtractiveQAPipeline
from .utils import RequestLimiter
from ._router import BaseRouter
from schemas.query import QueryRequest, QAResponse, SearchResponse, ReindexRequest
from haystack.nodes import EmbeddingRetriever

class QueryRouter(BaseRouter):
    def __init__(self,document_store:ElasticsearchDocumentStore,search_pipeline:SearchPipeline,extractive_qa_pipeline:ExtractiveQAPipeline,limiter:RequestLimiter,embedding_retriever:EmbeddingRetriever):
        super().__init__("/query")
        self.document_store = document_store
        self.search_pipeline = search_pipeline
        self.extractive_qa_pipeline = extractive_qa_pipeline
        self.limiter = limiter
        self.embedding_retriever = embedding_retriever
        
        self.router.add_api_route("/qa", self.qa, methods=["POST"], response_model=QAResponse, response_model_exclude_none=True)
        self.router.add_api_route("/search", self.search, methods=["POST"], response_model=SearchResponse, response_model_exclude_none=True)
        self.router.add_api_route("/reindex", self.reindex, methods=["POST"], response_model=bool)
        
    def qa(self,request: QueryRequest):
        """
        This endpoint receives the question as a string and allows the requester to set
        additional parameters that will be passed on to the Haystack pipeline.
        """
        with self.limiter.run():
            result = self._process_request(self.extractive_qa_pipeline, request)
            # Ensure answers and documents exist, even if they're empty lists
            if not "documents" in result:
                result["documents"] = []
            if not "answers" in result:
                result["answers"] = []
            return result
        
    def search(self, request: QueryRequest):
        with self.limiter.run():
            result = self._process_request(self.search_pipeline, request)
            # Ensure answers and documents exist, even if they're empty lists
            if not "documents" in result:
                result["documents"] = []
            return result


    def reindex(self, request:ReindexRequest)->bool:
        try:
            self.document_store.update_embeddings(
                retriever= self.embedding_retriever,
                update_existing_embeddings = request.update_existing_embeddings,
                batch_size = request.batch_size
            )
        except Exception as e:
            self.logger.exception(e)
            return  False
        return True


    def _process_request(self, pipeline, request) -> Dict[str, Any]:
        start_time = time.time()

        params = request.params or {}

        # format global, top-level filters (e.g. "params": {"filters": {"name": ["some"]}})
        if "filters" in params.keys():
            params["filters"] = self._format_filters(params["filters"])

        # format targeted node filters (e.g. "params": {"Retriever": {"filters": {"value"}}})
        for key in params.keys():
            if isinstance(params[key], Mapping) and "filters" in params[key].keys():
                params[key]["filters"] = self._format_filters(params[key]["filters"])

        result = pipeline.run(query=request.query, params=params, debug=request.debug)
        
        self.logger.info(
            json.dumps({"request": request, "response": result, "time": f"{(time.time() - start_time):.2f}"}, default=str)
        )
        return result
    
    def _format_filters(self, filters):
        """
        Adjust filters to compliant format:
        Put filter values into a list and remove filters with null value.
        """
        new_filters = {}
        if filters is None:
            self.logger.warning(
                f"Request with deprecated filter format ('\"filters\": null'). "
                f"Remove empty filters from params to be compliant with future versions"
            )
        else:
            for key, values in filters.items():
                if values is None:
                    self.logger.warning(
                        f"Request with deprecated filter format ('{key}: null'). "
                        f"Remove null values from filters to be compliant with future versions"
                    )
                    continue

                if not isinstance(values, list):
                    self.logger.warning(
                        f"Request with deprecated filter format ('{key}': {values}). "
                        f"Change to '{key}':[{values}]' to be compliant with future versions"
                    )
                    values = [values]

                new_filters[key] = values
        return new_filters
