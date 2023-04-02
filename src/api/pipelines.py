from typing import Dict, List, Optional, Any, Set, Tuple, Union
from haystack import Pipeline
from haystack.nodes import TransformersReader,EmbeddingRetriever,BM25Retriever,JoinDocuments
from haystack.schema import MultiLabel, Document
from .custom_nodes.tagging_nodes import DocumentTaggingNode

class CustomPipeline():
    """
    Warp the Haystack Pipeline class to make it compatible with dependency injector
    """
    def __init__(self) -> None:
        self.pipeline = Pipeline()
        
    def run(  # type: ignore
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[Union[dict, List[dict]]] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ):      
        return self.pipeline.run(query, file_paths, labels, documents, meta, params, debug)
       
    def run_batch(  # type: ignore
        self,
        queries: Optional[List[str]] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[Union[MultiLabel, List[MultiLabel]]] = None,
        documents: Optional[Union[List[Document], List[List[Document]]]] = None,
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ):
        return self.pipeline.run_batch(queries, file_paths, labels, documents, meta, params, debug)
    
class SearchPipeline(CustomPipeline):
    def __init__(self,bm25_retreiver:BM25Retriever,embedding_retriever:EmbeddingRetriever) -> None:
        super().__init__()
        self.pipeline.add_node(component=embedding_retriever,name="Embedding",inputs=["Query"])
        self.pipeline.add_node(component=DocumentTaggingNode(name="retriever",value="Embedding"),name="DPR_Meta_Tagger",inputs=["Embedding"])
    
        self.pipeline.add_node(component=bm25_retreiver,name="BM25",inputs=["Query"])
        self.pipeline.add_node(component=DocumentTaggingNode(name="retriever",value="BM25"),name="BM25_Meta_Tagger",inputs=["BM25"])
    
        self.pipeline.add_node(component=JoinDocuments(join_mode="concatenate"),name="Join",inputs=["DPR_Meta_Tagger","BM25_Meta_Tagger"])
    
     
class ExtractiveQAPipeline(SearchPipeline):
    def __init__(self,bm25_retreiver:BM25Retriever,embedding_retriever:EmbeddingRetriever,reader:TransformersReader) -> None:
        super().__init__(bm25_retreiver=bm25_retreiver,embedding_retriever=embedding_retriever)
        self.pipeline.add_node(component=reader, name="Reader", inputs=["Join"])