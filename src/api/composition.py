from dependency_injector import containers, providers

from .routers.utils import RequestLimiter
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import TransformersReader,EmbeddingRetriever,BM25Retriever
from .pipelines import SearchPipeline, ExtractiveQAPipeline
from .routers import HealthRouter,PipelineRouter
class Container(containers.DeclarativeContainer):

    config = providers.Configuration()

    limiter=providers.Singleton(
        RequestLimiter,
        limit=config.concurency_limit
    )
    
    document_store = providers.Singleton(
        ElasticsearchDocumentStore,
        host=config.elasticsearch_host,
        port=config.elasticsearch_port,
        username=config.elasticsearch_username,
        password=config.elasticsearch_password,
        embedding_dim=config.embedding_dim,
        similarity=config.similarity,
    )
    
    bm25_retriever = providers.Singleton(
        BM25Retriever,
        document_store=document_store
    )
    
    embedding_retriever = providers.Singleton(
        EmbeddingRetriever,
        embedding_model=config.embedding_model,
        document_store=document_store,
        use_gpu=config.use_gpu,
        use_auth_token=config.hf_token,
    )
    
    qa_reader = providers.Singleton(
        TransformersReader,
        model_name_or_path = config.extractive_qa_model,
        use_gpu=config.use_gpu,
        use_auth_token=config.hf_token,
        max_seq_len=512,
    )
    
    search_pipeline = providers.Singleton(
        SearchPipeline,
        bm25_retreiver=bm25_retriever,
        embedding_retriever=embedding_retriever,
    )
    
    extractive_qa_pipeline = providers.Singleton(
        ExtractiveQAPipeline,
        bm25_retreiver=bm25_retriever,
        embedding_retriever=embedding_retriever,
        reader=qa_reader,
    )
    
    health_router = providers.Factory(
        HealthRouter
    )
    
    pipeline_router = providers.Factory(
        PipelineRouter,
        search_pipeline=search_pipeline,
        extractive_qa_pipeline=extractive_qa_pipeline,
    )