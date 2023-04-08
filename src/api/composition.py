from dependency_injector import containers, providers

from .routers.utils import RequestLimiter
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import TransformersReader,EmbeddingRetriever,BM25Retriever
from .pipelines import SearchPipeline, ExtractiveQAPipeline
from .routers import HealthRouter,PipelineRouter,QueryRouter,DocumentRouter,ChatRouter
from .chat_models import adapter_factory
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
        context_window_size=150,
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
    
    query_router = providers.Factory(
        QueryRouter,
        document_store=document_store,
        search_pipeline=search_pipeline,
        extractive_qa_pipeline=extractive_qa_pipeline,
        limiter=limiter,
        embedding_retriever=embedding_retriever,
    )
    
    document_router = providers.Factory(
        DocumentRouter,
        document_store=document_store
    )
    
    chat_limiter=providers.Singleton(
        RequestLimiter,
        limit=2
    )
        
    chatmodel=providers.Singleton(
        adapter_factory,
        configuration=config
    )
    
    chat_router = providers.Factory(
        ChatRouter,
        chat_model=chatmodel,
        limiter=chat_limiter
    )
    
    