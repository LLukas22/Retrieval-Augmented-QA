import os
import sys
from pathlib import Path

root = str(Path(__file__).parent.parent)

if root not in sys.path:
    sys.path.insert(0, root)
    
import logging
import uvicorn
import transformers

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger(__name__)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from dependency_injector.wiring import Provide, inject

from api.composition import Container
from api.routers import HealthRouter,PipelineRouter,QueryRouter,DocumentRouter,ChatRouter
from api.errors.http_error import http_error_handler

from fastapi import FastAPI, HTTPException, APIRouter
from starlette.middleware.cors import CORSMiddleware

def parse_bool(value:str)->bool:
    if isinstance(value, bool):
        return value
    elif  isinstance(value, str):
        return value.lower() in ("yes", "true", "t", "1")
    else:
        return bool(value)

def parse_chatmodel(value:str)->bool:
    value = value.upper()
    if value not in ("GPU","CPU","OPENAI"):
        raise ValueError("Invalid value for CHATMODEL. Valid values are: GPU, CPU, OPENAI")
    return value
    
@inject
def App(
    container:Container,
    health_router:HealthRouter=Provide[Container.health_router],
    pipeline_router:PipelineRouter=Provide[Container.pipeline_router],
    document_router:DocumentRouter=Provide[Container.document_router],
    query_router:QueryRouter=Provide[Container.query_router],
    chat_router:ChatRouter=Provide[Container.chat_router],
    ):
    
    from haystack import __version__ as haystack_version
    app = FastAPI(title="Haystack REST API", debug=container.config.debug(), version=haystack_version, root_path="/")
    
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
    )
    
    app.add_exception_handler(HTTPException, http_error_handler)
       
    router = APIRouter()
    router.include_router(query_router.router)
    router.include_router(chat_router.router)
    router.include_router(health_router.router)
    router.include_router(pipeline_router.router)
    router.include_router(document_router.router)
    app.include_router(router)
    return app
        
    
if __name__ == "__main__":
    try:
        container = Container()
        container.config.hf_token.from_env("HUGGINGFACE_TOKEN")
        container.config.elasticsearch_host.from_env("ELASTICSEARCH_HOST",default="localhost")
        container.config.elasticsearch_port.from_env("ELASTICSEARCH_PORT",as_=int,default=9200)
        container.config.elasticsearch_user.from_env("ELASTICSEARCH_USER",default="")
        container.config.elasticsearch_password.from_env("ELASTICSEARCH_PASSWORD",default="")
        container.config.embedding_dim.from_env("EMBEDDING_DIM",as_=int,default=384)
        container.config.similarity.from_env("SIMILARITY",default="cosine")
        container.config.embedding_model.from_env("EMBEDDING_MODEL",default="LLukas22/all-MiniLM-L12-v2-embedding-all")
        container.config.extractive_qa_model.from_env("EXTRACTIVE_QA_MODEL",default="LLukas22/all-MiniLM-L12-v2-qa-en")
        container.config.use_gpu.from_env("USE_GPU",as_=parse_bool,default=False)
        container.config.use_8bit.from_env("USE_8BIT",as_=parse_bool,default=False)
        container.config.concurency_limit.from_env("CONCURENCY_LIMIT",as_=int,default=5)
        container.config.debug.from_env("DEBUG",as_=parse_bool,default=True)
        
        container.config.chatmodel.from_env("CHATMODEL",as_=parse_chatmodel,default="CPU")
        container.config.chat_max_length.from_env("CHAT_MAX_INPUT_LENGTH",as_=int,default=2000)
        
        #OpenAI Vars
        container.config.open_ai_token.from_env("OPENAI_TOKEN",default=None)
        
        #GPU Vars
        container.config.base_chat_model.from_env("BASE_CHAT_MODEL",default="decapoda-research/llama-7b-hf")
        container.config.use_peft.from_env("USE_PEFT",as_=parse_bool,default=True)
        container.config.adapter_chat_model.from_env("ADAPTER_CHAT_MODEL",default="tloen/alpaca-lora-7b")
        
        container.config.chat_apply_optimizations.from_env("ADAPTER_APPLY_OPTIMIZATIONS",as_=parse_bool,default=True)
        
        #CPU Vars
        container.config.cpu_model_repo.from_env("CPU_MODEL_REPO",default="LLukas22/alpaca-native-7B-4bit-ggjt")
        container.config.cpu_model_filename.from_env("CPU_MODEL_FILENAME",default="ggjt-model.bin")
        container.config.cpu_model_threads.from_env("CPU_MODEL_THREADS",as_=int,default=8)
        container.config.cpu_model_kv_16.from_env("CPU_MODEL_KV_16",as_=parse_bool,default=True)
        container.config.cpu_model_mmap.from_env("CPU_MODEL_MMAP",as_=parse_bool,default=True)
        container.wire(modules=[__name__])
        
    
        app = App(container)    
        logging.info("Open http://127.0.0.1:8001/docs to see Swagger API Documentation.")
        uvicorn.run(app, host="0.0.0.0", port=8001)
        
    except Exception as e:
        print("Something seams to be broken. Please check the logs.")
        print(e)
        logger.exception(e)
        