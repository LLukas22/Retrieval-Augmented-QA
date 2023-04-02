import os
import sys
from pathlib import Path

root = str(Path(__file__).parent.parent)

if root not in sys.path:
    sys.path.insert(0, root)
    
import logging
import uvicorn

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger(__name__)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from dependency_injector.wiring import Provide, inject

from api.composition import Container
from api.routers import HealthRouter,PipelineRouter
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

@inject
def App(
    health_router:HealthRouter=Provide[Container.health_router],
    pipeline_router:PipelineRouter=Provide[Container.pipeline_router]
    ):
    
    from haystack import __version__ as haystack_version
    app = FastAPI(title="Haystack REST API", debug=True, version=haystack_version, root_path="/")
    
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
    )
    
    app.add_exception_handler(HTTPException, http_error_handler)
       
    router = APIRouter()
    router.include_router(health_router.router)
    router.include_router(pipeline_router.router)

    app.include_router(router)
    return app
        
    
if __name__ == "__main__":
    container = Container()
    container.config.hf_token.from_env("HUGGINGFACE_TOKEN")
    container.config.open_ai_token.from_env("OPENAI_TOKEN")
    container.config.elasticsearch_host.from_env("ELASTICSEARCH_HOST",default="localhost")
    container.config.elasticsearch_port.from_env("ELASTICSEARCH_PORT",as_=int,default=9200)
    container.config.elasticsearch_user.from_env("ELASTICSEARCH_USER",default="")
    container.config.elasticsearch_password.from_env("ELASTICSEARCH_PASSWORD",default="")
    container.config.embedding_dim.from_env("EMBEDDING_DIM",as_=int,default=384)
    container.config.similarity.from_env("SIMILARITY",default="cosine")
    container.config.embedding_model.from_env("EMBEDDING_MODEL",default="sentence-transformers/all-MiniLM-L12-v2")
    container.config.extractive_qa_model.from_env("EXTRACTIVE_QA_MODEL",default="deepset/minilm-uncased-squad2")
    container.config.use_gpu.from_env("USE_GPU",as_=parse_bool,default=False)
    container.config.use_8bit.from_env("USE_8BIT",as_=parse_bool,default=False)
    container.config.concurency_limit.from_env("CONCURENCY_LIMIT",as_=int,default=5)
    container.wire(modules=[__name__])
    
   
    app = App()    
    logging.info("Open http://127.0.0.1:8001/docs to see Swagger API Documentation.")
    uvicorn.run(app, host="0.0.0.0", port=8001)