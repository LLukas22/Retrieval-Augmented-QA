from typing import Dict, Any
from .utils import RequestLimiter
from ._router import BaseRouter
from ..chat_models import ModelAdapter
from schemas.chat import ChatRequest, ChatResponse, DefaultConfigResponse,ModelInfo
from transformers import GenerationConfig
from fastapi.responses import StreamingResponse

class ChatRouter(BaseRouter):
    def __init__(self,chat_model:ModelAdapter,limiter:RequestLimiter):
        super().__init__("/chat")
        self.chat_model = chat_model
        self.limiter = limiter
        self.chat_model.load()
        self.router.add_api_route("/info", self.info, methods=["GET"],response_model=ModelInfo)
        self.router.add_api_route("/default_config", self.default_config, methods=["GET"],response_model= DefaultConfigResponse)
        self.router.add_api_route("/availability", self.check_availability, methods=["GET"])
        self.router.add_api_route("/prompt", self.prompt, methods=["POST"], response_model=ChatResponse)
        self.router.add_api_route("/prompt_streaming", self.prompt_streaming, methods=["POST"], response_model=StreamingResponse)
        
    def _get_config(self,config:dict)->GenerationConfig:
        generation_config=None
        if config is None or len(config)==0:
            generation_config=self.chat_model.default_config()
        else:
           generation_config=GenerationConfig.from_dict(config)
           
        return generation_config
    
    async def prompt(self,request: ChatRequest)->ChatResponse:
        """
        Prompts the chat model with the given messages and returns the response
        """
        config=self._get_config(request.config)
        stop_words = request.stop_words if request.stop_words else []
        with self.limiter.run():
            message = self.chat_model.generate(request.messages,config,stop_words)
            
            return {"content":message}
     
        
    async def prompt_streaming(self,request: ChatRequest)->StreamingResponse:
        """
        Streaming version of the prompt endpoint
        """
        config=self._get_config(request.config)
        stop_words = request.stop_words if request.stop_words else []
        with self.limiter.run():
            generator = self.chat_model.generate_streaming(request.messages,config,stop_words)
            return StreamingResponse(content=generator,media_type="text")
        
    
    async def check_availability(self)->bool:
        """
        Check if the chat model is available
        """
        with self.limiter.run():
            return True   
        
    async def info(self)->str:
        """
        Get infos about the chat model used
        """
        return self.chat_model.info()
    
    async def default_config(self)->DefaultConfigResponse:
        """
        Get the default config used by the chat model
        """
        config = self.chat_model.default_config().to_diff_dict()
        return {"config":config}
        
        