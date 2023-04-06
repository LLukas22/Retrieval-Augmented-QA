from .utils import RequestLimiter
from ._router import BaseRouter
from ..chat_models import ModelAdapter
from schemas.chat import ChatRequest, ChatResponse
from transformers import GenerationConfig
from fastapi.responses import StreamingResponse

class ChatRouter(BaseRouter):
    def __init__(self,chat_model:ModelAdapter,limiter:RequestLimiter):
        super().__init__("/chat")
        self.chat_model = chat_model
        self.limiter = limiter
        self.chat_model.load()
        self.router.add_api_route("/info", self.info, methods=["GET"])
        self.router.add_api_route("/prompt", self.prompt, methods=["POST"], response_model=ChatResponse)
        self.router.add_api_route("/prompt_streaming", self.prompt_streaming, methods=["POST"], response_model=StreamingResponse)
        
    def _get_config(self,config:dict)->GenerationConfig:
        generation_config=None
        if config is None or len(config)==0:
            generation_config=self.chat_model.default_config()
        else:
           generation_config=GenerationConfig.from_dict(config)
           
        return generation_config
    
    def prompt(self,request: ChatRequest)->ChatResponse:
        """
        Prompts the chat model with the given messages and returns the response
        """
        config=self._get_config(request.config)
        
        with self.limiter.run():
            message = self.chat_model.generate(request.messages,config)
            
            return {"content":message}
     
        
    async def prompt_streaming(self,request: ChatRequest)->StreamingResponse:
        """
        Streaming version of the prompt endpoint
        """
        config=self._get_config(request.config)
        
        with self.limiter.run():
            return StreamingResponse(self.chat_model.generate_streaming(request.messages,config),media_type="text")
        
        
        
    def info(self)->str:
        """
        Get infos about the chat model used
        """
        return self.chat_model.info()
        
        