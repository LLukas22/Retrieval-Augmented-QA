from .utils import RequestLimiter
from ._router import BaseRouter
from ..chat_models import ModelAdapter
from schemas.chat import ChatRequest, ChatResponse
from transformers import GenerationConfig

class ChatRouter(BaseRouter):
    def __init__(self,chat_model:ModelAdapter,limiter:RequestLimiter):
        super().__init__("/chat")
        self.chat_model = chat_model
        self.limiter = limiter
        self.chat_model.load()
        self.router.add_api_route("/info", self.info, methods=["GET"])
        self.router.add_api_route("/prompt", self.prompt, methods=["POST"], response_model=ChatResponse)
        
        
    def prompt(self,request: ChatRequest)->ChatResponse:
        """
        Prompts the chat model with the given messages and returns the response
        """
        config=None
        if request.config is None or len(request.config)==0:
            config=self.chat_model.default_config()
        else:
           config=GenerationConfig.from_dict(request.config)
            
        with self.limiter.run():
            message = self.chat_model.generate(request.messages,config)
            
            return {"content":message}
        
        
    def info(self)->str:
        """
        Get infos about the chat model used
        """
        return self.chat_model.info()
        
        