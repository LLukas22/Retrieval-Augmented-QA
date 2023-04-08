from typing import List, Optional, Literal,Dict,Any
from pydantic import BaseModel, Field, validator


class ChatMessage(BaseModel):
    role: Literal["system","user","assistant"] = Field(..., description="The sender of the message")
    content: str = Field(..., description="The content of the message")
    
class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="The messages to be used for the chat")
    config:Optional[Dict[str,Any]] = Field(None,description="The generation config to use for the chat. Dictionary of a transfomers GenerationConfig")
    stop_words:Optional[List[str]] = Field(None,description="The stop words to use for the chat")
    
class ChatResponse(BaseModel):
    content: str = Field(..., description="The generated response")
    
    
class DefaultConfigResponse(BaseModel):
    config:Dict[str,Any] = Field(...,description="The default generation config of the chat model")
    
class ModelInfo(BaseModel):
    name: str = Field(..., description="The name of the adapter")
    model: str = Field(..., description="The description of the model")
    accelerator: str = Field(..., description="The accelerator used for the model")