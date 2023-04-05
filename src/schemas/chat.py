from typing import List, Optional, Literal,Dict,Any
from pydantic import BaseModel, Field, validator


class ChatMessage(BaseModel):
    role: Literal["system","user","assistant"] = Field(..., description="The sender of the message")
    content: str = Field(..., description="The content of the message")
    
class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="The messages to be used for the chat")
    config:Optional[Dict[str,Any]] = Field(None,description="The generation config to use for the chat. Dictionary of a transfomers GenerationConfig")
    
class ChatResponse(BaseModel):
    content: str = Field(..., description="The generated response")