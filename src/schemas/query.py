from __future__ import annotations

from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

from pydantic import BaseModel, Field, Extra
from pydantic import BaseConfig

from haystack.schema import Document,Answer

BaseConfig.arbitrary_types_allowed = True
BaseConfig.json_encoders = {np.ndarray: lambda x: x.tolist(), pd.DataFrame: lambda x: x.to_dict(orient="records")}

PrimitiveType = Union[str, int, float, bool]

class RequestBaseModel(BaseModel,extra=Extra.forbid,frozen=False):
    pass
              
class SearchResponse(BaseModel):
    query: str
    answers: List[Answer] = []
    documents: List[Document] = []
    debug: Optional[Dict] = Field(None, alias="_debug")

class QueryRequest(RequestBaseModel):
    query: str=Field(None,example="Who is the US president?")
    params: Optional[Dict[str,Any]] = Field(None)
    debug: Optional[bool] = False

class QAResponse(BaseModel):
    query: str
    answers: List[Answer] = []
    documents: List[Document] = []
    debug: Optional[Dict] = Field(None, alias="_debug")
    
class FilterRequest(RequestBaseModel):
    filters: Optional[Dict[str, Union[PrimitiveType, List[PrimitiveType], Dict[str, PrimitiveType]]]] = None

class CreateLabelSerialized(RequestBaseModel):
    id: Optional[str] = None
    query: str
    document: Document
    is_correct_answer: bool
    is_correct_document: bool
    origin: Literal["user-feedback", "gold-label"]
    answer: Optional[Answer] = None
    no_answer: Optional[bool] = None
    pipeline_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    meta: Optional[dict] = None
    filters: Optional[dict] = None
    
    
class ReindexRequest(RequestBaseModel):
    update_existing_embeddings:bool = Field(False, description="If True, existing embeddings will be updated. If False, only unindexed documents will be indexed.")
    batch_size:int = Field(10000, description="Number of documents to index at once.")


