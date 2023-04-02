from __future__ import annotations

from typing import Dict, List, Optional, Union,Any

from pydantic import BaseModel, Field, Extra

PrimitiveType = Union[str, int, float, bool]

class ComponentDescription(BaseModel):
    name:str = Field(..., description="Name of the component")
    params:Dict[str,PrimitiveType] = Field(..., description="Parameters of the component")

class PipelineDescription(BaseModel):
    name:str = Field(..., description="Name of the pipeline")
    components:List[ComponentDescription] = Field(..., description="List of components in the pipeline")
    graph:str = Field(..., description="DOT-Graph of the pipeline")
    
class PipelinesResponse(BaseModel):
    pipelines:Optional[List[PipelineDescription]] = Field(..., description="List of pipelines")
    