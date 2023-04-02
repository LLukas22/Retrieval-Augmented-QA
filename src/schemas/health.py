from typing import List, Optional
from pydantic import BaseModel, Field, validator

class CPUUsage(BaseModel):
    used: float = Field(..., description="REST API average CPU usage in percentage")

    @validator("used")
    @classmethod
    def used_check(cls, v):
        return round(v, 2)


class MemoryUsage(BaseModel):
    used: float = Field(..., description="REST API used memory in percentage")

    @validator("used")
    @classmethod
    def used_check(cls, v):
        return round(v, 2)


class GPUUsage(BaseModel):
    kernel_usage: float = Field(..., description="GPU kernel usage in percentage")
    memory_total: int = Field(..., description="Total GPU memory in megabytes")
    memory_used: Optional[int] = Field(..., description="REST API used GPU memory in megabytes")

    @validator("kernel_usage")
    @classmethod
    def kernel_usage_check(cls, v):
        return round(v, 2)


class GPUInfo(BaseModel):
    index: int = Field(..., description="GPU index")
    name: str = Field(..., description="GPU name")
    usage: GPUUsage = Field(..., description="GPU usage details")


class HealthResponse(BaseModel):
    version: str = Field(..., description="Haystack version")
    cpu: CPUUsage = Field(..., description="CPU usage details")
    memory: MemoryUsage = Field(..., description="Memory usage details")
    gpus: List[GPUInfo] = Field(default_factory=list, description="GPU usage details")
