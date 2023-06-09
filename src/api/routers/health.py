from typing import List, Optional,Dict

import os
import pynvml
import psutil

from schemas.health import CPUUsage,MemoryUsage,GPUUsage,GPUInfo,HealthResponse


from ._router import BaseRouter

class HealthRouter(BaseRouter):
    def __init__(self):
        super().__init__(prefix="/health")
        self.router.add_api_route("/check", self.check, methods=["GET"])
        self.router.add_api_route("/version", self.versions, methods=["GET"])
        self.router.add_api_route("/usage", self.usage, methods=["GET"],response_model=HealthResponse, status_code=200)
    
    def check(self)-> bool:
        """
        This endpoint can be used during startup to understand if the
        server is ready to take any requests, or is still loading.

        The recommended approach is to call this endpoint with a short timeout,
        like 500ms, and in case of no reply, consider the server busy.
        """
        return True

    def versions(self)->Dict[str,str]:
        """
        Get the versions of the installed packages.
        """
        from haystack import __version__ as hs_version
        from transformers import __version__ as transformers_version
        from torch import __version__ as torch_version
        accelerate_version="Not installed"
        try:
            from accelerate import __version__ as accelerate_version
        except:
            pass
            
        from fastapi import __version__ as fastapi_version
        from starlette import __version__ as starlette_version
        from uvicorn import __version__ as uvicorn_version
        
        return {
            "haystack": hs_version,
            "transformers": transformers_version,
            "torch": torch_version,
            "accelerate": accelerate_version,
            "fastapi":fastapi_version,
            "starlette":starlette_version,
            "uvicorn":uvicorn_version 
            }

    def usage(self):
        """
        This endpoint allows external systems to monitor the uasge of the node running the api.
        """

        gpus: List[GPUInfo] = []

        try:
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem_total = float(info.total) / 1024 / 1024
                gpu_mem_used = float(info.used) / 1024 / 1024
                
                gpu_info = GPUInfo(
                    index=i,
                    name=pynvml.nvmlDeviceGetName(handle),
                    usage=GPUUsage(
                        memory_total=round(gpu_mem_total),
                        kernel_usage=pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                        memory_used=round(gpu_mem_used) if gpu_mem_used is not None else None,
                    ),
                )

                gpus.append(gpu_info)
        except pynvml.NVMLError:
            self.logger.warning("No NVIDIA GPU found.")

        p_cpu_usage = 0
        p_memory_usage = 0
        cpu_count = os.cpu_count() or 1
        p = psutil.Process()
        p_cpu_usage = p.cpu_percent() / cpu_count
        p_memory_usage = p.memory_percent()

        cpu_usage = CPUUsage(used=p_cpu_usage)
        memory_usage = MemoryUsage(used=p_memory_usage)

        from haystack import __version__
        return HealthResponse(version=__version__, cpu=cpu_usage, memory=memory_usage, gpus=gpus)
