from typing import List, Optional

import os
import pynvml
import psutil

from schemas.health import CPUUsage,MemoryUsage,GPUUsage,GPUInfo,HealthResponse


from ._router import BaseRouter

class HealthRouter(BaseRouter):
    def __init__(self):
        super().__init__(prefix="/health")
        self.router.add_api_route("/status", self.check_status, methods=["GET"])
        self.router.add_api_route("/hs_version", self.haystack_version, methods=["GET"])
        self.router.add_api_route("/health", self.get_health_status, methods=["GET"],response_model=HealthResponse, status_code=200)
    
    def check_status(self)-> bool:
        """
        This endpoint can be used during startup to understand if the
        server is ready to take any requests, or is still loading.

        The recommended approach is to call this endpoint with a short timeout,
        like 500ms, and in case of no reply, consider the server busy.
        """
        return True

    def haystack_version(self):
        """
        Get the running Haystack version.
        """
        from haystack import __version__
        return {"hs_version": __version__}

    def get_health_status(self):
        """
        This endpoint allows external systems to monitor the health of the Haystack REST API.
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
