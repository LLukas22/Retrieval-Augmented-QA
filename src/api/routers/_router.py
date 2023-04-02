from abc import ABC
from fastapi import FastAPI, APIRouter
import logging  

class BaseRouter(ABC):
    _router:APIRouter
    def __init__(self, prefix:str=""):
        self._router = APIRouter(prefix=prefix)
        self.logger = logging.getLogger(__name__)
    @property   
    def router(self)->APIRouter:
        return self._router