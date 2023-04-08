from typing import Optional,Any,Type,Generator,List,Dict
import os


from httpx import Client, Timeout, Response
from httpx._client import UseClientDefault
import streamlit as st
import logging 
from time import sleep
import pydantic
from schemas.health import HealthResponse
from schemas.pipelines import PipelinesResponse
from schemas.query import QAResponse,SearchResponse,QueryRequest,ReindexRequest
from schemas.chat import ChatResponse,ChatRequest,ChatMessage,ModelInfo,DefaultConfigResponse
    
class ApiConnector():
    def __init__(self) -> None:
        self._host = os.getenv("API_HOST", "localhost")
        self._port = os.getenv("API_PORT", 8001)
        self.timeout = Timeout(5, read=30, write=30)
        self.client = Client(base_url=self.endpoint, timeout=self.timeout)
        
       
    @property
    def endpoint(self)->str:
        return f"http://{self._host}:{self._port}"
    
    
    def __get(self,url:str,timeout:int|UseClientDefault=UseClientDefault())->Optional[Response]:
        try:
            response = self.client.get(url,timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as e:
            logging.exception(e)
        return None
    
    
    def __post(self,url:str,json:Any|None=None,timeout:int|UseClientDefault=UseClientDefault())->Optional[Response]:
        try:
            response = self.client.post(url,json=json,timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as e:
            logging.exception(e)
        return None  
    
    def __parse_response(self,response:Response,type:Type)->Optional[Any]:
        return pydantic.parse_obj_as(type_=type,obj=response.json()) 

    
    
    def healtcheck(self)->bool:
        url = "/health/check"
        try:
            result = self.__get(url)
            return result.json()
        except Exception as e:
            logging.exception(e)
            sleep(0.5)  # To avoid spamming a non-existing endpoint at startup
        return False
    
    st.cache_data
    def versions(self)->dict[str,str]:
        url = "/health/version"
        try:
            result = self.__get(url)
            return result.json()
        except Exception as e:
            logging.exception(e)
        return {}
    
    def system_usage(self)->Optional[HealthResponse]:
        url = "/health/usage"
        try:
            result = self.__get(url)
            return self.__parse_response(result,HealthResponse)
        except Exception as e:
            logging.exception(e)
        return None
    
    st.cache_data    
    def pipelines(self)->Optional[PipelinesResponse]:
        url = "/pipeline/pipelines"
        try:
            result = self.__get(url)
            return self.__parse_response(result,PipelinesResponse)
        except Exception as e:
            logging.exception(e)
        return None
    
    def reindex(self)->bool:
        url = "/query/reindex"
        try:
            request = ReindexRequest(update_existing_embeddings=False)
            result = self.__post(url,json=request.dict(),timeout=None)
            if result:
                return True    
        except Exception as e:
            logging.exception(e)
        return False
        
    def search(self,query:str, top_k:int=5)->Optional[SearchResponse]:
        url = "/query/search"
        request=QueryRequest(query=query,params={
            "Embedding":{ "top_k": top_k},
            "BM25":{ "top_k": top_k},
            })
        
        try:
            result = self.__post(url,json=request.dict())
            return self.__parse_response(result,SearchResponse)
        except Exception as e:
            logging.exception(e)
        return None
            
    def qa(self,query:str, top_k_retrievers:int=5,top_k_reader:int=5)->Optional[QAResponse]:
        url = "/query/qa"
        request=QueryRequest(query=query,params={
            "BM25":{ "top_k": top_k_retrievers},
            "Embedding":{ "top_k": top_k_retrievers},
            "Reader":{ "top_k": top_k_reader}
        })
        try:
            result = self.__post(url,json=request.dict())
            return self.__parse_response(result,QAResponse)
        except Exception as e:
            logging.exception(e)
        return None
        
    def chat_streaming(self,messages:List[ChatMessage],config:Dict[str,Any]=None,stop_words:List[str]=[])->Generator[str,None,None]:
        url="/chat/prompt_streaming"
        request = ChatRequest(messages=messages,config=config,stop_words=stop_words)
        try:
            with self.client.stream("POST",url,json=request.dict(),timeout=None) as response:
                for line in response.iter_text():
                    yield line
                    
        except Exception as e:
            logging.exception(e)
    
    st.cache_data        
    def chat_info(self)->Optional[ModelInfo]:
        url="/chat/info"
        try:
            result = self.__get(url)
            return self.__parse_response(result,ModelInfo)
        except Exception as e:
            logging.exception(e)
        return None
    
    st.cache_data
    def chat_default_config(self)->Optional[DefaultConfigResponse]:
        url="/chat/default_config"
        try:
            result = self.__get(url)
            return self.__parse_response(result,DefaultConfigResponse)
        except Exception as e:
            logging.exception(e)
        return None
    
    def chat_is_ready(self)->bool:
        url="/chat/availability"
        try:
            result = self.__get(url)
            return result.json()
        except Exception as e:
            logging.exception(e)
        return False
            
            

st.cache_resource
def get_api_connector()->ApiConnector:
    return ApiConnector()