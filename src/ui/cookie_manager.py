#See https://discuss.streamlit.io/t/cookies-support-in-streamlit/16144/45?page=2
import streamlit as st
import os
from typing import Optional,Any
from streamlit_cookies_manager import CookieManager
from streamlit_cookies_manager.cookie_manager import parse_cookies
import time


class StCookieManager(CookieManager):
    def __init__(self):
        super().__init__()
        self.counter=0
        
    def set_cookie(self,name:str,value:Any)->None:
        self[name]=value

    
    def get_cookie(self,name:str,default:Optional[Any]=None)->Optional[Any]:
        if name in self:
            return self[name]
        else:
            value = os.getenv(name, default)
            if value is not None:
                self.set_cookie(name,value)
                return value
        return None
    
    def _read_cookies(self)->None:
        raw_cookies = self._run_component(save_only=False, key=f"CookieManager.sync_cookies_{self.counter}")
        self.counter+=1
        if raw_cookies:
            self._cookies = parse_cookies(raw_cookies)
            self._clean_queue()
        
    def sync_cookies(self)->bool:
        try:
            self._read_cookies()
            with st.spinner("⌛️ &nbsp;&nbsp; Loading States..."):
                while not self.ready():
                    self._read_cookies()
                    time.sleep(0.1)
            return True
        except Exception as e:
            return False
         
st.cache(allow_output_mutation=True)    
def get_manager()->StCookieManager:
    cookie_manager = StCookieManager()
    return cookie_manager
