import os
import sys
from pathlib import Path
#Ensure the  modules are in the path
root = str(Path(__file__).parent.parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)
   
import streamlit as st 
from streamlit_chat import message
from ui.utils import sidebar_footer,get_api_connector
    
def render():
    st.set_page_config(page_title="💬Chat")
   
    connector = get_api_connector()
    st.title("💬Chat")
    st.sidebar.header("💬Chat")
    sidebar_footer()
    st.write("This demo allows you to chat with a Large-Language-Model (e.g. ChatGPT) that can search the document index and answer your questions.")
    st.markdown("----")
     
    message("Sorry im not implemented yet 🥲") 
    message("Are you sure i thought i implemented you yesterday🤔", is_user=True,seed=40)
    message("👁️ 👄 👁️") 
    
    next_input = st.text_input(
        max_chars=500,
        label="Ask"
    )
    
render()    