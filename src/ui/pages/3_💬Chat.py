import os
import sys
from pathlib import Path
#Ensure the  modules are in the path
root = str(Path(__file__).parent.parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)
   
from typing import List
import streamlit as st 
from streamlit_chat import message as show_message
from ui.utils import show_answer,sidebar_footer,get_api_connector,set_state_if_absent
import time
from schemas.chat import ChatMessage
from haystack.schema import Answer,Document

SYSTEM_PROMPT = "You are an helpfull assistant. Answer concise and try to answer the questions of the user. Explain your answers to the user. After answering stop the conversation."
  
def build_augmented_prompt(prompt:str,possible_answers:List[Answer],documents:List[Document]):
    
    filtered_contexts = []
    for answer in possible_answers:
        filtered_contexts.append(next(doc for doc in documents if doc.id == answer.document_ids[0]).content)
    
    augmented_prompt = f"Try to answer the following question in a few sentenes with the given contexts and answer hints. The answer probably is in the context documents but if it isn't state that you cant answer the question and tell the user he should look at the context documents.\nQuestion:\"{prompt}\"\n"   
    for i,context in enumerate(filtered_contexts):
        augmented_prompt += f"Context {i+1}:\"{context}\" Answer hint {i+1}:\"{possible_answers[i].answer}\"\n\n"
     
    return augmented_prompt
    
def render():
    set_state_if_absent("chat_prompt","")
    set_state_if_absent("chat_qa_results",None)
    set_state_if_absent("chat_messages",[ChatMessage(content=SYSTEM_PROMPT,role="system")])
    
    def reset_results(*args):
        pass
    
    st.set_page_config(page_title="💬Chat")
   
    connector = get_api_connector()
    st.title("💬Chat")
    st.sidebar.header("💬Chat")
    sidebar_footer()
    st.write("This demo allows you to chat with a Large-Language-Model (e.g. ChatGPT) that can search the document index and answer your questions.")
    st.markdown("----")
      
    placeholder=st.empty().container()
    
    def render_chat_history():
        with placeholder:
            for i,message in enumerate(st.session_state.chat_messages):
                if message.role == "assistant":
                    show_message(message.content,is_user=False,key=f"assistant_{i}")
                elif message.role == "user":
                    show_message(message.content,is_user=True,key=f"user_{i}")
                
    placeholder_info_massage=st.empty()
    placeholder_generated_massage=st.empty()     
       
    prompt = st.text_input(
        value=st.session_state["chat_prompt"],
        max_chars=500,
        on_change=reset_results,
        label="chat_prompt",
        label_visibility="hidden",
    )
    
    query_documents=st.checkbox("Query documents",value=False)
    
    if prompt and len(prompt) > 0:
        if query_documents:
            st.session_state.chat_qa_results = connector.qa(prompt)
            if st.session_state.chat_qa_results:
                possible_answers =  st.session_state.chat_qa_results.answers[:3]
                documents =  st.session_state.chat_qa_results.documents
                with placeholder_info_massage:
                    show_message(f"I found {len(documents)} documents that might answer you question. I will now try to formulate an answer based on the top 3 documents.")
                prompt=build_augmented_prompt(prompt,possible_answers,documents)
                
                
        st.session_state["chat_prompt"] = ""
        message=ChatMessage(content=prompt,role="user")
        st.session_state.chat_messages.append(message)
        render_chat_history()
        answer = ""
        for piece in connector.chat_streaming(st.session_state.chat_messages):
            answer+=piece
            with placeholder_generated_massage:
                show_message(answer)
        st.session_state.chat_messages.append(ChatMessage(content=answer,role="assistant"))
        prompt=None
    else:
        render_chat_history()
    
    if st.session_state.chat_qa_results is not None:
        st.write("## Used Context Documents:")
        possible_answers =  st.session_state.chat_qa_results.answers[:3]
        documents =  st.session_state.chat_qa_results.documents 
        for possible_answer in possible_answers:
            show_answer(possible_answer,documents)   
   

render()    
    