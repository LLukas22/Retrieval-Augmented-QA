import os
import sys
from pathlib import Path
#Ensure the  modules are in the path
root = str(Path(__file__).parent.parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)
   
from typing import List,Optional,Dict,Any,Iterator,Generator
import streamlit as st 
from streamlit_chat import message as show_message
from ui.utils import show_answer,sidebar_footer,get_api_connector,set_state_if_absent,set_default_generation_config,get_generation_config
import time
from schemas.chat import ChatMessage,DefaultConfigResponse
from haystack.schema import Answer,Document


SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT","The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. \n\n Current Conversation:")
WELCOME_MESSAGE = os.getenv("CHAT_WELCOME_MESSAGE","Hello, i will try to answer any questions you have for me! Untick the checkbox to disable the document search and chat with me normaly.")

def batch_generator(generator:Generator[str,None,None],size:int=7)->Generator[str,None,None]:
    buffer = []
    for entry in generator:
        buffer.append(entry)
        if len(buffer) >= size:
            yield "".join(buffer)
            buffer=[]
    if len(buffer) > 0:
        yield "".join(buffer)
        buffer=[]
                
def build_alpaca_prompt(question:str,contexts:List[str],answers:List[str],include_answer_spans:bool=False):
    instruction = f"Try to answer the following question concise and well formulated in a few sentenes using only the given informations. Explain your answer.\nQuestion:\"{question}\"\n"
    
    if include_answer_spans:
        instruction += f"Here are some hints for the answer:"
        for i,answer in enumerate(answers):
            instruction+= f"Hint for context {i+1}: \"{answer}\""
            
    input = ""
    for i,context in enumerate(contexts):
        input+= f"Context {i+1}: \"{context}\""   
    prompt = f"""### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    return [ChatMessage(role="system",content=prompt)]


def build_chatgpt_prompt(question:str,contexts:List[str],answers:List[str],include_answer_spans:bool=False):
    system_prompt = "You will be given a question and multiple contexts. Try to answer the question concise and well formulated in a few sentences using only the provided information. The contexts most likely contain the answer but its not assured."
    def with_answer_template_chatgpt(question:str,contexts:List[str],answers:List[str])->str:
        joined_contexts = ""
        for i,context in enumerate(contexts):
            joined_contexts += f"Context {i+1}:\"{context}\""
            if include_answer_spans:
                joined_contexts += "\nHint for context {i+1}: \"{answers[i]}\"\n"
        joined_contexts = joined_contexts.strip()
        return f"Question:{question}\nContexts:{joined_contexts}" 
    return [ChatMessage(role="system",content=system_prompt),ChatMessage(role="user",content=with_answer_template_chatgpt(question,contexts,answers))]

class UIChatMessage:
    def __init__(self,chatmessage:ChatMessage,display_message:str=None, should_send:bool=True) -> None:
        self.chatmessage = chatmessage
        self.display_message = display_message
        self.should_send = should_send
        
    @property
    def display_text(self)->str:
        if self.display_message:
            return self.display_message
        return self.chatmessage.content
    
    @property
    def role(self)->str:
        return self.chatmessage.role
    
def render():
    set_state_if_absent("chat_prompt","")
    set_state_if_absent("chat_qa_results",None)
    set_state_if_absent("chat_document_count",3)
    set_state_if_absent("chat_use_document_search",True)
    
    set_state_if_absent("chat_messages",[UIChatMessage(ChatMessage(content=SYSTEM_PROMPT,role="system")),UIChatMessage(ChatMessage(content=WELCOME_MESSAGE,role="assistant"),should_send=False)])
    
    connector = get_api_connector()
    if not connector.healtcheck():
        st.error("The API is not reachable")
        return
    
    chat_info =connector.chat_info()
    default_config = connector.chat_default_config()
    if not default_config or not chat_info:
        st.error("The API does not support chat")
        return
    set_default_generation_config(default_config)
    
    def get_messages_to_send():
        return [uimessage.chatmessage for uimessage in st.session_state.chat_messages if uimessage.should_send]
    
    def reset_results(*args):
        pass
    
    st.set_page_config(page_title="💬Chat")
   
    
    st.title("💬Chat")
    
    # region Sidebar
    st.sidebar.header("💬Chat")
    document_count = st.sidebar.slider(
        "Documents to consider",
        min_value=1,
        max_value=10,
        value=st.session_state["chat_document_count"],
    )
    include_answer_spans = st.sidebar.checkbox(
        "Include answer-spans in context",
        value=False
    )
    st.sidebar.markdown("----")
    st.sidebar.write("## Generation Configs:")
    
    temperature= st.sidebar.slider(
        "temperature",
        min_value=0.1,
        max_value=2.0,
        value=st.session_state["temperature"],
        step=0.2
    )
    
    top_p = st.sidebar.slider(
        "Top P",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state["top_p"],
        step=0.05
    )
    
    max_new_tokens = st.sidebar.slider(
        "Max New Tokens",
        min_value=20,
        max_value=512,
        value=st.session_state["max_new_tokens"],
        step=20
    )
    
    repetition_penalty= st.sidebar.slider(
        "Repetition Penalty",
        min_value=-2.0,
        max_value=2.0,
        value=st.session_state["repetition_penalty"],
        step=0.1
    )
      

    st.sidebar.markdown("----")
    st.sidebar.write("## Adapter Infos:")
    st.sidebar.write(f"Adapter: {chat_info.name}")
    st.sidebar.write(f"Model: {chat_info.model}")
    st.sidebar.write(f"Accelerator: {chat_info.accelerator}")
    sidebar_footer()
    # endregion
    
    st.write("This demo allows you to chat with a Large-Language-Model (e.g. Alpaca or ChatGPT) that can search the document index and answer your questions.")
    if chat_info.name == "LLaMA.cpp":
        st.write("⚠️WARNING: This model is running on a CPU! Expect long response times!⚠️")
    st.markdown("----")
      
    
    
    def build_augmented_prompt(prompt:str,possible_answers:List[Answer],documents:List[Document])->List[ChatMessage]:
    
        filtered_contexts = []
        for answer in possible_answers:
            filtered_contexts.append(next(doc for doc in documents if doc.id == answer.document_ids[0]).content)
            
        if chat_info.name == "OpenAI":
            return build_chatgpt_prompt(prompt,filtered_contexts,[answer.answer for answer in possible_answers],include_answer_spans)
        return build_alpaca_prompt(prompt,filtered_contexts,[answer.answer for answer in possible_answers],include_answer_spans)
    
    placeholder=st.empty().container()
    
    def render_chat_history():
        placeholder.empty()
        with placeholder:
            for i,message in enumerate(st.session_state.chat_messages):
                if message.role == "assistant":
                    show_message(message.display_text,is_user=False,key=f"assistant_{i}",seed="Felix")
                elif message.role == "user":
                    show_message(message.display_text,is_user=True,key=f"user_{i}",seed="Angel",avatar_style="identicon")
                
    placeholder_generated_massage=st.empty()    
    
    
    def prompt_model(chat_messages,stop_words=[])->str:
        render_chat_history()
        answer = ""
        config = get_generation_config(temperature,top_p,max_new_tokens,repetition_penalty)
        for i,piece in enumerate(batch_generator(connector.chat_streaming(chat_messages,config=config,stop_words=stop_words),size=10)):
            answer+=piece
            with placeholder_generated_massage:
                show_message(answer,key=f"generated_answer_{i}",seed="Felix")
        st.session_state.chat_messages.append(UIChatMessage(ChatMessage(content=answer,role="assistant")))
        return answer
            
    
    
       
    form = st.form("ask_form",clear_on_submit=True)
    
    use_document_search=form.checkbox("Query documents",value=True)
    
    prompt = form.text_input(
        value="",
        max_chars=500,
        label="chat_prompt",
        label_visibility="hidden",
    )
    submitted = form.form_submit_button("Submit")
    
    if submitted and prompt and len(prompt) > 0:
        
        if not connector.chat_is_ready():
            st.info("The model is at capacity right now. Please wait a few seconds and try again.")
            render_chat_history()
            return 
        
        if use_document_search:
            st.session_state.chat_messages.append(UIChatMessage(ChatMessage(content=prompt,role="user")))
            st.session_state.chat_qa_results = connector.qa(prompt)
            if st.session_state.chat_qa_results:
                possible_answers =  st.session_state.chat_qa_results.answers[:document_count]
                documents =  st.session_state.chat_qa_results.documents
                augmented_prompts=build_augmented_prompt(prompt,possible_answers,documents)
                
                st.session_state.chat_messages.append(UIChatMessage(ChatMessage(content=f"I found {len(documents)} documents that might answer your question and linked them bellow. I will now try to formulate an answer based on the top {document_count} documents.",role="assistant"),should_send=False))
                if possible_answers[0].score < 0.1:
                    st.session_state.chat_messages.append(UIChatMessage(ChatMessage(content=f"Im very uncertain that the documents contain the answer. My answer will probably be incorrect, please verify it with the context documents bellow.😅",role="assistant"),should_send=False))
                
                prompt_model(augmented_prompts)
            else:
                st.session_state.chat_messages.append(UIChatMessage(ChatMessage(content=f"Sorry i found no documents regarding your question, are you sure the document store is running?",role="assistant"),should_send=False))
                render_chat_history()
        else:      
            st.session_state.chat_messages.append(UIChatMessage(ChatMessage(content=prompt,role="user")))    
            prompt_model(get_messages_to_send(),stop_words=["Human:"])
            
        prompt=None    
    else:
        render_chat_history()
    
    if st.session_state.chat_qa_results is not None:
        st.write("## Used Context Documents:")
        possible_answers =  st.session_state.chat_qa_results.answers[:document_count]
        documents =  st.session_state.chat_qa_results.documents 
        for possible_answer in possible_answers:
            show_answer(possible_answer,documents)   
   

render()    
    