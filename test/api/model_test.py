import os
import sys
from pathlib import Path

root = Path(__file__).parent.parent.parent
src=str(root/"src")
if src not in sys.path:
    sys.path.insert(0, src)
    
import pytest
from typing import List, Dict, Any, Tuple, Optional, Generator
import api
from api.chat_models import HF_Gpu_Adapter,Cpu_Adapter
from schemas.chat import ChatMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

def test_gpu_model_can_stream():
    gpu_adapter = HF_Gpu_Adapter("bigscience/bloomz-560m",
                                 False,
                                 adapter_model=None,
                                 use_8bit=False,
                                 max_length=512,
                                 apply_optimications=False,
                                 model_prototype=AutoModelForCausalLM,
                                 tokenizer_prototype=AutoTokenizer)
    gpu_adapter.load()
    
    prompt = "The meaning of life is"
    messages=[ChatMessage(role="system",content=prompt)]
    iterations = 0
    config = GenerationConfig(max_new_tokens=50)
    generated_message=""
    for i,word in enumerate(gpu_adapter.generate_streaming(messages,config)):
        iterations+=1
        generated_message+=word
        
    assert iterations>1
    assert len(generated_message)>0
    assert not generated_message.startswith(prompt)
 
def test_gpu_model_does_not_act_as_user():
    #The model should stop generating if it starts to generate a "<user>:" tag
    
    gpu_adapter = HF_Gpu_Adapter("decapoda-research/llama-7b-hf",
                                 True,
                                 adapter_model="tloen/alpaca-lora-7b",
                                 use_8bit=False,
                                 max_length=1024,
                                 apply_optimications=False)
    gpu_adapter.load()
    
    messages=[
        ChatMessage(role="system",content="The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. \n\n Current Conversation:"),
        ChatMessage(role="user",content="What is your task?"),
              ]
    
    iterations = 0
    config = gpu_adapter.default_config()
    config.max_new_tokens=200
    
    generated_message=""
    for i,word in enumerate(gpu_adapter.generate_streaming(messages,config)):
        iterations+=1
        generated_message+=word
        
    assert iterations>1
    assert len(generated_message)>0
    assert "Human:" not in generated_message
    assert gpu_adapter.stop_reason=="Stopword detected!"
    
    
def test_cpu_model_can_stream():
    cpu_adapter = Cpu_Adapter()
    cpu_adapter.load()
    
    prompt = "User: How are you doing?\nBot:"
    messages=[ChatMessage(role="system",content=prompt)]
    config = cpu_adapter.default_config()
    config.max_new_tokens=200
    stream=cpu_adapter.generate_streaming(messages,config) 
    
    iterations = 0
    generated_message=""
    for i,word in enumerate(stream):
        generated_message+=word
        iterations+=1
       
    assert iterations>1 
    assert len(generated_message)>0
    assert not generated_message.startswith(prompt)