from typing import Dict,List,Generator,Type
import openai
from transformers import GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteriaList

from peft import PeftModel
from pyllamacpp.model import Model
from pathlib import Path
from huggingface_hub import hf_hub_download
import urllib.request
from tqdm import tqdm
import torch
from abc import ABC, abstractmethod
import logging 
from sentencepiece import SentencePieceProcessor
import os
from dependency_injector.providers import Configuration  
from schemas.chat import ChatMessage
from .model_utils import generator_from_callback,GeneratorStreamer,ManualStopCondition
import threading

CAN_RUN_LLAMA = False
try:
    from transformers import AutoModel,AutoTokenizer,AutoModelForCausalLM
    from transformers import LlamaForCausalLM,LlamaTokenizer
    CAN_RUN_LLAMA=True
except:
    pass

class ModelAdapter(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def generate(self,messages:List[ChatMessage],generationConfig:GenerationConfig)->str:
        pass
    
    @abstractmethod
    def generate_streaming(self,messages:List[ChatMessage],generationConfig:GenerationConfig)->Generator[str,None,None]:
        pass
    
    @abstractmethod
    def load(self):
        pass
    
    def info(self)->str:
        return type(self).__name__
    
    def default_config(self)->GenerationConfig:
        return GenerationConfig()


class ChatGPT_Adapter(ModelAdapter):
    def __init__(self,token:str=None) -> None:
        self.token = token
        self.total_tokens = 0
        if token:
            openai.api_key = token
        else:
            raise Exception("No OpenAI Token Provided! Please provide it over the envirnoment variable OPENAI_TOKEN or use the GPU or CPU models!")
    
    def load(self):
        pass
    
    def generate(self,messages:List[ChatMessage],generationConfig:GenerationConfig)->str:
        
        transformed_messages = [m.dict() for m in messages]
        result = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=transformed_messages,
                    temperature=generationConfig.temperature,
                    top_p=generationConfig.top_p,
                    )
        
        used_tokens = result['usage']['total_tokens']
        self.total_tokens += used_tokens
        logging.info(f"OpenAI: Used {used_tokens} Tokens! Accumulated costs: ({(self.total_tokens/1000)*0.002}$)")
        
        return result['choices'][0]['message']['content']
    
    def generate_streaming(self,messages:List[ChatMessage],generationConfig:GenerationConfig)->Generator[str,None,None]:
        transformed_messages = [m.dict() for m in messages]
        tokens_of_request=0
        for result in openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=transformed_messages,
                    temperature=generationConfig.temperature,
                    top_p=generationConfig.top_p,
                    stream=True
                    ):
            
            delta = result['choices'][0]['delta']
            if "role" in delta:
                continue
            elif "content" in delta:
                yield delta['content']
            else:
                break
            
        self.total_tokens += tokens_of_request
        logging.info(f"OpenAI: Used {tokens_of_request} Tokens! Accumulated costs: ({(self.total_tokens/1000)*0.002}$)")
    
def build_llm_prompt(messages:List[ChatMessage])->str:
    prompt=""
    
    for message in messages:
        if message.role == "system":
            prompt += message.content
        else:
            prompt += f"<{message.role}>:\"{message.content}\"\n"
        
    if  messages[-1].role == "user":
        prompt += "<assistant>:"
    return prompt
    
class HF_Gpu_Adapter(ModelAdapter):
    def __init__(self,base_model:str,
                 use_peft:bool,
                 adapter_model:str,
                 use_8bit:bool,
                 apply_optimications:bool,
                 max_length:int=2000,
                 model_prototype:Type[AutoModel]=LlamaForCausalLM,
                 tokenizer_prototype:Type[AutoTokenizer]=LlamaTokenizer,
                 stopwords:List[str]=["<user>","<user>:","user:"]) -> None:
        self.base_model = base_model
        self.use_peft = use_peft
        self.adapter_model = adapter_model
        self.use_8bit = use_8bit
        self.apply_optimications = apply_optimications
        self.max_length = max_length
        self.model_prototype = model_prototype
        self.tokenizer_prototype = tokenizer_prototype
        self.stopwords = stopwords
        self.stop_reason = None
        
        if not torch.cuda.is_available():
            raise Exception("No GPU available! Please use the CPU or OpenAI models!")
        
    def apply_optimizations(self):
        #enable flash attention and tf32 computations
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cudnn.benchmark=True
     
    def default_config(self)->GenerationConfig:
        return GenerationConfig(top_p=0.9,num_beams=1,max_new_tokens=256,use_cache=True)
    def load(self):
        if self.apply_optimications:
            self.apply_optimizations()
            
        self.model = self.model_prototype.from_pretrained(self.base_model,
                                                      torch_dtype=torch.float16,
                                                      device_map="auto",
                                                      load_in_8bit=self.use_8bit)
        
        if self.use_peft:
            self.model = PeftModel.from_pretrained(self.model, self.adapter_model, device_map={'': 0})
            
        self.model = self.model.eval()
        
        if self.apply_optimications:
            self.model = torch.compile(self.model,mode="max-autotune")
            
        self.tokenizer = self.tokenizer_prototype.from_pretrained(self.base_model)
        self.tokenizer.max_length = self.max_length
            
            
    def generate(self,messages:List[Dict[str,str]],generationConfig:GenerationConfig)->str:
        prompt=build_llm_prompt(messages)
        #These are only used here for the stopword detection
        manual_stop = ManualStopCondition()
        streamer = GeneratorStreamer(self.tokenizer,manual_stop,stop_words=self.stopwords)
        
        with torch.no_grad():
            input = self.tokenizer(prompt, return_tensors="pt")
            input_ids = input["input_ids"].to("cuda")
            generation_output = self.model.generate(
                    input_ids=input_ids,
                    generation_config=generationConfig,
                    return_dict_in_generate=True,
                    output_scores=False,
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([manual_stop])    
                )
            generated_tokens = generation_output.sequences[0]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
        if manual_stop.should_stop.is_set():
            self.stop_reason="Stopword detected!"
        else:
            self.stop_reason="Max Tokens!"
            
        return generated_text[len(prompt):]
    
    def generate_streaming(self,messages:List[ChatMessage],generationConfig:GenerationConfig)->Generator[str,None,None]:
        
        prompt=build_llm_prompt(messages)
        manual_stop = ManualStopCondition()
        streamer = GeneratorStreamer(self.tokenizer,manual_stop,stop_words=self.stopwords)
        
        self.model.generate()
        with torch.no_grad():
            input = self.tokenizer(prompt, return_tensors="pt")
            input_ids = input["input_ids"].to("cuda")
            thread = threading.Thread(target=self.model.generate,
                    kwargs={
                        "input_ids":input_ids,
                        "generation_config":generationConfig,
                        "streamer":streamer,
                        "stopping_criteria":StoppingCriteriaList([manual_stop])              
                    },
                    daemon=True)
            thread.start()
        yield from streamer
        
        if manual_stop.should_stop.is_set():
            self.stop_reason="Stopword detected!"
        else:
            self.stop_reason="Max Tokens!"
    
class Cpu_Adapter(ModelAdapter): 
    def __init__(self,model_directory:str,max_length:int=2000,threads:int=8) -> None:
        self.max_length = max_length
        self.threads=threads
        if model_directory == None:
            raise Exception("No model directory provided! Please provide it over the envirnoment variable MODEL_DIR or use the GPU or OpenAI models!")
        
        self.model_directory = Path(model_directory)
        self.raw_dir = self.model_directory/"raw"
        self.converted_dir = self.model_directory/"converted"
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.converted_dir.mkdir(parents=True, exist_ok=True)
        
        self.raw_model = self.raw_dir/"gpt4all-lora-quantized.bin"
        self.raw_tokenizer = self.raw_dir/"tokenizer.model"

        self.converted_model = self.converted_dir/"ggml-gpt4all-model.bin"
        self.ggjt_model = self.converted_dir/"ggjt-gpt4all-model.bin"

    def _download_raw_model(self):
        if not self.raw_tokenizer.exists():
            hf_hub_download(repo_id="decapoda-research/llama-7b-hf", filename="tokenizer.model", local_dir=self.raw_dir)
    
        if not self.raw_model.exists():
            class DownloadProgressBar(tqdm):
                def update_to(self, b=1, bsize=1, tsize=None):
                    if tsize is not None:
                        self.total = tsize
                    self.update(b * bsize - self.n)
            
        def download_url(url, output_path):
            with DownloadProgressBar(unit='B', unit_scale=True,
                                    miniters=1, desc=url.split('/')[-1]) as t:
                urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
                
            download_url("https://the-eye.eu/public/AI/models/nomic-ai/gpt4all/gpt4all-lora-quantized.bin", self.raw_model) 
    
    def default_config(self)->GenerationConfig:
        return GenerationConfig(top_p=0.9,num_beams=1,max_new_tokens=256,use_cache=True)
    
    def load(self):
        if not self.ggjt_model.exists():
            
            if not self.converted_model.exists():
                from api.ggml_scripts.convert_gpt4all_to_ggml import convert_one_file
                
                self._download_raw_model()
                tokenizer = SentencePieceProcessor(str(self.raw_tokenizer))
                convert_one_file(str(self.raw_model), tokenizer)

                os.rename(self.raw_model, self.converted_model)
                os.rename(str(self.raw_model) + ".orig", self.raw_model)
                
            from api.ggml_scripts.migrate_ggml_2023_03_30_pr613 import main
            main(str(self.converted_model),str(self.ggjt_model))
            
        self.model = Model(ggml_model=str(self.ggjt_model), n_ctx=self.max_length)#use_mlock=True)
    
    def generate(self,messages:List[ChatMessage],generationConfig:GenerationConfig)->str:
        prompt=build_llm_prompt(messages)
        generated_text = self.model.generate(prompt,
                                   n_predict=generationConfig.max_new_tokens,
                                   n_threads=self.threads,
                                   top_k=generationConfig.top_k,
                                   top_p=generationConfig.top_p,
                                   temp=generationConfig.temperature,
                                   )
        return generated_text[len(prompt):]
    
    def generate_streaming(self,messages:List[ChatMessage],generationConfig:GenerationConfig)->Generator[str,None,None]:
        prompt=build_llm_prompt(messages)
        
        return generator_from_callback(lambda c: self.model.generate(prompt,
                            new_text_callback=c,
                            n_predict=generationConfig.max_new_tokens,
                            n_threads=self.threads,
                            top_k=generationConfig.top_k,
                            top_p=generationConfig.top_p,
                            temp=generationConfig.temperature,
                            ))

        
def adapter_factory(configuration:Configuration)->ModelAdapter:
    model_to_use = configuration["chatmodel"]
    if model_to_use == "OPENAI":
        return ChatGPT_Adapter(configuration["open_ai_token"])
    elif model_to_use == "GPU":
        if not CAN_RUN_LLAMA:
            raise Exception("Cannot run GPU model because LLAMA models are not supported by your transformers installation, please use CPU or OPENAI!")
        
        return HF_Gpu_Adapter(
            base_model=configuration["base_chat_model"],
            use_peft=configuration["use_peft"],
            adapter_model=configuration["adapter_chat_model"],
            use_8bit=configuration["use_8bit"],
            apply_optimications=configuration["chat_apply_optimizations"]
        )
    elif model_to_use == "CPU":
        return Cpu_Adapter(
            model_directory= configuration["cpu_model_cache_dir"],
            threads=configuration["cpu_model_threads"]
            )
    else:
        raise Exception("Unknown model type: " + model_to_use)
        
            