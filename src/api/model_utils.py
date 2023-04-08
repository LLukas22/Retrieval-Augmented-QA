import time 
import queue
import threading
import collections.abc
from transformers.generation.streamers import BaseStreamer
from transformers.generation.stopping_criteria import StoppingCriteria
from transformers import AutoTokenizer
from typing import List
import torch

class ManualStopCondition(StoppingCriteria):
    """
    Stop condition that can be manually called from a different thread to stop generation.
    """
    def __init__(self) -> None:
        super().__init__()
        self.should_stop=threading.Event()
        
    def stop(self):
        self.should_stop.set()
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return self.should_stop.is_set()
    
    
class TextStreamer(BaseStreamer):
    def __init__(self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        # variables used in the streaming process
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

    def put(self, value):
        """
        Recives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        # Flush the cache, if it exists
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        self.next_tokens_are_prompt = True
        self.on_finalized_text(printable_text, stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Prints the new text to stdout. If the stream is ending, also prints a newline."""
        print(text, flush=True, end="" if not stream_end else None)
        
#see https://stackoverflow.com/questions/42983569/how-to-write-a-generator-class
class GeneratorStreamer(TextStreamer):
    """
    Base class from which `.generate()` streamers should inherit.
    """
    def __init__(self, tokenizer: "AutoTokenizer",stop_condition:ManualStopCondition,skip_prompt:bool=True,stop_words:List[str]=None) -> None:
        super().__init__(tokenizer,skip_prompt)
        self.finished=False
        self.stop_condition=stop_condition
        self.generated_text = queue.Queue()
        self.total_generated_text=""
        self.stop_words=stop_words if stop_words else []
              
    def on_finalized_text(self, token: str, stream_end: bool = False):
        self.total_generated_text+=token
        for stop_word in self.stop_words:
            if stop_word in self.total_generated_text:
                self.finished=True
                self.stop_condition.stop()
                break
            
        if not self.finished:    
            self.generated_text.put(token)
            if stream_end:
                self.finished=True
            
    def __iter__(self)->str:
        while not self.finished or not self.generated_text.empty():
            if self.generated_text.empty():
                time.sleep(0.1)
            else:
                next_word = self.generated_text.get()
                yield next_word