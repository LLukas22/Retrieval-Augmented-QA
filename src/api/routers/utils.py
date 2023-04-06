from contextlib import contextmanager
from threading import Semaphore
from fastapi import HTTPException
import queue
import threading
import collections.abc
from transformers.generation.streamers import BaseStreamer
from transformers import AutoTokenizer
import time 


class RequestLimiter:
    def __init__(self, limit):
        self.semaphore = Semaphore(limit - 1)

    @contextmanager
    def run(self):
        acquired = self.semaphore.acquire(blocking=False)
        if not acquired:
            raise HTTPException(status_code=503, detail="The server is busy processing requests.")
        try:
            yield acquired
        finally:
            self.semaphore.release()
 
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
    def __init__(self, tokenizer: "AutoTokenizer",skip_prompt:bool=True) -> None:
        super().__init__(tokenizer,skip_prompt)
        self.finished=False 
        self.generated_text = queue.Queue()
              
    def on_finalized_text(self, token: str, stream_end: bool = False):
        self.generated_text.put(token)
        if stream_end:
            self.finished=True
            
    def __iter__(self)->str:
        while not self.finished or not self.generated_text.empty():
            if self.generated_text.empty():
                time.sleep(0.1)
            else:
                yield self.generated_text.get()
        

        
    
        
  
#see https://github.com/eric-wieser/generatorify/blob/master/generatorify.py          
class generator_from_callback(collections.abc.Generator):
    """
    A generator wrapper for a function that invokes a callback multiple times.
    Calling `send` on the generator emits a value from one callback, and returns
    the next.
    Note this starts a background thread
    """
    def __init__(self, func):
        self._ready_queue = queue.Queue(1)
        self._done_queue = queue.Queue(1)
        self._done_holder = [False]

        # local to avoid reference cycles
        ready_queue = self._ready_queue
        done_queue = self._done_queue
        done_holder = self._done_holder

        def callback(value):
            done_queue.put((False, value))
            cmd, val = ready_queue.get()
            if cmd == 'send':
                return val
            elif cmd == 'throw':
                raise val
            else:
                assert False  # pragma: no cover

        def thread_func():
            while True:
                cmd, val = ready_queue.get()
                if cmd == 'send' and val is not None:
                    done_queue.put((True, TypeError("can't send non-None value to a just-started generator")))
                    continue
                break
            try:
                if cmd == 'throw':
                    raise val
                ret = func(callback)
                raise StopIteration(ret) if ret is not None else StopIteration
            except BaseException as e:
                done_holder[0] = True
                done_queue.put((True, e))
        self._thread = threading.Thread(target=thread_func)
        self._thread.start()

    def _put(self, *args):
        if self._done_holder[0]:
            raise StopIteration
        self._ready_queue.put(args)
        is_exception, val = self._done_queue.get()
        if is_exception:
            try:
                raise val
            finally:
                # prevent val's traceback containing a reference cycle
                del val
        else:
            return val

    def send(self, value):
        return self._put('send', value)

    def throw(self, exc):
        return self._put('throw', exc)

    def __next__(self):
        return self.send(None)

    def close(self):
        try:
            self.throw(GeneratorExit)
        except StopIteration:
            self._thread.join()
        except GeneratorExit:
            self._thread.join()
        except BaseException:
            self._thread.join()
            raise
        else:
            # yielded again, can't clean up the thread
            raise RuntimeError('Task with callback ignored GeneratorExit')

    def __del__(self):
        self.close()