import os

API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", 8001)

def API_ENDPOINT()->str:
    """
    Return the API endpoint constructed from the environment variables.
    """
    return f"http://{API_HOST}:{API_PORT}"
