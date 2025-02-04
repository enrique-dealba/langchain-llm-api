# Langchain LLM API

A Langchain compatible implementation which enables the integration with [LLM-API](https://github.com/1b5d/llm-api) 

The main reason for implementing this package is to be able to use Langchain with any model run locally.

# Usage

You can install this as a python library using the command (until it's integrated with langchain itself)

```
pip install langchain-llm-api
```

To use this implementation with Banana (app.banana.dev) on deployed models:
```
from langchain_llm_api.banana_llm import BananaLLM

api_key = "" # add api_key here
model_key = "" # add model_key here
url = "https://demo-codellama-13b-instruct-gptq-llziq7mdd2.run.banana.dev" # add your banana url here

llm = BananaLLM(api_key, model_key, url)

llm("Write a python function that adds two numbers together.")
```

To use this langchain implementation with the LLM-API:

```
from langchain_llm_api import LLMAPI, APIEmbeddings

llm = LLMAPI(
    host_name="http://localhost:8000",
    params={"temp": 0.2},
    verbose=True
)

llm("What is the capital of France?")

```

Or with streaming:

```
from langchain_llm_api import LLMAPI, APIEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = LLMAPI(
    host_name="http://localhost:8000",
    params={"temp": 0.2},
    verbose=True,
    streaming=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

llm("What is the capital of France?")

```

Check [LLM-API](https://github.com/1b5d/llm-api) for the possible models and thier params

## Embeddings

to use the embeddings endpoint:

```
emb = APIEmbeddings(
    host_name="your api host name e.g. http://localhost:8000 or wherever your LLM API is hosted",
    params = {"n_predict": 300, "temp": 0.2, ...}
)
```
