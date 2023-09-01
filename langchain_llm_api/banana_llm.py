import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_llm_api.llm import LLMAPI
import banana_dev as client


class BananaLLM(LLMAPI):
    def __init__(self, api_key, model_key, url, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.__dict__['banana_client'] = client.Client(
            api_key=api_key,
            model_key=model_key,
            url=url,
        )

    def remove_instruction_tags(self, text: str):
        return re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
    
    def clean_text(self, text: str):
        return re.sub(r'\s+', ' ', text).strip()
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        start = "[INST] "
        llama_prompt = start + prompt # make sure " [/INST]" is at end of `prompt`
        inputs = {
            "prompt": llama_prompt
            }
        
        # Making the API call
        result, _ = self.banana_client.call("/", inputs)
        
        if isinstance(result, dict):
            if "outputs" in result.keys():
                result = str(result["outputs"])
                result = self.remove_instruction_tags(result)
                result = self.clean_text(result) # removes newlines and whitespace
        
        return result

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        raise NotImplementedError("Async call not currently supported.")
    
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "banana-llm"
