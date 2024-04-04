import requests
from typing import Dict, Any


class TextGeneration: 
    def __init__(self, base_url: str):
        self.base_url = base_url


    def text_generation(
        self, 
        messages: list,
        model: str = "gemma-7b",
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        max_tokens: int = 256,
        headers: dict = {}
    ) -> Dict[str, Any]:
        """Generate text with messages input.

        Args:
            messages (list): The list of messages prompt for generation.
            model (str): The type of model to be used. Default is "gemma-7b".
            temperature (float): Temperature for generation. Default is 0.7.
            top_k (int): Top-k for generation. Default is 50.
            top_p (float): Top-p for generation. Default is 0.95.
            max_tokens (int): Max tokens for generation. Default is 256.
        
        Returns:
            A dictionary with the generated text.
        """
        if not isinstance(messages, list):
            raise ValueError("Messages must be a list. Example: [{'role': 'user', 'content': 'Hello'}]")
        for message in messages:
            keys = list(message.keys())
            if set(keys) != set(['role', 'content']):
                raise ValueError("Message must contain 'role' and 'content'. Example: [{'role': 'user', 'content': 'Hello'}]")
            if message['role'] not in ['system', 'user', 'assistant']:
                raise ValueError("Message role must be 'system', 'user' or 'assistant'. Example: [{'role': 'user', 'content': 'Hello'}]")
        
        url = f"{self.base_url}/ai/text_completion"
        configs = dict(
            model=model, temperature=temperature, top_k=top_k, top_p=top_p, max_new_tokens=max_tokens
        )
        payload = {"messages": messages, "configs": configs}
        response = requests.post(url, json=payload, headers=headers)
        
        return response.json()

    def generate_text_stream(
        self, 
        messages: list,
        model: str = "gemma-7b",
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        max_tokens: int = 256, 
        headers: dict = {}
    ):
        
        """Generate text with messages input in streaming.

        Args:
            messages (list): The list of messages prompt for generation.
            model (str): The type of model to be used. Default is "gemma-7b".
            temperature (float): Temperature for generation. Default is 0.7.
            top_k (int): Top-k for generation. Default is 50.
            top_p (float): Top-p for generation. Default is 0.95.
            max_tokens (int): Max tokens for generation. Default is 256.
        
        Returns:
            A dictionary with the generated text.
        """
        if not isinstance(messages, list):
            raise ValueError("Messages must be a list. Example: [{'role': 'user', 'content': 'Hello'}]")
        for message in messages:
            keys = list(message.keys())
            if set(keys) != set(['role', 'content']):
                raise ValueError("Message must contain 'role' and 'content'. Example: [{'role': 'user', 'content': 'Hello'}]")
            if message['role'] not in ['system', 'user', 'assistant']:
                raise ValueError("Message role must be 'system', 'user' or 'assistant'. Example: [{'role': 'user', 'content': 'Hello'}]")
            
        url = f"{self.base_url}/ai/text_completion?stream=true"
        configs = dict(
            model=model, temperature=temperature, top_k=top_k, top_p=top_p, max_new_tokens=max_tokens
        )
        
        payload = {"messages": messages, "configs": configs}
        with requests.post(url, json=payload, stream=True, headers=headers) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    yield line

    def create_text_embedding(self, text: str, model: str="mistral", headers: dict={}):
        """Create text embedding

        Args:
            text (str): The text for embedding.
            model (str): The type of model to be used. Default is "mistral".

        Returns:
            A dictionary with the generated embedding.
        """
        url = f"{self.base_url}/ai/text_embedding"
        configs = dict(
            model=model
        )
        payload = {"text": text, "configs": configs}
        response = requests.post(url, json=payload, headers=headers)
        return response.json()

    def result_text_embedding(self, task_id: str, headers: dict={}):
        """Get result of text embedding with task_id

        Args:
            task_id (str): Unique task ID of the text embedding task.

        Returns:
            A dictionary with detail response.
        """
        url = f"{self.base_url}/ai/text_embedding/{task_id}"
        response = requests.get(url, headers=headers)
        return response.json()