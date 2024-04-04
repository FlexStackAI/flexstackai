import requests
from typing import Dict, Any

class FlexstackInfo: 
    def __init__(self, base_url: str):
        self.base_url = base_url

    def get_all_models(self, headers: dict = {}):
        """Get all models"""
        url = f"{self.base_url}/ai/models"
        response = requests.get(url, headers=headers)
        return response.json()
    
    def get_models(self, task: str, headers: dict = {}):
        """Get models of a task
        
        Args:
            task (str): Task name
        """

        if task not in [
            'image_generation', 'video_generation', 'text_completion', 'audio_generation', 'text_embedding'
        ]:
            raise ValueError('task must be one of ["image_generation", "video_generation", "text_completion", "audio_generation", "text_embedding"]')
        
        url = f"{self.base_url}/models/{task}"
        response = requests.get(url, headers=headers)
        return response.json()