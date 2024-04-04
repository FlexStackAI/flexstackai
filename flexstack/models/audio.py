import requests
from typing import Any, Dict


class AudioGeneration: 
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def create_txt2audio(
        self, 
        prompt: str,
        model: str = "musicgen",
        headers: dict = {},
        **kwargs
    ) -> Dict[str, Any]:
        """Generate audio with prompt.

        Args:
            prompt (str): The prompt for generation.
            model (str): The type of model to be used. Default is "musicgen".
            **kwargs: Additional keyword arguments for the model.
        
        Returns:
            A dictionary with the generated audio.
        """

        _kwargs_valids = {
            "audiogen": ["duration", "top_k", "top_p"],
            "musicgen": ["duration", "top_k", "top_p"],
            "bark": []
        }

        # Valid kwargs
        for k, v in kwargs.items():
            if k not in _kwargs_valids[model]:
                raise ValueError(f"Model `{model}` does not support keyword argument: `{k}`. Only suport in [{', '.join(_kwargs_valids[model] + ['prompt', 'model'])}]")
        
        # Defaut kwargs
        if model == "audiogen" or model == "musicgen":
            kwargs["duration"] = 5 if "duration" not in kwargs else kwargs["duration"]
            kwargs["top_k"] = 15 if "top_k" not in kwargs else kwargs["top_k"]
            kwargs["top_p"] = 0.9 if "top_p" not in kwargs else kwargs["top_p"]

        url = f"{self.base_url}/ai/audio_generation"
        configs = dict(
            model=model
        )
        configs.update(kwargs)
        payload = {"prompt": prompt, "configs": configs}
        response = requests.post(url, json=payload, headers=headers)
        
        return response.json()
    
    def result_txt2audio(self, task_id: str, headers: dict = {}):
        """Get result of text to audio generation with task_id

        Args:
            task_id (str): Unique task ID of the text-to-audio generation task.

        Returns:
            A dictionary with detail response.
        """

        url = f"{self.base_url}/ai/audio_generation/{task_id}"
        response = requests.get(url, headers=headers)
        return response.json()
    