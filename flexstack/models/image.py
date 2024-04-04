import os
from typing import Any, Dict, List
import requests


class ImageGeneration: 
    def __init__(self, base_url: str):
        self.base_url = base_url

    @staticmethod
    def create_config(rotation: str = "square", steps: int = 50, negative_prompt: str = "", enhance_prompt: bool = False) -> Dict[str, Any]:
        """Helper function to create a configuration for the create_sdxl_task.
        
        Args:
            rotation: The orientation of the image ('square', 'horizontal', 'vertical').
            steps: The number of steps for the image generation process.
            negative_prompt: Aspects to avoid in the generated image.
        
        Returns:
            A configuration dictionary for the image task.
        """
        return {
            "rotation": rotation,
            "steps": steps,
            "negative_prompt": negative_prompt,
            "lora_weight_url": "",
            "enhance_prompt": enhance_prompt        
        }
    
    # IMAGE ============
    def create_sd_task(self, prompt: str, rotation: str = "square", steps: int = 50, negative_prompt: str = "", enhance_prompt: bool = False, headers: dict = {}) -> Dict[str, Any]:
        """Create an SD1.5 image task with a prompt and configuration.
        
        Args:
            prompt: Description of the image to be generated.
            rotation: Image rotation preference.
            steps: Number of generation steps.
            negative_prompt: Negative aspects to avoid in the image.
        
        Returns:
            A dictionary with the task creation response.
        """
        config = ImageGeneration.create_config(rotation, steps, negative_prompt, enhance_prompt)
        url = f"{self.base_url}/ai/create_sd_task"
        payload = {"prompt": prompt, "config": config}
        response = requests.post(url, json=payload, headers=headers)
        return response.json()
    
    def get_result_sd_task(self, task_id: str, headers: dict = {}) -> Dict[str, Any]:
        """Retrieve the result of a previously submitted SD task.
        
        Args:
            task_id: The unique identifier of the SD task.
        
        Returns:
            A dictionary containing the result of the task.
        """
        url = f"{self.base_url}/ai/get_result_sd_task"
        payload = {"task_id": task_id}
        response = requests.post(url, json=payload, headers=headers)
        return response.json()
    
    def create_sdxl_task(self, prompt: str, rotation: str = "square", steps: int = 50, negative_prompt: str = "", enhance_prompt: bool = False, headers:dict = {}) -> Dict[str, Any]:
        """Create an SDXL-turbo image task with a prompt and configuration.
        
        Args:
            prompt: Description of the image to be generated.
            rotation: Image rotation preference.
            steps: Number of generation steps.
            negative_prompt: Negative aspects to avoid in the image.
        
        Returns:
            A dictionary with the task creation response.
        """
        config = ImageGeneration.create_config(rotation, steps, negative_prompt, enhance_prompt)
        url = f"{self.base_url}/ai/create_sdxl_task"
        payload = {"prompt": prompt, "config": config}
        response = requests.post(url, json=payload, headers=headers)
        return response.json()
    
    def get_result_sdxl_task(self, task_id: str, headers: dict = {}) -> Dict[str, Any]:
        """Retrieve the result of a previously submitted SDXL Turbo task.
        
        Args:
            task_id: The unique identifier of the SDXL Turbo task.
        
        Returns:
            A dictionary containing the result of the task.
        """
        url = f"{self.base_url}/ai/get_result_sdxl_task"
        payload = {"task_id": task_id}
        response = requests.post(url, json=payload, headers=headers)
        return response.json()
    
    def create_txt2img(
        self,
        prompt: str,
        model: str = "sdxl-lightning",
        lora: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 8,
        seed: int = -1,
        negative_prompt: str = "",
        enhance_prompt: bool = False,
        headers: dict = {}
    ) -> Dict[str, Any]:
        """Create an text to image generation task with a prompt and configuration.

        Args:
            prompt (str): Description of the image to be generated.
            model (str): The type of model to be used. Default is "sdxl-lightning".
            lora (str): Optional parameter for LORA model.
            width (int): Width of the image. Default is 1024.
            height (int): Height of the image. Default is 1024.
            steps (int): Number of steps for image generation. Default is 8.
            seed (int): Random seed for image generation
            negative_prompt (str): Prompt for generating negative examples.
            enhance_prompt (bool): Whether to enhance the prompt or not. Default is False.

        Returns:
            A dictionary with the task creation response.
        """

        configs = dict(
            model=model, lora=lora, height=height, ưidth=width, steps=steps, negative_prompt=negative_prompt, seed=seed, enhance_prompt=enhance_prompt
        )
        url = f"{self.base_url}/ai/image_generation"
        payload = {"prompt": prompt, "configs": configs}
        response = requests.post(url, json=payload, headers=headers)

        return response.json()
    
    def result_txt2img(self, task_id: str, headers: dict = {}):
        """Get result of text to image generation with task_id

        Args:
            task_id (str): The unique identifier of the text-to-image generation task.

        Returns:
            A dictionary with detail response.
        """
        url = f"{self.base_url}/ai/image_generation/{task_id}"
        response = requests.get(url, headers=headers)
        return response.json()
    
    # VIDEO ============
    def create_txt2vid(
        self,
        prompt: str,
        model: str = "damo-text-to-video",
        width: int = 256,
        height: int = 256,
        fps: int = 8,
        num_frames: int = 16,
        steps: int = 25,
        seed: int = -1,
        negative_prompt: str = "",
        enhance_prompt: bool = False,
        headers: dict = {}
    ) -> Dict[str, Any]:
        """Create an text to video generation task with a prompt and configuration.

        Args:
            prompt (str): Description of the video to be generated.
            model (str): The type of model to be used. Default is "modelscope-txt2vid"".
            width (int): Width of the image. Default is 512.
            height (int): Height of the image. Default is 512.
            fps (int): Number of frames per second.
            num_frames (int): Number of frames in the video.
            steps (int): Number of steps for video generation. Default is 8.
            seed (int): Random seed for video generation
            negative_prompt (str): Prompt for generating negative examples.
            enhance_prompt (bool): Whether to enhance the prompt or not. Default is False.

        Returns:
            A dictionary with the task creation response.
        """
        configs = dict(
            model=model, width=width, height=height, fps=fps, num_frames=num_frames, steps=steps, negative_prompt=negative_prompt, seed=seed, enhance_prompt=enhance_prompt
        )
        url = f"{self.base_url}/ai/video_generation"
        payload = {"prompt": prompt, "configs": configs}
        response = requests.post(url, json=payload, headers=headers)
        return response.json()

    def result_txt2vid(self, task_id: str, headers: dict = {}):
        """Get result of text to video generation with task_id

        Args:
            task_id (str): Unique task ID of the text-to-video generation task.

        Returns:
            A dictionary with detail response.
        """
        url = f"{self.base_url}/ai/video_generation/{task_id}"
        response = requests.get(url, headers=headers)
        return response.json()
    
    # LORA ============
    def get_lora_types(self, headers: dict = {}):
        """Get LORA types
        
        Returns:
            A dictionary containing the LORA types
        """

        url = f"{self.base_url}/lora/types"
        response = requests.get(url, headers=headers)
        return response.json()
    
    def get_lora_cates(self, headers: dict = {}):
        """Get LORA categories

        Returns:
            A dictionary containing the LORA categories
        """

        url = f"{self.base_url}/lora/cates"
        response = requests.get(url, headers=headers)
        return response.json()     

    def get_lora_models(self, type: str = None, cate: str = None, headers: dict = {}):
        """Get LORA models
        
        Args:
            type (str): Type of LoRA model
            cate (str): Category of LoRA model
        
        Returns:
            A dictionary containing the LORA models
        
        """

        params = {}
        params['type'] = type
        params['cate'] = cate

        url = f"{self.base_url}/lora/"
        response = requests.get(url, headers=headers, params=params)
        return response.json()
    
    def create_lora_trainer_task(self, prompt: str, images: List[str], headers: dict = {}) -> Dict[str, Any]:
        """Create a LORA training task with the given prompt and images.

        Args:
            prompt: The text prompt for fine-tuning.
            images: A list of URLs pointing to images for fine-tuning.

        Returns:
            A dictionary containing the task ID and other response data.
        """
        url = f"{self.base_url}/ai/create_lora_trainner_task"

        files = [('files', image) for image in images]
        data = {'prompt': (None, prompt)}

        response = requests.post(url, files=files + list(data.items()), headers=headers)
        return response.json()
    
    def get_result_lora_trainer_task(self, task_id: str, headers: dict = {}) -> Dict[str, Any]:
        """Retrieve the result of a previously submitted LoRA Trainer task.
        
        Args:
            task_id: The unique identifier of the LoRA task.
        
        Returns:
            A dictionary containing the result of the task.
        """
        url = f"{self.base_url}/ai/get_result_lora_trainner_task"
        payload = {"task_id": task_id}
        response = requests.post(url, json=payload, headers=headers)
        return response.json()

