"""Qwen-Image-Edit generator using diffusers."""

import time
from typing import Optional, Union

import torch
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    QwenImageEditPlusPipeline,
)
from PIL import Image

import config


class QwenImageEditGenerator:
    """Wrapper for Qwen-Image-Edit model."""

    def __init__(self, device: Optional[Union[torch.device, str]] = None):
        """
        Initialize Qwen-Image-Edit generator.

        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = (
            device
            if device
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        self._loaded = False

    def load(self):
        """Load the Qwen-Image-Edit model."""
        if self._loaded:
            return

        print("Loading Qwen-Image-Edit model...")

        # Initialize pipeline with bfloat16
        with torch.cuda.device(self.device):
            self.pipeline = QwenImageEditPlusPipeline.from_pretrained(
                config.QWEN_IMAGE_EDIT_CHECKPOINT,
                device_map="cuda",
                torch_dtype=torch.bfloat16,
            )
        self.pipeline.set_progress_bar_config(disable=True)

        self._loaded = True
        print(f"Qwen-Image-Edit model loaded on {self.device}")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int = 40,
        true_cfg_scale: float = 10.0,
        generator_seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate an image with a rectangle drawn around the object described in the prompt.

        Args:
            image: Input PIL Image or list of PIL Images in RGB format
            prompt: Text description or list of descriptions of the object(s)
            num_inference_steps: Number of inference steps (default: 50)
            true_cfg_scale: True CFG scale for generation (default: 10.0)
            generator_seed: Optional random seed for reproducibility

        Returns:
            Generated PIL Image or list of PIL Images with rectangle drawn around the object
        """
        if not self._loaded:
            self.load()

        # Augment the prompts
        # augmented_prompt = f"Draw a single bright red outline rectangle (ie. don't fill it in) around {prompt}. Make sure you draw a the rectangle only around the object described in the prompt. Make sure that you only draw a single rectangle. Do not draw any other rectangles or objects. Do not remove any existing objects from the image. Only add the new rectangle and make no other changes to the image."
        augmented_prompt = f"Draw bright red outline rectangles (ie. don't fill it in) around {prompt}. DO NOT modify the image in any other way. Assume that all the objects specified in the prompt are visible in the image and you are not allowed to add any new objects. You must find them in the image and draw the rectangle around them and not add any new objects."
        # augmented_prompt = f"""
        # # General Instructions
        # - Make sure you draw a the rectangle only around the object described in the prompt
        # - Make sure that you only draw a single rectangle. You will be penalized if you draw more than 1 rectangle.
        # - Make sure you only draw the boundary of the rectangle and don't fill it in.
        # - I will die if you don't follow these instructions.

        # User Input: {prompt}.
        # """

        # Create generator with seed if provided
        if generator_seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(generator_seed)
        else:
            generator = torch.Generator(device=self.device).manual_seed(
                int(time.time())
            )

        inputs = {
            "prompt": augmented_prompt,
            "image": [image],
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": true_cfg_scale,
            # "guidance_scale": guidance_scale,
            "negative_prompt": "Multiple red rectangles annotated on the image. Remove any existing objects from the image.",
            # "height": image.height,
            # "width": image.width,
        }
        with torch.inference_mode():
            output = self.pipeline(**inputs)
            generated_images = output.images  # type: ignore

        return generated_images[0]
