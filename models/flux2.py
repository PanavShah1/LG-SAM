"""FLUX.2-dev image generator using diffusers."""

from typing import List, Optional, Union

import torch
from diffusers import Flux2Pipeline
from PIL import Image


class Flux2Generator:
    """Wrapper for FLUX.2-dev image generation model."""

    def __init__(self, device: Optional[Union[torch.device, str]] = None):
        """
        Initialize FLUX.2-dev generator.

        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._loaded = False

    def load(self):
        """Load the FLUX.2-dev model."""
        if self._loaded:
            return

        print("Loading FLUX.2-dev model...")

        # Initialize pipeline with bfloat16 for H100
        self.pipe = Flux2Pipeline.from_pretrained(
            "black-forest-labs/FLUX.2-dev", torch_dtype=torch.bfloat16
        )

        # Enable CPU offload for memory management (needed even for H100)
        # This ensures text-encoder, transformer, and VAE are offloaded appropriately
        self.pipe.enable_model_cpu_offload()

        self._loaded = True
        print(f"FLUX.2-dev model loaded on {self.device}")

    def generate(
        self,
        image: Union[Image.Image, List[Image.Image]],
        prompt: Union[str, List[str]],
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        generator_seed: Optional[int] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Generate an image with a rectangle drawn around the object described in the prompt.

        Args:
            image: Input PIL Image or list of PIL Images in RGB format
            prompt: Text description or list of descriptions of the object(s)
            num_inference_steps: Number of inference steps (default: 50)
            guidance_scale: Guidance scale for generation (default: 4.0)
            generator_seed: Optional random seed for reproducibility

        Returns:
            Generated PIL Image or list of PIL Images with rectangle drawn around the object
        """
        if not self._loaded:
            self.load()

        # Handle batch input
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        prompts = prompt if isinstance(prompt, list) else [prompt]

        if len(images) != len(prompts) and len(prompts) != 1:
            # If we have multiple images but one prompt, replicate the prompt
            if len(prompts) == 1:
                prompts = prompts * len(images)
            else:
                raise ValueError(
                    "Number of images and prompts must match, or provide a single prompt."
                )
        elif len(images) != len(prompts):
            # This case handles if we have multiple prompts but one image (unlikely but possible)
            if len(images) == 1:
                images = images * len(prompts)
            else:
                raise ValueError("Number of images and prompts must match.")

        # Augment the prompts
        augmented_prompts = [
            f"Draw a bright red rectangle around the object described below: {p}."
            for p in prompts
        ]

        # Create generator with seed if provided
        if generator_seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(generator_seed)
        else:
            generator = torch.Generator(device=self.device)

        # Generate image using image-to-image (inpainting/editing) mode
        # FLUX2Pipeline supports multi-image input
        generated_images = self.pipe(
            prompt=augmented_prompts,
            image=images,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images

        # Return list if batch input, else single image
        if is_batch or (isinstance(prompt, list) and len(prompt) > 1):
            return generated_images
        return generated_images[0]
