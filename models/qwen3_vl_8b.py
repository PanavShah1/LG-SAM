from typing import List, Optional, Union, overload

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)

import config


class Qwen3VL8B:
    """Generic wrapper for Qwen3-VL-8B model."""

    def __init__(
        self,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Initialize Qwen3-VL-8B.

        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
            model_id: Hugging Face model ID
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
        """Load the Qwen3-VL model."""
        if self._loaded:
            return

        print(f"Loading {config.QWEN3_VL_8B_CHECKPOINT}...")

        # Try to import the specific class, otherwise use AutoModel
        try:
            from transformers import Qwen3VLForConditionalGeneration

            model_class = Qwen3VLForConditionalGeneration
        except ImportError:
            model_class = AutoModelForCausalLM

        with torch.cuda.device(self.device):
            self.model = model_class.from_pretrained(
                config.QWEN3_VL_8B_CHECKPOINT,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

            self.processor = AutoProcessor.from_pretrained(
                config.QWEN3_VL_8B_CHECKPOINT, trust_remote_code=True
            )
            self.processor.tokenizer.padding_side = "left"

        self._loaded = True
        print(f"{config.QWEN3_VL_8B_CHECKPOINT} loaded on {self.device}")

    @overload
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> str: ...
    @overload
    def generate(
        self,
        image: List[Image.Image],
        prompt: List[str],
        max_new_tokens: int = 256,
    ) -> List[str]: ...

    def generate(
        self,
        image: Union[Image.Image, List[Image.Image]],
        prompt: Union[str, List[str]],
        max_new_tokens: int = 256,
    ) -> Union[str, List[str]]:
        """
        Generate text based on image and prompt using Qwen3-VL.

        Args:
            image: Input PIL Image or list of PIL Images
            prompt: Text prompt or list of prompts
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Generated text(s)
        """
        if not self._loaded:
            self.load()

        # Handle batch
        is_batch = isinstance(image, list)
        images = image if is_batch else [image]
        prompts = prompt if isinstance(prompt, list) else [prompt]

        # Broadcast prompts if necessary
        if len(images) != len(prompts) and len(prompts) != 1:
            if len(prompts) == 1:
                prompts = prompts * len(images)
            else:
                raise ValueError("Number of images and prompts must match.")
        elif len(images) != len(prompts) and len(images) == 1:
            images = images * len(prompts)

        # Prepare batch messages
        batch_messages = []
        for img, p in zip(images, prompts):
            batch_messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": p},
                        ],
                    }
                ]
            )

        with torch.cuda.device(self.device):
            # Prepare inputs
            inputs = self.processor.apply_chat_template(
                batch_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            # Generate
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                )

            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        if is_batch or (isinstance(prompt, list) and len(prompt) > 1):
            return output_texts
        return output_texts[0]

    @overload
    def generate_text_only(
        self,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> str: ...
    @overload
    def generate_text_only(
        self,
        prompt: List[str],
        max_new_tokens: int = 256,
    ) -> List[str]: ...

    def generate_text_only(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 256,
    ) -> Union[str, List[str]]:
        """
        Generate text based on prompt using Qwen3-VL.

        Args:
            prompt: Text prompt or list of prompts
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Generated text(s)
        """
        if not self._loaded:
            self.load()

        # Handle batch
        is_batch = isinstance(prompt, list)
        prompts = prompt if isinstance(prompt, list) else [prompt]

        # Prepare batch messages
        batch_messages = []
        for prompt in prompts:
            batch_messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            )

        with torch.cuda.device(self.device):
            # Prepare inputs
            inputs = self.processor.apply_chat_template(
                batch_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            # Generate
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                )

            # Decode
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        if is_batch or (isinstance(prompt, list) and len(prompt) > 1):
            return output_texts
        return output_texts[0]
