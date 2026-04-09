from typing import List, Optional, Union, overload

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    LogitsProcessor,
    LogitsProcessorList,
)

# MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Thinking"
MODEL_ID = "./checkpoints/Qwen3-VL-30B-A3B-Thinking"


class ThinkingBudgetLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, budget: int):
        """
        :param tokenizer: The model's tokenizer
        :param budget: The maximum number of tokens allowed inside the <think> block
        """
        self.budget = budget

        self.start_token_id = tokenizer.convert_tokens_to_ids("<think>")
        self.end_token_id = tokenizer.convert_tokens_to_ids("</think>")

        if self.start_token_id is None or self.end_token_id is None:
            raise ValueError(
                "The tokenizer does not recognize <think> or </think> tokens."
            )

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        for i, seq in enumerate(input_ids):
            # Find the positions of all <think> and </think> tokens
            start_indices = (seq == self.start_token_id).nonzero(as_tuple=True)[0]
            end_indices = (seq == self.end_token_id).nonzero(as_tuple=True)[0]

            # Check if we are currently "thinking"
            # We are thinking if we have a start token, and the number of end tokens
            # is less than the number of start tokens (meaning the last one is open)
            is_thinking = len(start_indices) > 0 and len(end_indices) < len(
                start_indices
            )

            if is_thinking:
                # Get the index of the most recent <think> token
                last_start_index = start_indices[-1].item()

                # Calculate how many tokens have been generated since <think>
                # (current length - position of start token)
                current_thought_length = seq.shape[0] - last_start_index

                # 4. If over budget, Force the End Token
                if current_thought_length >= self.budget:
                    # Set all scores to -infinity
                    scores[i, :] = -float("inf")
                    # Set the end token score to 0 (which becomes probability 1.0 after softmax)
                    scores[i, self.end_token_id] = 0

        return scores


class Qwen3VL:
    """Generic wrapper for Qwen3-VL model."""

    def __init__(
        self,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Initialize Qwen3-VL.

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

        print(f"Loading {MODEL_ID}...")

        # Try to import the specific class, otherwise use AutoModel
        try:
            from transformers import Qwen3VLMoeForConditionalGeneration

            model_class = Qwen3VLMoeForConditionalGeneration
        except ImportError:
            model_class = AutoModelForCausalLM

        with torch.cuda.device(self.device):
            self.model = model_class.from_pretrained(
                MODEL_ID,
                device_map=self.device,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

            self.processor = AutoProcessor.from_pretrained(
                MODEL_ID, trust_remote_code=True
            )
            self.processor.tokenizer.padding_side = "left"

        self._loaded = True
        print(f"{MODEL_ID} loaded on {self.device}")

    @overload
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
        max_thinking_tokens: int = 1024,
    ) -> str: ...
    @overload
    def generate(
        self,
        image: List[Image.Image],
        prompt: List[str],
        max_new_tokens: int = 256,
        max_thinking_tokens: int = 1024,
    ) -> List[str]: ...

    def generate(
        self,
        image: Union[Image.Image, List[Image.Image]],
        prompt: Union[str, List[str]],
        max_new_tokens: int = 256,
        max_thinking_tokens: int = 1024,
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

            budget_processor = ThinkingBudgetLogitsProcessor(
                self.processor.tokenizer, budget=max_thinking_tokens
            )

            # Generate
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens + max_thinking_tokens,
                    logits_processor=LogitsProcessorList([budget_processor]),
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
