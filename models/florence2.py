from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

MODEL_ID = "florence-community/Florence-2-large-ft"


class Florence2Detector:
    def __init__(
        self,
        device: Optional[torch.device | str] = None,
    ):
        """
        Initialize Florence-2 detector.

        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._loaded = False

    def load(self):
        """Load the model and processor from HuggingFace."""
        if self._loaded:
            return

        print("Loading Florence-2 model...")
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, trust_remote_code=True, use_fast=True
        )
        self.model = Florence2ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        print(f"Florence-2 model loaded on {self.device}")

    def _run_florence2(
        self,
        task_prompt: str,
        text_input: Optional[str],
        image: Image.Image,
    ) -> dict:
        """
        Run Florence-2 model inference.

        Args:
            task_prompt: Task prompt (e.g., "<OPEN_VOCABULARY_DETECTION>")
            text_input: Optional text input for the task
            image: PIL Image

        Returns:
            Parsed results dictionary
        """
        if not self._loaded:
            self.load()

        # Construct prompt
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        # Process inputs
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(  # type: ignore[reportCallIssue]
            self.device, torch.bfloat16
        )

        # Run inference
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )

        # Decode and post-process
        generated_text = self.processor.batch_decode(  # type: ignore[reportCallIssue]
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self.processor.post_process_generation(  # type: ignore[reportCallIssue]
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )

        return parsed_answer

    def detect(
        self,
        image: Union[Image.Image, np.ndarray],
        text_prompt: Optional[str] = None,
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Detect objects in an image using text prompt.

        Args:
            image: PIL Image or numpy array
            text_prompt: Text description of objects to detect (required for open_vocabulary_detection)

        Returns:
            Tuple of (bounding_boxes, scores)
            - bounding_boxes: List of [x1, y1, x2, y2] in pixel coordinates
            - scores: List of confidence scores (default 1.0 for Florence-2)
        """
        if not self._loaded:
            self.load()

        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        # Get task prompt
        task_prompt = "<OPEN_VOCABULARY_DETECTION>"

        # Run Florence-2
        results = self._run_florence2(task_prompt, text_prompt, image)
        bboxes = results[task_prompt]["bboxes"]
        # labels = results[task_prompt]["bboxes_labels"]

        boxes = [[float(b) for b in bbox] for bbox in bboxes]

        # Florence-2 doesn't provide scores, so we use default scores
        scores = [1.0] * len(boxes)

        return boxes, scores
