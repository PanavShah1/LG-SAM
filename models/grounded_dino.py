from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

MODEL_ID = "IDEA-Research/grounding-dino-base"


class GroundedDINODetector:
    def __init__(
        self,
        device: Optional[torch.device | str] = None,
    ):
        """
        Initialize Grounded DINO detector.

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

        print("Loading Grounded DINO model...")
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        print(f"Grounded DINO model loaded on {self.device}")

    def _prepare_text(self, text_prompt: str) -> str:
        """
        Prepare text prompt for Grounded DINO.
        Text queries need to be lowercased + end with a dot.

        Args:
            text_prompt: Original text prompt

        Returns:
            Formatted text prompt
        """
        text = text_prompt.lower().strip()
        if not text.endswith("."):
            text += "."
        return text

    def detect(
        self,
        image: Union[Image.Image, np.ndarray],
        text_prompt: str,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Detect objects in an image using text prompt.

        Args:
            image: PIL Image or numpy array
            text_prompt: Text description of objects to detect
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching

        Returns:
            Tuple of (bounding_boxes, scores)
            - bounding_boxes: List of [x1, y1, x2, y2] in pixel coordinates
            - scores: List of confidence scores
        """
        if not self._loaded:
            self.load()

        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        # Prepare text prompt
        formatted_text = self._prepare_text(text_prompt)

        # Process inputs
        inputs = self.processor(images=image, text=formatted_text, return_tensors="pt")  # type: ignore[reportCallIssue]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        results = self.processor.post_process_grounded_object_detection(  # type: ignore[reportCallIssue]
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],  # (height, width) format
        )

        # Extract bounding boxes, scores, and labels
        if len(results) > 0 and len(results[0]["boxes"]) > 0:
            boxes = results[0]["boxes"].tolist()
            scores = results[0]["scores"].tolist()
            # labels = results[0]["text_labels"]

            return boxes, scores
        else:
            return [], []
