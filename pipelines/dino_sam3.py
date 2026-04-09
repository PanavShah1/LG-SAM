from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from models.grounded_dino import GroundedDINODetector
from models.sam3 import SAM3Segmenter
from pipelines.base import BasePipeline


class DINOSAM3Pipeline(BasePipeline):
    def __init__(
        self,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Initialize the detection pipeline.

        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        super().__init__(device=device)

    def _initialize_models(self):
        """Initialize Grounded DINO detector and SAM 3 segmenter."""
        self.detector = GroundedDINODetector(device=self.device)
        self.sam3 = SAM3Segmenter(device=self.device)

    def _detect_objects(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
        **kwargs,
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Detect objects in an image using Grounded DINO.

        Args:
            image: PIL Image in RGB format
            text_prompt: Text description of objects to detect (required)
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching

        Returns:
            Tuple of (bboxes, scores) where:
            - bboxes: List of bounding boxes, each as [x1, y1, x2, y2]
            - scores: List of confidence scores
        """
        bboxes, scores = self.detector.detect(
            image, text_prompt, box_threshold, text_threshold
        )
        return bboxes, scores

    def process_image(
        self,
        image: Union[str, Image.Image, np.ndarray],
        text_prompt: str,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
        **kwargs,
    ) -> List[Dict]:
        """
        Process a single image and return oriented bounding boxes.

        Args:
            image: Image input (file path, PIL Image, or numpy array)
            text_prompt: Text description of objects to detect (required)
            box_threshold: Confidence threshold for Grounded DINO bounding boxes
            text_threshold: Confidence threshold for Grounded DINO text matching

        Returns:
            List of dictionaries, each containing:
            - 'mask': Binary segmentation mask as numpy array
            - 'oriented_bbox': (center_x, center_y, width, height, angle_degrees)
            - 'score': Confidence score from Grounded DINO
        """
        return super().process_image(
            image,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
