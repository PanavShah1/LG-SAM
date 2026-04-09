from typing import List, Optional, Tuple, Union

import torch
from PIL import Image

from models.florence2 import Florence2Detector
from models.sam3 import SAM3Segmenter
from pipelines.base import BasePipeline


class Florence2SAM3Pipeline(BasePipeline):
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
        """Initialize Florence-2 detector and SAM 3 segmenter."""
        self.detector = Florence2Detector(device=self.device)
        self.sam3 = SAM3Segmenter(device=self.device)

    def _detect_objects(
        self,
        image: Image.Image,
        text_prompt: str,
        **kwargs,
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Detect objects in an image using Florence-2.

        Args:
            image: PIL Image in RGB format
            text_prompt: Optional text description of objects to detect
            **kwargs: Additional arguments (not used by Florence-2)

        Returns:
            Tuple of (bboxes, scores) where:
            - bboxes: List of bounding boxes, each as [x1, y1, x2, y2]
            - scores: List of confidence scores
        """
        bboxes, scores = self.detector.detect(image, text_prompt)
        return bboxes, scores
