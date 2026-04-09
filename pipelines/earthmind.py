from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from models.earthmind import EarthMind
from pipelines.base import BasePipeline
from utils.bbox import hbb_to_corners
from utils.image import load_image


class EarthMindPipeline(BasePipeline):
    """Wrapper for EarthMind grounding model."""

    def __init__(self, device: Optional[torch.device | str] = None):
        """
        Initialize EarthMind pipeline.

        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        super().__init__(device=device)

    def _initialize_models(self):
        """
        Initialize EarthMind model.
        """
        self.model = EarthMind(device=self.device)

    def process_image(
        self,
        image: Union[str | Image.Image, np.ndarray],
        text_prompt: str,
        **kwargs,
    ) -> List[dict]:
        """
        Process a single image and text prompt to generate oriented bounding boxes.

        Args:
            image: PIL Image or numpy array
            text_prompt: Text description of object to detect
        Returns:
            List of dictionaries containing 'oriented_bbox' and 'score' for the object
        """
        image = load_image(image)

        bboxes, masks = self.model.detect(image=image, text_prompt=text_prompt)

        if not bboxes:
            print("Warning: No objects detected!")
            return [
                {
                    "oriented_bbox": hbb_to_corners([0, 0, image.width, image.height]),
                    "mask": None,
                    "score": 0.0,
                }
            ]

        results = []
        for bbox, mask in zip(bboxes, masks):
            results.append(
                {
                    "oriented_bbox": bbox,
                    "mask": mask,
                    "score": 0.0,
                }
            )

        return results
