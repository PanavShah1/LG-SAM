from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from models.lqvg import LQVGSegmenter
from pipelines.base import BasePipeline
from utils.image import load_image


class LQVGPipeline(BasePipeline):
    """Wrapper for LQVG visual grounding model."""

    def __init__(self, device: Optional[torch.device | str] = None):
        """
        Initialize LQVG segmenter.

        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        super().__init__(device=device)

    def _initialize_models(self):
        self.model = LQVGSegmenter(device=self.device)
        self.model.load()

    def process_image(
        self,
        image: Union[str | Image.Image, np.ndarray],
        text_prompt: str,
        **kwargs,
    ) -> List[dict]:
        """
        Process a single image and text prompt to generate segmentation masks.
        """
        bbox = self.model.get_bbox(
            image=image,
            text_prompt=text_prompt,
            args=kwargs.get('args', None),
        )
        
        print("Predicted bbox (x1,y1,x2,y2):", bbox)
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        
        
        results = []
        results.append({
            "oriented_bbox": [x1, y1, x2, y1, x2, y2, x1, y2],
            "mask": None,
            "score": 0.0,
        })
        
        return results
