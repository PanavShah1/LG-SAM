from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from models.falcon import Falcon
from pipelines.base import BasePipeline
from utils.image import load_image


class FalconPipeline(BasePipeline):
    """Wrapper for Falcon visual grounding model."""

    def __init__(self, device: Optional[torch.device | str] = None):
        """
        Initialize SAM3 segmenter.

        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        super().__init__(device=device)

    def _initialize_models(self):
        self.model = Falcon(device=self.device)

    def process_image(
        self,
        image: Union[str | Image.Image, np.ndarray],
        text_prompt: str,
        **kwargs,
    ) -> List[dict]:
        """
        Process a single image and text prompt to generate segmentation masks.
        """
        return self.process_batch(
            images=[image],
            text_prompts=[text_prompt],
            **kwargs,
        )[0]

    def process_batch(
        self,
        images: List[Union[str | Image.Image, np.ndarray]],
        text_prompts: List[str],
        **kwargs,
    ) -> List[List[dict]]:
        """
        Process a batch of images and text prompts to generate segmentation masks.

        Args:
            images: List of PIL Images or numpy arrays
            text_prompts: List of text descriptions for each image
        Returns:
            List of lists of dictionaries containing 'mask' and 'prob' for each image
        """

        if not self._models_loaded:
            self.load_models()

        images = [load_image(image) for image in images]

        bboxes_list = self.model.detect_batch(
            images,  # type: ignore
            text_prompts,
        )

        results = []
        for bboxes in bboxes_list:
            image_results = []
            for bbox in bboxes:
                image_results.append(
                    {
                        "oriented_bbox": bbox,
                        "mask": None,
                        "score": 0,
                    }
                )
            results.append(image_results)
        return results
