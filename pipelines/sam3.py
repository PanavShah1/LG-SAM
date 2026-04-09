from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from models.sam3 import SAM3Segmenter
from pipelines.base import BasePipeline
from utils.bbox import M2BSingle, hbb_to_corners
from utils.image import load_image


class SAM3Pipeline(BasePipeline):
    """Wrapper for SAM3 segmentation model."""

    def __init__(self, device: Optional[torch.device | str] = None):
        """
        Initialize SAM3 segmenter.

        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        super().__init__(device=device)

    def _initialize_models(self):
        """
        Initialize detector and segmenter models.
        This method should be implemented by subclasses to set up:
        - self.detector: The detection model
        - self.sam3: The SAM3Segmenter instance
        """
        self.model = SAM3Segmenter(device=self.device)

    def process_image(
        self,
        image: Union[str | Image.Image, np.ndarray],
        text_prompt: str,
        **kwargs,
    ) -> List[dict]:
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

        image = load_image(image)

        masks, probs, scores = self.model.segment_text_only(
            image,  # type: ignore
            text_prompt,
            return_scores=True,
        )
        masks = masks.astype(np.uint8)

        results = []
        for mask, prob, score in zip(masks, probs, scores):
            bbox = M2BSingle(mask, prob, box_type="obb")
            if bbox is not None:
                results.append(
                    {
                        "oriented_bbox": bbox[:8],
                        "mask": mask,
                        "score": score * bbox[8],
                    }
                )

        if len(results) == 0:
            h, w = image.size[1], image.size[0]
            return [
                {
                    "oriented_bbox": hbb_to_corners([0, 0, w, h]),
                    "mask": None,
                    "score": 0.0,
                }
            ]

        return results
