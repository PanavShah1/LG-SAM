from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from models.remotesam import RemoteSAM
from pipelines.base import BasePipeline
from utils.bbox import compute_area_rectangle, hbb_to_corners
from utils.image import load_image


class RemoteSAMPipeline(BasePipeline):
    """Wrapper for RemoteSAM segmentation model."""

    def __init__(self, device: Optional[torch.device | str] = None):
        """
        Initialize RemoteSAM pipeline.

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
        self.model = RemoteSAM(device=self.device)

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

        if len(images) != len(text_prompts):
            raise ValueError("Number of images must match number of text prompts.")

        images = [load_image(img) for img in images]

        bboxes_list, scores_list = self.model.detect_batch(
            images,  # type: ignore
            text_prompts,
            bbox_type="obb",
        )

        results = []

        for img, bboxes, scores in zip(images, bboxes_list, scores_list):
            area_idx = []
            for bbox, score in zip(bboxes, scores):
                area_idx.append(compute_area_rectangle(bbox))

            max_area = max(area_idx) if area_idx else 0.0
            # Take only boxes with area >= 30% of max area
            filtered_bboxes = []
            filtered_scores = []
            for bbox, score, area in zip(bboxes, scores, area_idx):
                if area >= 0.3 * max_area:
                    filtered_bboxes.append(bbox)
                    filtered_scores.append(score)

            image_results = []
            for bbox, score in zip(filtered_bboxes, filtered_scores):
                image_results.append(
                    {
                        "oriented_bbox": bbox,
                        "mask": None,
                        "score": score,
                    }
                )
            if len(image_results) == 0:
                image_results.append(
                    {
                        "oriented_bbox": hbb_to_corners([0, 0, img.width, img.height]),
                        "mask": None,
                        "score": 0.0,
                    }
                )
            results.append(image_results)

        return results
