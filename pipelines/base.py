from abc import ABC
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from utils.bbox import M2B, hbb_to_corners
from utils.image import load_image


class BasePipeline(ABC):
    def __init__(
        self,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Initialize the detection pipeline.

        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._models_loaded = False
        self._initialize_models()

    def _initialize_models(self):
        """
        Initialize detector and segmenter models.
        This method should be implemented by subclasses to set up:
        - self.detector: The detection model
        - self.sam3: The SAM3Segmenter instance
        """
        raise NotImplementedError("This method is not implemented")

    def load_models(self):
        """Load all models. Override in subclass if custom loading logic is needed."""
        if not self._models_loaded:
            if hasattr(self, "detector") and hasattr(self.detector, "load"):
                self.detector.load()
            if hasattr(self, "sam3") and hasattr(self.sam3, "load"):
                self.sam3.load()
            if hasattr(self, "model") and hasattr(self.model, "load"):
                self.model.load()
            self._models_loaded = True

    def _detect_objects(
        self,
        image: Image.Image,
        text_prompt: str,
        **kwargs,
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Detect objects in an image using the detector model.

        Args:
            image: PIL Image in RGB format
            text_prompt: Optional text description of objects to detect
            **kwargs: Additional arguments for detection (e.g., thresholds)

        Returns:
            Tuple of (bboxes, scores) where:
            - bboxes: List of bounding boxes, each as [x1, y1, x2, y2]
            - scores: List of confidence scores
        """
        raise NotImplementedError("This method is not implemented")

    def _calculate_oriented_bboxes(
        self,
        bboxes: List[List[float]],
        masks: Union[List[np.ndarray], np.ndarray],
        probs: Union[List[np.ndarray], np.ndarray],
    ) -> List[List[float]]:
        """
        Calculate oriented bounding boxes from masks.

        Args:
            bboxes: List of initial bounding boxes [x1, y1, x2, y2]
            masks: List of binary segmentation masks
            probs: List of per-pixel confidence maps aligned with masks

        Returns:
            List of oriented bounding boxes, each as (center_x, center_y, width, height, angle_degrees)
        """
        oriented_bboxes_list = []
        for idx, (bbox, mask, prob) in enumerate(zip(bboxes, masks, probs)):
            # Calculate oriented bbox from mask
            oriented_bboxes = M2B(mask, prob, box_type="obb")

            # Use the first (largest) oriented bbox if multiple found
            if len(oriented_bboxes) > 0:
                oriented_bbox = oriented_bboxes[0]
            else:
                # Fallback to axis-aligned bbox if no oriented bbox found
                oriented_bbox = hbb_to_corners(bbox)

            oriented_bboxes_list.append(oriented_bbox)

        return oriented_bboxes_list

    def _create_result_dicts(
        self,
        oriented_bboxes: List[List[float]],
        masks: List[np.ndarray],
        scores: List[float],
    ) -> List[Dict]:
        """
        Create result dictionaries from detection outputs.

        Args:
            oriented_bboxes: List of oriented bounding boxes
            masks: List of binary segmentation masks
            scores: List of confidence scores

        Returns:
            List of dictionaries, each containing:
            - 'mask': Binary segmentation mask as numpy array
            - 'oriented_bbox': (center_x, center_y, width, height, angle_degrees)
            - 'score': Confidence score
        """
        results = []
        for idx, (oriented_bbox, mask, score) in enumerate(
            zip(oriented_bboxes, masks, scores)
        ):
            result = {
                "mask": mask,
                "oriented_bbox": oriented_bbox,
                "score": score,
            }
            results.append(result)
        return results

    def process_image(
        self,
        image: Union[str, Image.Image, np.ndarray],
        text_prompt: str,
        **kwargs,
    ) -> List[Dict]:
        """
        Process a single image and return oriented bounding boxes.

        Args:
            image: Image input (file path, PIL Image, or numpy array)
            text_prompt: Optional text description of objects to detect
            **kwargs: Additional arguments passed to _detect_objects

        Returns:
            List of dictionaries, each containing:
            - 'mask': Binary segmentation mask as numpy array
            - 'oriented_bbox': (center_x, center_y, width, height, angle_degrees)
            - 'score': Confidence score
        """
        if not self._models_loaded:
            self.load_models()

        pil_image = load_image(image)

        bboxes, scores = self._detect_objects(
            pil_image, text_prompt=text_prompt, **kwargs
        )

        if len(bboxes) == 0:
            return []

        if not hasattr(self, "sam3"):
            raise AttributeError(
                "sam3 attribute not found. Ensure _initialize_models() sets self.sam3"
            )
        masks, probs, scores = self.sam3.segment(
            pil_image, bboxes, text_prompt, return_scores=True
        )

        oriented_bboxes = self._calculate_oriented_bboxes(bboxes, masks, probs)
        results = self._create_result_dicts(oriented_bboxes, masks, scores)

        return results

    def process_batch(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        text_prompts: List[str],
        **kwargs,
    ) -> List[List[Dict]]:
        """
        Process a batch of images and return oriented bounding boxes for each.

        Args:
            images: List of image inputs (file path, PIL Image, or numpy array)
            text_prompts: List of text descriptions of objects to detect
            **kwargs: Additional arguments passed to _detect_objects

        Returns:
            List of lists of dictionaries, where each inner list contains results for one image.
        """
        if not self._models_loaded:
            self.load_models()

        if len(images) != len(text_prompts):
            raise ValueError("Number of images and text prompts must match.")

        # Default implementation: loop through images
        # Subclasses can override this for true batch processing
        results = []
        for img, prompt in zip(images, text_prompts):
            results.append(self.process_image(img, prompt, **kwargs))

        return results
