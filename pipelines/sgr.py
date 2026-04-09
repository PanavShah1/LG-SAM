from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from models.qwen3_vl_8b import Qwen3VL8B
from models.remotesam import RemoteSAM
from models.sam3 import SAM3Segmenter
from pipelines.base import BasePipeline
from prompts import EXTRACT_OBJECT_CLASS_PROMPT
from utils.bbox import M2B, hbb_to_corners
from utils.image import load_image


class SGRPipeline(BasePipeline):
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
        """Initialize RemoteSAM detector and SAM 3 segmenter."""
        self.qwen = Qwen3VL8B(device=self.device)
        self.detector = RemoteSAM(device=self.device)
        self.sam3 = SAM3Segmenter(device=self.device)

    def load_models(self):
        """Load all models."""
        if not self._models_loaded:
            self.qwen.load()
            self.detector.load()
            self.sam3.load()
            self._models_loaded = True

    def process_image(
        self,
        image: Union[str, Image.Image, np.ndarray],
        text_prompt: str,
        **kwargs,
    ) -> List[dict]:
        return self.process_batch(
            images=[image],
            text_prompts=[text_prompt],
            **kwargs,
        )[0]

    def process_batch(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        text_prompts: List[str],
        **kwargs,
    ) -> List[List[dict]]:
        """
        Process a batch of images using batched RemoteSAM detection.

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

        # Load images as PIL Images
        pil_images = [load_image(img) for img in images]

        object_classes = self.qwen.generate_text_only(
            [
                EXTRACT_OBJECT_CLASS_PROMPT.format(question=text_prompt)
                for text_prompt in text_prompts
            ],
        )

        # Run batched detection
        bboxes_list, scores_list = self.detector.detect_batch(
            images=pil_images,  # type: ignore
            text_prompts=text_prompts,
            bbox_type="hbb",
        )

        # Process each image's results (SAM3 segmentation is per-image)
        results = []
        for idx, (img, bboxes, scores) in enumerate(
            zip(pil_images, bboxes_list, scores_list)
        ):
            # For each image, segment each bbox with SAM3 and choose the best segment mask for each bbox
            image_results = []
            for bbox, score1 in zip(bboxes, scores):
                masks, probs, scores = self.sam3.segment_with_boxes(
                    img, [bbox], object_classes[idx], return_scores=True
                )
                if len(masks) == 0:
                    continue

                i = np.argmax(scores)
                mask = masks[i].astype(np.uint8)
                prob = probs[i]
                score2 = scores[i]
                oriented_bboxes = M2B(mask, prob, box_type="obb")
                if len(oriented_bboxes) > 0:
                    oriented_bbox = oriented_bboxes[0][:8]
                else:
                    oriented_bbox = hbb_to_corners(bbox)

                image_results.append(
                    {
                        "mask": mask,
                        "oriented_bbox": oriented_bbox,
                        "score": score1 * score2,
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
