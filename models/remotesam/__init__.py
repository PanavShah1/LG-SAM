from typing import List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from utils.bbox import M2B

from .model import RemoteSAM as RemoteSAMWrapper
from .model import init_demo_model

CHECKPOINT = "./checkpoints/RemoteSAMv1.pth"


class RemoteSAM:
    def __init__(
        self,
        device: Optional[torch.device | str] = None,
    ):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._loaded = False

    def load(self):
        """Load the model."""
        if self._loaded:
            return

        print("Loading RemoteSAM model...")
        self.model = init_demo_model(CHECKPOINT, self.device)
        self.model.to(self.device)
        self.model.eval()

        self.remote_sam = RemoteSAMWrapper(self.model, self.device)

        self._loaded = True
        print(f"RemoteSAM model loaded on {self.device}")

    def detect(
        self,
        image: Union[Image.Image, np.ndarray],
        text_prompt: str,
        bbox_type: Literal["hbb", "obb"] = "obb",
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Detect objects in an image using text prompt.

        Args:
            image: PIL Image or numpy array
            text_prompt: Text description of objects to detect

        Returns:
            Tuple of (bounding_boxes, scores)
            - bounding_boxes: List of [center_x, center_y, width, height] in pixel coordinates
            - scores: List of confidence scores
        """
        boxes, scores = self.detect_batch(
            images=[image],
            text_prompts=[text_prompt],
            bbox_type=bbox_type,
        )
        return boxes[0], scores[0]

    def detect_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        text_prompts: List[str],
        bbox_type: Literal["hbb", "obb"] = "obb",
    ) -> Tuple[List[List[List[float]]], List[List[float]]]:
        """
        Detect objects in a batch of images using text prompts.

        Args:
            images: List of PIL Images or numpy arrays
            text_prompts: List of text descriptions of objects to detect

        Returns:
            Tuple of (bounding_boxes_list, scores_list) where:
            - bounding_boxes_list: List of lists, each containing [x1, y1, x2, y2] boxes
            - scores_list: List of lists, each containing confidence scores
        """
        assert bbox_type in ["hbb", "obb"], "Invalid bbox type"

        if not self._loaded:
            self.load()

        if len(images) != len(text_prompts):
            raise ValueError("Number of images and text prompts must match")

        # Convert numpy arrays to PIL Images if needed
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_img = Image.fromarray(img).convert("RGB")
            elif not isinstance(img, Image.Image):
                pil_img = Image.open(img).convert("RGB")
            else:
                pil_img = img
            pil_images.append(pil_img)

        # Run batched inference
        masks, probs = self.remote_sam.referring_seg_batch(
            pil_images, text_prompts, return_prob=True
        )

        # Format results
        bboxes_list = []
        scores_list = []
        for mask, prob in zip(masks, probs):
            boxes_with_conf = M2B(mask, prob, box_type=bbox_type)
            boxes = []
            scores = []
            for box in boxes_with_conf:
                if bbox_type == "obb":
                    boxes.append(box[:8])
                    scores.append(box[8])
                elif bbox_type == "hbb":
                    boxes.append(box[:4])
                    scores.append(box[4])

            bboxes_list.append(boxes)
            scores_list.append(scores)
        return bboxes_list, scores_list
