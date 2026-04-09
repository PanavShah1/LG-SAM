from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

import config
from utils.image import load_image

CHECKPOINT = config.EARTHMIND_CHECKPOINT
FINETUNED_CHECKPOINT = config.EARTHMIND_FT_CHECKPOINT


def _contour_to_obb(contour):
    """
    Convert a single contour to oriented bounding box corners.

    Args:
        contour: OpenCV contour

    Returns:
        corners: numpy array of shape (4, 2) with corner coordinates
    """
    if len(contour) < 5:
        # Not enough points for minAreaRect, use bounding rect
        x, y, w, h = cv2.boundingRect(contour)
        corners = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
        )
    else:
        # Get minimum area rotated rectangle
        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect)
    return corners


def _order_corners_clockwise(corners):
    """
    Order 4 corners in clockwise order starting from top-left.

    Args:
        corners: Array of shape (4, 2) with corner coordinates

    Returns:
        Ordered corners array (4, 2)
    """
    corners = np.array(corners)

    # Calculate centroid
    centroid = np.mean(corners, axis=0)

    # Calculate angles from centroid
    angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])

    # Sort by angle (clockwise means decreasing angle from top)
    # Adjust to start from top-left (-135 to -180 degrees or so)
    sorted_indices = np.argsort(angles)

    # Reorder to start from top-left (smallest y, then smallest x)
    sorted_corners = corners[sorted_indices]

    # Find top-left: minimum sum of x + y
    sums = sorted_corners[:, 0] + sorted_corners[:, 1]
    top_left_idx = np.argmin(sums)

    # Rotate array to start from top-left and go clockwise
    ordered = np.roll(sorted_corners, -top_left_idx, axis=0)

    # Verify clockwise order (cross product should be negative for clockwise)
    # If not clockwise, reverse the order (except first point)
    v1 = ordered[1] - ordered[0]
    v2 = ordered[2] - ordered[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]

    if cross > 0:  # Counter-clockwise, need to reverse
        ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]])

    return ordered


# NOTE: EarthMind's implementation of mask_to_oriented_bbox is slightly
# different from what we are using in other places from utils.bbox
def _mask_to_oriented_bboxes(mask, image_width, image_height, min_area=100):
    """
    Convert a binary mask to oriented bounding boxes for ALL objects in the mask.
    Each connected component (contour) gets its own OBB.

    Args:
        mask: Binary mask (numpy array)
        image_width: Original image width
        image_height: Original image height
        min_area: Minimum contour area to consider (filters noise)

    Returns:
        List of OBBs, each OBB is [x1,y1, x2,y2, x3,y3, x4,y4] in clockwise order (integers)
    """
    # Ensure mask is binary uint8
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    # Resize mask to original image dimensions if needed
    mask_h, mask_w = mask.shape[:2]
    if mask_w != image_width or mask_h != image_height:
        mask = cv2.resize(
            mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST
        )

    # Find ALL contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    all_obbs = []

    # Process each contour (each separate object in the mask)
    for contour in contours:
        # Filter out tiny contours (noise)
        if cv2.contourArea(contour) < min_area:
            continue

        corners = _contour_to_obb(contour)

        # Order corners clockwise starting from top-left
        corners = _order_corners_clockwise(corners)

        # Convert to pixel coordinates as integers
        pixel_corners = []
        for corner in corners:
            x_px = int(round(float(corner[0])))
            y_px = int(round(float(corner[1])))
            pixel_corners.extend([x_px, y_px])

        all_obbs.append(pixel_corners)

    # Sort by area (largest first) for consistency
    all_obbs.sort(key=lambda obb: -((obb[2] - obb[0]) ** 2 + (obb[3] - obb[1]) ** 2))

    return all_obbs


def _merge_masks(masks, image_width, image_height):
    """
    Merge multiple binary masks into a single mask using logical OR.
    This prevents counting the same region multiple times when multiple
    <seg> tokens segment overlapping regions.

    Args:
        masks: List of binary masks
        image_width: Target image width
        image_height: Target image height

    Returns:
        Single merged binary mask (numpy array)
    """
    # Initialize empty mask
    merged = np.zeros((image_height, image_width), dtype=np.uint8)

    for mask in masks:
        # Ensure mask is binary uint8
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255

        # Resize mask to target dimensions if needed
        mask_h, mask_w = mask.shape[:2]
        if mask_w != image_width or mask_h != image_height:
            mask = cv2.resize(
                mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST
            )

        # Merge using logical OR
        merged = np.maximum(merged, mask)

    return merged


class EarthMind:
    """EarthMind segmentation model wrapper."""

    def __init__(
        self, device: Optional[torch.device | str] = None, load_finetuned: bool = False
    ):
        """
        Initialize EarthMind segmenter.

        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._loaded = False
        self.load_finetuned = load_finetuned

    def load(self):
        """
        Load EarthMind model.
        """
        if self._loaded:
            return

        print("Loading EarthMind model...")
        with torch.cuda.device(self.device):
            self.model = AutoModel.from_pretrained(
                FINETUNED_CHECKPOINT if self.load_finetuned else CHECKPOINT,
                torch_dtype="auto",
                trust_remote_code=True,
                use_flash_attn=True,
                device_map=self.device,
            )
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(
                FINETUNED_CHECKPOINT if self.load_finetuned else CHECKPOINT,
                trust_remote_code=True,
            )
        print(f"EarthMind model loaded on {self.device}")
        self._loaded = True

    def detect(
        self,
        image: Union[Image.Image, np.ndarray],
        text_prompt: str,
    ) -> Tuple[List[List[float]], List[np.ndarray]]:
        """
        Detect objects in an image using text prompt.
        """
        if not self._loaded:
            self.load()

        pil_image = load_image(image)
        image_width, image_height = pil_image.size
        with torch.cuda.device(self.device):
            result = self.model.predict_forward(
                image=pil_image,
                text=f"<image>Please segment {text_prompt}",
                tokenizer=self.tokenizer,
            )
        masks = []
        if result.get("prediction_masks") and len(result["prediction_masks"]) > 0:
            for pred_mask in result["prediction_masks"]:
                if isinstance(pred_mask, (list, np.ndarray)) and len(pred_mask) > 0:
                    mask = pred_mask[0]
                    masks.append(mask)

        if len(masks) == 0:
            return [], []

        merged_mask = _merge_masks(masks, image_width, image_height)
        bboxes = _mask_to_oriented_bboxes(merged_mask, image_width, image_height)
        return bboxes, [merged_mask] * len(bboxes)

    def answer(
        self,
        image: Union[Image.Image, np.ndarray],
        text_prompt: str,
    ) -> str:
        """
        Answer a question about the image using text prompt.
        """
        if not self._loaded:
            self.load()

        pil_image = load_image(image)
        with torch.cuda.device(self.device):
            result = self.model.predict_forward(
                image=pil_image,
                text=f"<image>{text_prompt}",
                tokenizer=self.tokenizer,
            )
        return (
            result["prediction"]
            .replace("<|end|>", "")
            .replace("<|endoftext|>", "")
            .strip()
        )
