from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from models.qwen3_vl import Qwen3VL
from models.qwen_image_edit import QwenImageEditGenerator
from models.remotesam import RemoteSAM
from models.sam3 import SAM3Segmenter
from pipelines.base import BasePipeline
from prompts import REWRITE_PROMPT
from utils.bbox import M2B, hbb_to_corners, compute_area_rectangle
from utils.image import load_image, remove_haziness


class QwenSGRPipeline(BasePipeline):
    """
    Pipeline that uses Qwen-Image-Edit to draw rectangles around objects,
    SAM3 to detect the drawn rectangles, and RemoteSAM to refine the segmentation
    on the cropped regions to get oriented bounding boxes.
    """

    def __init__(
        self,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Initialize the Qwen-SGR detection pipeline.

        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        super().__init__(device=device)

    def _initialize_models(self):
        """Initialize Qwen-Image-Edit generator, Qwen3-VL, SAM 3 segmenter, and RemoteSAM."""
        self.qwen_image_edit = QwenImageEditGenerator(device=self.device)
        self.sam3 = SAM3Segmenter(device=self.device)
        self.remotesam = RemoteSAM(device=self.device)
        self.qwen3vl = Qwen3VL(device=self.device)

    def load_models(self):
        """Load all models."""
        if not self._models_loaded:
            if hasattr(self, "qwen") and hasattr(self.qwen_image_edit, "load"):
                self.qwen_image_edit.load()
            if hasattr(self, "sam3") and hasattr(self.sam3, "load"):
                self.sam3.load()
            if hasattr(self, "remotesam") and hasattr(self.remotesam, "load"):
                self.remotesam.load()
            # Do not load Qwen3-VL eagerly as it is optional
            # if hasattr(self, "qwen3vl") and hasattr(self.qwen3vl, "load"):
            #     self.qwen3vl.load()
            self._models_loaded = True

    def process_image(
        self,
        image: Union[str, Image.Image, np.ndarray],
        text_prompt: str,
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        generator_seed: Optional[int] = None,
        rewrite_prompt: bool = False,
        crop_padding: int = 10,
        **kwargs,
    ) -> List[Dict]:
        """
        Process an image using Qwen, SAM3, and RemoteSAM.

        Args:
            image: Image input
            text_prompt: Text description
            num_inference_steps: Steps for Qwen
            true_cfg_scale: True CFG scale for Qwen
            generator_seed: Seed for Qwen
            rewrite_prompt: Whether to rewrite the prompt using Qwen3-VL
            crop_padding: Padding for crops
            **kwargs: Additional arguments

        Returns:
            List of result dictionaries.
        """
        if not self._models_loaded:
            self.load_models()

        pil_image = load_image(image)

        # Step 0: Batch rewrite prompts
        # if rewrite_prompt:
        #     if "{prompt}" in REWRITE_PROMPT:
        #         rewriting_prompt = REWRITE_PROMPT.format(prompt=text_prompt)
        #     else:
        #         rewriting_prompt = f"{REWRITE_PROMPT}\nOriginal Prompt: {text_prompt}"

        #     text_prompt = self.qwen3vl.generate(pil_image, rewriting_prompt)  # type: ignore

        # Step 1: Generate images with Qwen
        generated_image = self.qwen_image_edit.generate(
            image=remove_haziness(pil_image),
            prompt=text_prompt,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator_seed=generator_seed,
        )

        # Step 2: Detect rectangle using SAM3
        rectangle_masks, rectangle_probs = self.sam3.segment_text_only(
            image=generated_image, text_prompt="bright red rectangle"
        )
        rectangle_masks = [mask.astype(np.uint8) for mask in rectangle_masks]

        # Step 3: Process each image
        roi_bboxes = []
        w, h = pil_image.size
        gen_size = generated_image.size
        scale_x = w / gen_size[0] if gen_size[0] > 0 else 1.0
        scale_y = h / gen_size[1] if gen_size[1] > 0 else 1.0

        for mask, prob in zip(rectangle_masks, rectangle_probs):
            mask_bboxes = M2B(mask, prob, box_type="hbb")
            for bbox in mask_bboxes:
                x1, y1, x2, y2, _ = bbox
                x1_scaled = int(x1 * scale_x)
                y1_scaled = int(y1 * scale_y)
                x2_scaled = int(x2 * scale_x)
                y2_scaled = int(y2 * scale_y)
                roi_bboxes.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled])

        if not roi_bboxes:
            roi_bboxes = [[0, 0, w, h]]

        image_results = []

        for roi in roi_bboxes:
            roi_x1, roi_y1, roi_x2, roi_y2 = roi

            # Apply padding to get crop coordinates
            crop_x1 = max(0, roi_x1 - crop_padding)
            crop_y1 = max(0, roi_y1 - crop_padding)
            crop_x2 = min(w, roi_x2 + crop_padding)
            crop_y2 = min(h, roi_y2 + crop_padding)

            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue

            # Calculate expected crop dimensions
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1

            # Skip very small crops that could cause issues
            if crop_w < 2 or crop_h < 2:
                continue

            cropped_image = pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            # Calculate cropped image area as percentage of original
            original_area = w * h
            cropped_area = crop_w * crop_h
            crop_percentage = cropped_area / original_area if original_area > 0 else 1.0

            cropped_image = remove_haziness(cropped_image)

            # NOTE: We assume that each cropped region can have at max one object

            mask = None
            prob = None

            # Use SAM3 for small crops (< 20% of original), else use RemoteSAM
            if crop_percentage < 0.10:
                # Use SAM3 for small crops
                masks, probs, scores = self.sam3.segment_text_only(
                    image=cropped_image,
                    text_prompt=text_prompt.replace("right", "")
                    .replace("left", "")
                    .replace("top", "")
                    .replace("bottom", "")
                    .replace("center", "")
                    .replace("middle", "")
                    .replace("-", "")
                    .strip(),
                    return_scores=True,
                )
                masks = masks.astype(np.uint8)
                if len(masks) > 0:
                    i = np.argmax(scores)
                    mask = masks[i]
                    prob = probs[i]
            else:
                # Use RemoteSAM for larger crops
                masks, probs = self.remotesam.remote_sam.referring_seg_batch(
                    images=[cropped_image],
                    text_prompts=[text_prompt],
                    return_prob=True,
                )
                if len(masks) > 0:
                    mask = masks[0]
                    prob = probs[0]

            if len(masks) == 0:
                res = {
                    "oriented_bbox": hbb_to_corners([roi_x1, roi_y1, roi_x2, roi_y2]),
                    "score": 1.0,
                    "mask": None,
                    "generated_image": generated_image,
                }
                image_results.append(res)
                continue

            assert mask is not None and prob is not None

            # Get mask dimensions and compute scale factors
            # (mask might be different size than crop due to model internal resizing)
            h_mask, w_mask = mask.shape
            mask_scale_x = crop_w / w_mask if w_mask > 0 else 1.0
            mask_scale_y = crop_h / h_mask if h_mask > 0 else 1.0

            obb_boxes = M2B(mask, prob, box_type="obb")
            for obb in obb_boxes:
                # obb: [px1, py1, ..., px4, py4, conf]
                # Scale OBB coordinates from mask space to crop space, then translate to original image
                shifted_points = []
                for i in range(0, 8, 2):
                    # Scale from mask coordinates to crop coordinates
                    crop_px = obb[i] * mask_scale_x
                    crop_py = obb[i + 1] * mask_scale_y
                    # Translate from crop coordinates to original image coordinates
                    orig_px = crop_px + crop_x1
                    orig_py = crop_py + crop_y1
                    shifted_points.extend([orig_px, orig_py])

                conf = obb[8]

                # Create result dict
                res = {
                    "oriented_bbox": shifted_points,
                    "score": conf,
                    "mask": None,
                    "generated_image": generated_image,
                }

                # Map mask back to full size
                # Resize mask to match crop dimensions if needed
                full_mask = np.zeros((h, w), dtype=np.uint8)
                if mask_scale_x != 1.0 or mask_scale_y != 1.0:
                    # Resize mask to crop dimensions
                    mask_pil = Image.fromarray(mask)
                    mask_resized = mask_pil.resize(
                        (crop_w, crop_h), Image.Resampling.NEAREST
                    )
                    mask_resized = np.array(mask_resized)
                else:
                    mask_resized = mask

                # Place resized mask into full image
                h_resized, w_resized = mask_resized.shape
                y_end = min(crop_y1 + h_resized, h)
                x_end = min(crop_x1 + w_resized, w)

                full_mask[crop_y1:y_end, crop_x1:x_end] = mask_resized[
                    : (y_end - crop_y1), : (x_end - crop_x1)
                ]
                res["mask"] = full_mask

                image_results.append(res)

        if not image_results:
            image_results.append(
                {
                    "oriented_bbox": hbb_to_corners([0, 0, w, h]),
                    "score": 1.0,
                    "mask": None,
                    "generated_image": generated_image,
                }
            )
            
        image_results_final = []
        max_area = max(
            image_results, key=lambda x: compute_area_rectangle(x["oriented_bbox"])
        )
        for image_result in image_results:
            oriented_bbox = image_result["oriented_bbox"]
            if compute_area_rectangle(oriented_bbox) >= 0.3 * compute_area_rectangle(
                max_area["oriented_bbox"]
            ):
                image_results_final.append(image_result)

        return image_results_final

    def process_batch(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        text_prompts: List[str],
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        generator_seed: Optional[int] = None,
        rewrite_prompt: bool = True,
        crop_padding: int = 10,
        **kwargs,
    ) -> List[List[Dict]]:
        """
        Process a batch of images using Qwen, SAM3, and RemoteSAM.
        NOTE: This method currently just calls the process_image method for each
        image in the batch since qwen image edit does not support batch
        generation.

        Args:
            images: List of image inputs
            text_prompts: List of text descriptions
            num_inference_steps: Steps for Qwen
            true_cfg_scale: True CFG scale for Qwen
            generator_seed: Seed for Qwen
            rewrite_prompt: Whether to rewrite the prompt using Qwen3-VL
            crop_padding: Padding for crops
            **kwargs: Additional arguments

        Returns:
            List of lists of result dictionaries.
        """

        if len(images) != len(text_prompts):
            raise ValueError("Number of images and text prompts must match.")

        # Convert all images to PIL
        pil_images = [load_image(img) for img in images]

        results = []
        for pil_image, text_prompt in zip(pil_images, text_prompts):
            results.append(
                self.process_image(
                    image=pil_image,
                    text_prompt=text_prompt,
                    num_inference_steps=num_inference_steps,
                    true_cfg_scale=true_cfg_scale,
                    generator_seed=generator_seed,
                    rewrite_prompt=rewrite_prompt,
                    crop_padding=crop_padding,
                )
            )

        return results
