from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from models.qwen3_vl import Qwen3VL
from models.qwen_image_edit import QwenImageEditGenerator
from models.sam3 import SAM3Segmenter
from pipelines.base import BasePipeline
from prompts import REWRITE_PROMPT
from utils.bbox import mask_to_bbox
from utils.image import load_image


class QwenSAM3Pipeline(BasePipeline):
    """
    Pipeline that uses Qwen-Image-Edit to draw rectangles around objects,
    then SAM3 to detect and segment them.
    """

    def __init__(
        self,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Initialize the Qwen-SAM3 detection pipeline.

        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        super().__init__(device=device)

    def _initialize_models(self):
        """Initialize Qwen-Image-Edit generator, Qwen3-VL, and SAM 3 segmenter."""
        self.qwen = QwenImageEditGenerator(device=self.device)
        self.sam3 = SAM3Segmenter(device=self.device)
        self.qwen3vl = Qwen3VL(device=self.device)

    def load_models(self):
        """Load all models."""
        if not self._models_loaded:
            if hasattr(self, "qwen") and hasattr(self.qwen, "load"):
                self.qwen.load()
            if hasattr(self, "sam3") and hasattr(self.sam3, "load"):
                self.sam3.load()
            # Do not load Qwen3-VL eagerly as it is optional
            # if hasattr(self, "qwen3vl") and hasattr(self.qwen3vl, "load"):
            #     self.qwen3vl.load()
            self._models_loaded = True

    def process_batch(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        text_prompts: List[str],
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        generator_seed: Optional[int] = None,
        rewrite_prompt: bool = True,
        **kwargs,
    ) -> List[List[Dict]]:
        """
        Process a batch of images using Qwen and SAM3 with optimized batch generation.

        Args:
            images: List of image inputs
            text_prompts: List of text descriptions
            num_inference_steps: Steps for Qwen
            true_cfg_scale: True CFG scale for Qwen Image Edit
            generator_seed: Seed for Qwen
            rewrite_prompt: Whether to rewrite the prompt using Qwen3-VL
            **kwargs: Additional arguments

        Returns:
            List of lists of result dictionaries.
        """
        if not self._models_loaded:
            self.load_models()

        if len(images) != len(text_prompts):
            raise ValueError("Number of images and text prompts must match.")

        # Convert all images to PIL
        pil_images = [load_image(img) for img in images]

        # Step 0: Batch rewrite prompts
        if rewrite_prompt:
            # Construct full prompts for rewriting
            rewriting_prompts = []
            for p in text_prompts:
                if "{prompt}" in REWRITE_PROMPT:
                    rewriting_prompts.append(REWRITE_PROMPT.format(prompt=p))
                else:
                    rewriting_prompts.append(f"{REWRITE_PROMPT}\nOriginal Prompt: {p}")

            text_prompts = self.qwen3vl.generate(pil_images, rewriting_prompts)
            if isinstance(text_prompts, str):
                text_prompts = [text_prompts]

        # Step 1: Generate images with Qwen (qwen image doesn't support batch generation)
        generated_images = []
        for pil_image, text_prompt in zip(pil_images, text_prompts):
            generated_image = self.qwen.generate(
                image=pil_image,
                prompt=text_prompt,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                generator_seed=generator_seed,
            )
            generated_images.append(generated_image)

        # If only one image was generated (edge case handling in generate), wrap it
        if not isinstance(generated_images, list):
            generated_images = [generated_images]

        results_batch = []

        # Step 2: Process each image sequentially for SAM3 parts
        for i, (pil_image, generated_image, prompt) in enumerate(
            zip(pil_images, generated_images, text_prompts)
        ):
            # Use SAM3 text-only segmentation to detect the rectangle on generated image
            rectangle_masks = self.sam3.segment_text_only(
                image=generated_image, text_prompt="bright red rectangle"
            )

            bboxes = []
            scores = []

            orig_size = pil_image.size
            gen_size = generated_image.size
            scale_x = orig_size[0] / gen_size[0] if gen_size[0] > 0 else 1.0
            scale_y = orig_size[1] / gen_size[1] if gen_size[1] > 0 else 1.0

            for mask in rectangle_masks:
                mask_bboxes = mask_to_bbox(mask)
                for bbox in mask_bboxes:
                    x1, y1, x2, y2 = bbox
                    x1_scaled = x1 * scale_x
                    y1_scaled = y1 * scale_y
                    x2_scaled = x2 * scale_x
                    y2_scaled = y2 * scale_y
                    bboxes.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled])
                    scores.append(1.0)

            # --- Logic from process_image ---
            if len(bboxes) == 0:
                results_batch.append([])
                continue

            # Segment original image using detected bboxes
            masks, probs, scores = self.sam3.segment_with_boxes(
                pil_image, bboxes, prompt, return_scores=True
            )

            oriented_bboxes = self._calculate_oriented_bboxes(bboxes, masks, probs)
            results = self._create_result_dicts(
                bboxes, masks.tolist(), scores.tolist(), oriented_bboxes
            )

            # Attach generated image to each result
            for res in results:
                res["generated_image"] = generated_image

            results_batch.append(results)

        return results_batch
