from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from models.flux2 import Flux2Generator
from models.qwen3_vl import Qwen3VL
from models.sam3 import SAM3Segmenter
from pipelines.base import BasePipeline
from utils.bbox import mask_to_bbox
from utils.image import load_image


class Flux2SAM3Pipeline(BasePipeline):
    """
    Pipeline that uses FLUX.2-dev to draw rectangles around objects,
    then SAM3 to detect and segment them.
    """

    def __init__(
        self,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Initialize the FLUX2-SAM3 detection pipeline.

        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        super().__init__(device=device)

    def _initialize_models(self):
        """Initialize FLUX.2-dev generator, Qwen3-VL, and SAM 3 segmenter."""
        self.flux2 = Flux2Generator(device=self.device)
        self.sam3 = SAM3Segmenter(device=self.device)
        self.qwen3vl = Qwen3VL(device=self.device)

    def load_models(self):
        """Load all models."""
        if not self._models_loaded:
            if hasattr(self, "flux2") and hasattr(self.flux2, "load"):
                self.flux2.load()
            if hasattr(self, "sam3") and hasattr(self.sam3, "load"):
                self.sam3.load()
            if hasattr(self, "qwen3vl") and hasattr(self.qwen3vl, "load"):
                self.qwen3vl.load()
            self._models_loaded = True

    def _detect_objects(
        self,
        image: Image.Image,
        text_prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        generator_seed: Optional[int] = None,
        rectangle_prompt: str = "bright red rectangle",
        rewrite_prompt: bool = True,
        rewrite_instruction: Optional[str] = None,
        **kwargs,
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Detect objects using FLUX2 and SAM3.

        Step 0: Rewrite the prompt using Qwen3-VL (optional).
        Step 1: Use FLUX2 to generate an image with a rectangle drawn around the object.
        Step 2: Use SAM3 text-only segmentation to detect the rectangle in the generated image.

        Args:
            image: PIL Image in RGB format
            text_prompt: Text description of objects to detect
            num_inference_steps: Number of inference steps for FLUX2 (default: 50)
            guidance_scale: Guidance scale for FLUX2 generation (default: 4.0)
            generator_seed: Optional random seed for FLUX2 generation
            rectangle_prompt: Text prompt to use for detecting the rectangle in SAM3
                             (default: "bright red rectangle")
            rewrite_prompt: Whether to rewrite the prompt using Qwen3-VL (default: True)
            rewrite_instruction: Instruction for prompt rewriting
            **kwargs: Additional arguments (not used)

        Returns:
            Tuple of (bboxes, scores) where:
            - bboxes: List of bounding boxes, each as [x1, y1, x2, y2]
            - scores: List of confidence scores (set to 1.0 for detected rectangles)
        """
        # Step 0: Rewrite prompt using Qwen3-VL
        if rewrite_prompt:
            if rewrite_instruction is None:
                rewrite_instruction = "Describe the {prompt} visible in this image."

            if "{prompt}" in rewrite_instruction:
                final_prompt = rewrite_instruction.format(prompt=text_prompt)
            else:
                final_prompt = f"{rewrite_instruction}\nOriginal Prompt: {text_prompt}"

            text_prompt = self.qwen3vl.generate(image, final_prompt)

        # Step 1: Generate image with FLUX2 that has a rectangle drawn around the object
        generated_image = self.flux2.generate(
            image=image,
            prompt=text_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator_seed=generator_seed,
        )

        # Step 2: Use SAM3 text-only segmentation to detect the rectangle
        # The rectangle should be visible in the generated image
        rectangle_masks, _ = self.sam3.segment_text_only(
            image=generated_image, text_prompt=rectangle_prompt
        )

        # Extract bounding boxes from the masks
        bboxes = []
        scores = []

        # Check if image sizes match (FLUX2 should maintain size, but verify)
        orig_size = image.size  # (width, height)
        gen_size = generated_image.size  # (width, height)
        scale_x = orig_size[0] / gen_size[0] if gen_size[0] > 0 else 1.0
        scale_y = orig_size[1] / gen_size[1] if gen_size[1] > 0 else 1.0

        for mask in rectangle_masks:
            # Convert mask to axis-aligned bounding boxes
            mask_bboxes = mask_to_bbox(mask)

            # Add each detected bounding box, scaling if necessary
            for bbox in mask_bboxes:
                x1, y1, x2, y2 = bbox
                # Scale bounding box coordinates to match original image size
                x1_scaled = x1 * scale_x
                y1_scaled = y1 * scale_y
                x2_scaled = x2 * scale_x
                y2_scaled = y2 * scale_y
                bboxes.append([x1_scaled, y1_scaled, x2_scaled, y2_scaled])
                # Use a default score of 1.0 for detected rectangles
                # (since we're detecting a visual element, not a confidence score)
                scores.append(1.0)

        # If no rectangles found, return empty lists
        if len(bboxes) == 0:
            return [], []

        return bboxes, scores

    def process_batch(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        text_prompts: List[str],
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        generator_seed: Optional[int] = None,
        rectangle_prompt: str = "bright red rectangle",
        rewrite_prompt: bool = True,
        rewrite_instruction: Optional[str] = None,
        **kwargs,
    ) -> List[List[Dict]]:
        """
        Process a batch of images using FLUX2 and SAM3 with optimized batch generation.

        Args:
            images: List of image inputs
            text_prompts: List of text descriptions
            num_inference_steps: Steps for FLUX2
            guidance_scale: Guidance scale for FLUX2
            generator_seed: Seed for FLUX2
            rectangle_prompt: Prompt for detecting the rectangle
            rewrite_prompt: Whether to rewrite the prompt using Qwen3-VL
            rewrite_instruction: Instruction for prompt rewriting
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
            if rewrite_instruction is None:
                rewrite_instruction = "Describe the {prompt} visible in this image."

            # Construct full prompts for rewriting
            rewriting_prompts = []
            for p in text_prompts:
                if "{prompt}" in rewrite_instruction:
                    rewriting_prompts.append(rewrite_instruction.format(prompt=p))
                else:
                    rewriting_prompts.append(
                        f"{rewrite_instruction}\nOriginal Prompt: {p}"
                    )

            text_prompts = self.qwen3vl.generate(pil_images, rewriting_prompts)
            if isinstance(text_prompts, str):
                text_prompts = [text_prompts]

        # Step 1: Batch generate images with FLUX2 (this is the speedup)
        generated_images = self.flux2.generate(
            image=pil_images,
            prompt=text_prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator_seed=generator_seed,
        )

        # If only one image was generated (edge case handling in generate), wrap it
        if not isinstance(generated_images, list):
            generated_images = [generated_images]

        results_batch = []

        # Step 2: Process each image sequentially for SAM3 parts
        for i, (pil_image, generated_image, prompt) in enumerate(
            zip(pil_images, generated_images, text_prompts)
        ):
            # --- Logic from _detect_objects ---
            # Use SAM3 text-only segmentation to detect the rectangle on generated image
            rectangle_masks, _ = self.sam3.segment_text_only(
                image=generated_image, text_prompt=rectangle_prompt
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
            segment_output = self.sam3.segment_with_boxes(
                pil_image, bboxes, prompt, return_confidence=True
            )

            if isinstance(segment_output, tuple):
                masks, confidence_maps = segment_output
            else:
                masks = segment_output
                confidence_maps = None

            oriented_bboxes = self._calculate_oriented_bboxes(
                bboxes, masks, confidence_maps
            )
            results = self._create_result_dicts(
                bboxes, masks, scores, oriented_bboxes, confidence_maps
            )
            results_batch.append(results)

        return results_batch
