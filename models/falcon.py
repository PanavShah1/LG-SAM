import re
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

import config


class CoordinatesQuantizer:
    """Quantize and dequantize coordinates between pixel space and token space."""

    def __init__(self, mode: str = "floor", bins: tuple = (1000, 1000)):
        self.mode = mode
        self.bins = bins

    def dequantize(
        self, coordinates: torch.Tensor, size: Tuple[int, int]
    ) -> torch.Tensor:
        """Dequantize coordinates from bin space to pixel space."""
        bins_w, bins_h = self.bins
        size_w, size_h = size
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h

        x, y = coordinates.split(1, dim=-1)
        dequantized_x = (x + 0.5) * size_per_bin_w
        dequantized_y = (y + 0.5) * size_per_bin_h

        return torch.cat((dequantized_x, dequantized_y), dim=-1)


class Falcon:
    """Wrapper for Falcon model to handle visual grounding tasks."""

    def __init__(self, device: Optional[torch.device | str] = None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.quantizer = CoordinatesQuantizer()
        self._loaded = False

    def load(self):
        """Load the Falcon model and processor."""
        if self._loaded:
            return

        print(f"Loading Falcon model from {config.FALCON_CHECKPOINT}...")
        with torch.cuda.device(self.device):
            self.model = AutoModelForCausalLM.from_pretrained(
                config.FALCON_CHECKPOINT,
                trust_remote_code=True,
                device_map=self.device,
            )

            self.processor = AutoProcessor.from_pretrained(
                config.FALCON_CHECKPOINT,
                trust_remote_code=True,
                device_map=self.device,
            )
        self._loaded = True
        print(f"Falcon model from {config.FALCON_CHECKPOINT} loaded on {self.device}")

    def extract_hbb_bboxes(self, text: str) -> List:
        """
        Extract horizontal bounding boxes (4 points) from generated text.
        Only matches exactly 4 consecutive coordinate tokens, not 8.
        """
        # First remove any 8-point sequences to avoid partial matches
        text_cleaned = re.sub(r"<\d+><\d+><\d+><\d+><\d+><\d+><\d+><\d+>", "", text)
        pattern = r"<(\d+)><(\d+)><(\d+)><(\d+)>"
        matches = re.findall(pattern, text_cleaned)
        return [list(map(int, match)) for match in matches]

    def extract_obb_bboxes(self, text: str) -> List:
        """
        Extract oriented bounding boxes (8 points) from generated text.
        Matches exactly 8 consecutive coordinate tokens.
        """
        pattern = r"<(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)><(\d+)>"
        matches = re.findall(pattern, text)
        return [list(map(int, match)) for match in matches]

    def hbb_to_corners(self, bbox: List, image_size: Tuple[int, int]) -> List:
        """
        Convert HBB [x1, y1, x2, y2] to 8-point corner format.
        Returns: [x1, y1, x2, y1, x2, y2, x1, y2] (clockwise from top-left)
        """
        x1, y1, x2, y2 = bbox
        # Dequantize the coordinates
        coords = np.array([[x1, y1], [x2, y2]])
        dequantized = self.quantizer.dequantize(
            torch.tensor(coords), size=image_size
        ).numpy()

        dx1, dy1 = dequantized[0]
        dx2, dy2 = dequantized[1]

        # Convert to 8-point format (clockwise from top-left) as integers
        return [
            int(round(dx1)),
            int(round(dy1)),  # top-left
            int(round(dx2)),
            int(round(dy1)),  # top-right
            int(round(dx2)),
            int(round(dy2)),  # bottom-right
            int(round(dx1)),
            int(round(dy2)),  # bottom-left
        ]

    def obb_to_corners(self, bbox: List, image_size: Tuple[int, int]) -> List:
        """
        Convert OBB 8-point quantized to pixel coordinates.
        """
        coords = np.array(bbox).reshape(4, 2)
        dequantized = (
            self.quantizer.dequantize(torch.tensor(coords), size=image_size)
            .reshape(-1)
            .tolist()
        )
        return [int(round(x)) for x in dequantized]

    def clean_output(self, text: str) -> str:
        """Clean special tokens from output."""
        return (
            text.replace("</s>", "")
            .replace("<s>", "")
            .replace("<pad>", "")
            .replace("</pad>", "")
            .strip()
        )

    def detect_batch(
        self,
        images: List[Image.Image],
        text_prompts: List[str],
    ) -> List[List[List[float]]]:
        """
        Detect objects in a batch of images using text prompts.

        Args:
            images: List of PIL Images or numpy arrays
            text_prompts: List of text descriptions of objects to detect

        Returns:
            List of lists, each containing [x1, y1, x2, y2, x3, y3, x4, y4] boxes
        """

        text_prompts = [
            f"Detect {text_prompt}\nUse oriented bounding boxes."
            for text_prompt in text_prompts
        ]
        with torch.cuda.device(self.device):
            inputs = self.processor(
                text=text_prompts,
                images=images,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                )

        raw_outputs = self.processor.batch_decode(outputs, skip_special_tokens=False)

        bounding_boxes_list = []

        for image, raw_output in zip(images, raw_outputs):
            image_size = (image.width, image.height)
            bounding_boxes = []

            obb_boxes = self.extract_obb_bboxes(raw_output)
            if obb_boxes:
                for i, bbox in enumerate(obb_boxes):
                    corners = self.obb_to_corners(bbox, image_size)
                    bounding_boxes.append(corners)
            else:
                # Fall back to HBB (4-point) boxes and convert to 8-point
                hbb_boxes = self.extract_hbb_bboxes(raw_output)
                if hbb_boxes:
                    for i, bbox in enumerate(hbb_boxes):
                        corners = self.hbb_to_corners(bbox, image_size)
                        bounding_boxes.append(corners)
                else:
                    print("Warning: no bounding boxes found in output")
                    bounding_boxes.append(
                        [
                            0,
                            0,
                            image.width,
                            0,
                            image.width,
                            image.height,
                            0,
                            image.height,
                        ]
                    )

            bounding_boxes_list.append(bounding_boxes)

        return bounding_boxes_list
