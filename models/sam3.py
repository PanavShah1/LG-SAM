"""SAM 3 segmenter using sam3 package."""

from copy import deepcopy
from typing import Dict, List, Literal, Optional, Tuple, Union, overload

import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

import config


class SAM3Segmenter:
    """Wrapper for SAM 3 segmentation model."""

    def __init__(self, device: Optional[torch.device | str] = None):
        """
        Initialize SAM 3 segmenter.

        Args:
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._loaded = False

    def load(self):
        """Load the SAM 3 model and processor."""
        if self._loaded:
            return

        print("Loading SAM 3 model...")
        with torch.cuda.device(self.device):
            self.model = build_sam3_image_model(checkpoint_path=config.SAM3_CHECKPOINT)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.processor = Sam3Processor(self.model)
        self._loaded = True
        print(f"SAM 3 model loaded on {self.device}")

    @torch.inference_mode()
    def _add_points_prompt(
        self,
        points: Union[List[Tuple[float, float]], np.ndarray],
        labels: List,
        state: Dict,
    ):
        """Adds point prompts and runs the inference.
        The image needs to be set, but not necessarily the text prompt.
        Points are assumed to be in (x, y) pixel coordinates.
        Labels are True/1 for positive points, False/0 for negative points.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before adding point prompts")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"],
                device=self.device,  # type: ignore
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # Normalize points to [0, 1] range
        img_h = state["original_height"]
        img_w = state["original_width"]

        # Convert points list to tensor and normalize
        points_tensor = torch.tensor(points, device=self.device, dtype=torch.float32)
        points_tensor[:, 0] = points_tensor[:, 0] / img_w  # normalize x
        points_tensor[:, 1] = points_tensor[:, 1] / img_h  # normalize y

        # Add sequence and batch dimensions: [num_points, batch_size=1, 2]
        points_tensor = points_tensor.unsqueeze(1)

        # Convert labels to tensor with correct shape: [num_points, batch_size=1]
        labels_tensor = torch.tensor(labels, device=self.device, dtype=torch.long).view(
            -1, 1
        )

        state["geometric_prompt"].append_points(points_tensor, labels_tensor)

        return self.processor._forward_grounding(state)

    @overload
    def _process_outputs(
        self, output: Dict[str, torch.Tensor], return_scores: Literal[False] = False
    ) -> Tuple[np.ndarray, np.ndarray]: ...
    @overload
    def _process_outputs(
        self, output: Dict[str, torch.Tensor], return_scores: Literal[True]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def _process_outputs(
        self, output: Dict[str, torch.Tensor], return_scores: bool = False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        masks = []
        probs = []
        scores = []
        for i in range(len(output["masks"])):
            masks.append(output["masks"][i].squeeze(0).cpu().numpy())
            probs.append(output["masks_logits"][i].squeeze(0).cpu().numpy())
            scores.append(output["scores"][i].squeeze(0).cpu().numpy())
        if return_scores:
            return np.array(masks), np.array(probs), np.array(scores)
        return np.array(masks), np.array(probs)

    @overload
    def segment_with_boxes(
        self,
        image: Union[Image.Image, np.ndarray],
        bboxes: List[List[float]],
        text_prompt: str,
        return_scores: Literal[False] = False,
    ) -> Tuple[np.ndarray, np.ndarray]: ...
    @overload
    def segment_with_boxes(
        self,
        image: Union[Image.Image, np.ndarray],
        bboxes: List[List[float]],
        text_prompt: str,
        return_scores: Literal[True],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def segment_with_boxes(
        self,
        image: Union[Image.Image, np.ndarray],
        bboxes: List[List[float]],
        text_prompt: str,
        return_scores: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Generate segmentation masks from bounding boxes.

        Args:
            image: PIL Image or numpy array
            bboxes: List of bounding boxes, each as [x1, y1, x2, y2] in pixel coordinates
            return_scores: When True, also return per-pixel scores

        Returns:
            Tuple of (masks, probs). When return_scores is True, returns
            a tuple of (masks, probs, scores).
        """
        if not self._loaded:
            self.load()

        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        with torch.cuda.device(self.device):
            inference_state = self.processor.set_image(image)
            inference_state = self.processor.set_text_prompt(
                state=inference_state, prompt=text_prompt
            )

            masks = []
            probs = []
            scores = []

            # Process each bounding box
            for bbox in bboxes:
                # bbox format: [x1, y1, x2, y2]
                # Convert to format expected by SAM3: [cx, cy, w, h]
                box = [
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2,
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]
                box = [
                    box[0] / image.size[0],
                    box[1] / image.size[1],
                    box[2] / image.size[0],
                    box[3] / image.size[1],
                ]

                # Set box prompt
                output = self.processor.add_geometric_prompt(
                    state=deepcopy(inference_state), box=box, label=True
                )

                # Extract masks from output
                for i in range(len(output["masks"])):
                    masks.append(output["masks"][i].squeeze(0).cpu().numpy())
                    probs.append(output["masks_logits"][i].squeeze(0).cpu().numpy())
                    scores.append(output["scores"][i].squeeze(0).cpu().numpy())

            if return_scores:
                return np.array(masks), np.array(probs), np.array(scores)

            return np.array(masks), np.array(probs)

    @overload
    def segment_text_only(
        self,
        image: Union[Image.Image, np.ndarray],
        text_prompt: str,
        return_scores: Literal[False] = False,
    ) -> Tuple[np.ndarray, np.ndarray]: ...
    @overload
    def segment_text_only(
        self,
        image: Union[Image.Image, np.ndarray],
        text_prompt: str,
        return_scores: Literal[True],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def segment_text_only(
        self,
        image: Union[Image.Image, np.ndarray],
        text_prompt: str,
        return_scores: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Generate segmentation masks from text prompt only (without bounding boxes).

        Args:
            image: PIL Image or numpy array
            text_prompt: Text description of objects to segment
            return_confidence: When True, also return per-pixel confidence maps

        Returns:
            List of binary segmentation masks. When return_confidence is True, returns
            a tuple of (masks, confidence_maps).
        """
        if not self._loaded:
            self.load()

        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        with torch.cuda.device(self.device):
            inference_state = self.processor.set_image(image)
            inference_state = self.processor.set_text_prompt(
                state=inference_state, prompt=text_prompt
            )

            return self._process_outputs(inference_state, return_scores=return_scores)

    @overload
    def segment_with_points(
        self,
        image: Union[Image.Image, np.ndarray],
        points: Union[List[Tuple[float, float]], np.ndarray],
        text_prompt: str,
        return_scores: Literal[False] = False,
    ) -> Tuple[np.ndarray, np.ndarray]: ...
    @overload
    def segment_with_points(
        self,
        image: Union[Image.Image, np.ndarray],
        points: Union[List[Tuple[float, float]], np.ndarray],
        text_prompt: str,
        return_scores: Literal[True],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    def segment_with_points(
        self,
        image: Union[Image.Image, np.ndarray],
        points: Union[List[Tuple[float, float]], np.ndarray],
        text_prompt: str,
        return_scores: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """
        Generate segmentation masks from points.

        Args:
            image: PIL Image or numpy array
            points: List of points, each as (x, y) in pixel coordinates
            return_scores: When True, also return per-pixel scores

        Returns:
            Tuple of (masks, probs). When return_scores is True, returns
            a tuple of (masks, probs, scores).
        """
        if not self._loaded:
            self.load()

        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        with torch.cuda.device(self.device):
            inference_state = self.processor.set_image(image)
            inference_state = self.processor.set_text_prompt(
                state=inference_state, prompt=text_prompt
            )
            inference_state = self._add_points_prompt(
                points=points, labels=[1] * len(points), state=inference_state
            )

            # Process outputs
            return self._process_outputs(inference_state, return_scores=return_scores)
