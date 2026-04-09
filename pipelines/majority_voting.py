import concurrent.futures
from typing import List, Optional, Tuple, Union
import concurrent.futures
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
import math

from pipelines.base import BasePipeline
from utils.bbox import greedy_obb_matching, hbb_to_corners
from utils.image import load_image


class MajorityVotingPipeline(BasePipeline):
    """Class to perform majority voting on multiple model predictions."""

    def __init__(
        self, pipelines: List[BasePipeline], device: Optional[torch.device | str] = None
    ):
        """
        Initialize majority voting pipeline.

        Args:
            pipelines: List of pipelines to vote on
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        super().__init__(device=device)
        self.pipelines = pipelines

    def _initialize_models(self):
        """
        Initialize models.
        """
        pass  # Models are initialized externally

    def load_models(self):
        """
        Load models.
        """
        if not self._models_loaded:
            for pipeline in self.pipelines:
                pipeline.load_models()
            self._models_loaded = True

    def _vote(
        self, results_list: List[List[dict]], image_size: Tuple[int, int]
    ) -> List[dict]:
        """
        Perform majority voting on the results from multiple pipelines.

        Args:
            results_list: List of results (detections) from each pipeline.
                          Each element is a list of detection dictionaries for the same image.

        Returns:
            The result list from the pipeline that has the highest average IoU with others.
        """
        backup_bbox = hbb_to_corners([0, 0, image_size[0], image_size[1]])

        num_pipelines = len(results_list)
        miou_matrix = np.zeros((num_pipelines, num_pipelines))
        for i in range(num_pipelines):
            for j in range(i + 1, num_pipelines):
                obbs_i = [
                    res["oriented_bbox"]
                    for res in results_list[i]
                    if res["oriented_bbox"] != backup_bbox
                ]
                obbs_j = [
                    res["oriented_bbox"]
                    for res in results_list[j]
                    if res["oriented_bbox"] != backup_bbox
                ]
                matches = greedy_obb_matching(obbs_i, obbs_j)
                if len(matches) == 0:
                    miou = 0.0
                    miou_matrix[i, j] = miou
                    miou_matrix[j, i] = miou
                    continue
                miou = sum(match[2] for match in matches) / len(matches)
                count_diff = abs(len(obbs_i) - len(obbs_j))
                score = miou * math.exp(-0.5*count_diff)
                miou_matrix[i, j] = score
                miou_matrix[j, i] = score
        for i in range(num_pipelines):
            miou_matrix[i, i] = 1.0

        best_pipeline_idx = np.argmax(np.sum(miou_matrix, axis=1))
        return results_list[best_pipeline_idx]

    def process_image(
        self,
        image: Union[str | Image.Image, np.ndarray],
        text_prompt: str,
        **kwargs,
    ) -> List[dict]:
        """
        Process a single image and text prompt to detect objects using multiple models and perform majority voting.

        Args:
            image: PIL Image, numpy array, or path
            text_prompt: Text description for the image
        Returns:
            List of dictionaries containing 'oriented_bbox', 'mask' and 'score' for the image
        """
        if not self._models_loaded:
            self.load_models()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(pipeline.process_image, image, text_prompt)
                for pipeline in self.pipelines
            ]
            results_list = [f.result() for f in futures]

        return self._vote(results_list, image.size)

    def process_batch(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        text_prompts: List[str],
        **kwargs,
    ) -> List[List[dict]]:
        """
        Process a batch of images using multiple pipelines and perform majority voting.

        Args:
            images: List of image inputs
            text_prompts: List of text descriptions
        Returns:
            List of lists of dictionaries, where each inner list contains results for one image.
        """
        if not self._models_loaded:
            self.load_models()

        if len(images) != len(text_prompts):
            raise ValueError("Number of images and text prompts must match.")

        # Run process_batch for each pipeline in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(pipeline.process_batch, images, text_prompts, **kwargs)
                for pipeline in self.pipelines
            ]
            # all_pipelines_results is a list of (list of list of dicts)
            # Outer list: results for each pipeline (size = num_pipelines)
            # Inner list: results for each image in the batch (size = num_images)
            all_pipelines_results = [f.result() for f in futures]

        num_images = len(images)
        final_results = []

        for i in range(num_images):
            # Gather results for image i from all pipelines
            image_i_results = [
                pipeline_res[i] for pipeline_res in all_pipelines_results
            ]
            final_results.append(self._vote(image_i_results, images[i].size))  # type: ignore

        return final_results
