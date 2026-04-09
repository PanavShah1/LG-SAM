from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from shapely.geometry import Point, Polygon
from sklearn.cluster import DBSCAN

from models.qwen3_vl_8b import Qwen3VL8B
from models.remotesam import RemoteSAM
from models.sam3 import SAM3Segmenter
from pipelines.base import BasePipeline
from prompts import EXTRACT_OBJECT_CLASS_PROMPT
from utils.bbox import M2B
from utils.image import load_image


class CGRPipeline(BasePipeline):
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
        pixel_count_threshold: int = 50,
        dbscan_eps: float = 20.0,
        dbscan_min_samples: int = 5,
        **kwargs,
    ) -> List[List[dict]]:
        """
        Process a batch of images using batched RemoteSAM detection and SAM3 segmentation.

        Args:
            images: List of image inputs (file path, PIL Image, or numpy array)
            text_prompts: List of text descriptions of objects to detect
            pixel_threshold: Threshold for pixel probability
            dbscan_eps: Epsilon for DBSCAN
            dbscan_min_samples: Minimum samples for DBSCAN

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
            ]
        )

        results = []
        rs_masks, rs_probs = self.detector.remote_sam.referring_seg_batch(
            pil_images, text_prompts, return_prob=True
        )

        for idx in range(len(pil_images)):
            image_results = []
            rs_prob = rs_probs[idx]

            points = []
            pixel_threshold = 0.99
            while len(points) < pixel_count_threshold:
                points = np.argwhere(rs_prob >= pixel_threshold)
                # np.argwhere returns [y, x] but we need [x, y]
                points = points[:, ::-1]
                pixel_threshold *= 0.9

            clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(
                points
            )
            clusters_by_label = {}
            for label, point in zip(clusterer.labels_, points):
                if label == -1:
                    continue
                if label not in clusters_by_label:
                    clusters_by_label[label] = []
                clusters_by_label[label].append(point)

            for cluster_idx in range(len(clusters_by_label)):
                cluster_points = clusters_by_label[cluster_idx]
                sam3_masks, sam3_probs = self.sam3.segment_with_points(
                    image=pil_images[idx],
                    points=np.array(cluster_points),
                    text_prompt=object_classes[idx],
                )
                if len(sam3_masks) == 0:
                    print(f"No masks found for cluster {cluster_idx} in image {idx}")
                    continue

                mask = np.max(sam3_masks, axis=0).astype(np.uint8)
                prob = np.max(sam3_probs, axis=0)
                bboxes = M2B(mask, prob, box_type="obb")

                if len(bboxes) == 0:
                    print(f"No boxes found for cluster {cluster_idx} in image {idx}")
                    continue

                max_cnt = 0
                best_bbox_idx = 0
                best_bbox_score = 0
                for bbox_idx, bbox in enumerate(bboxes):
                    cnt = 0
                    x1, y1, x2, y2, x3, y3, x4, y4, conf = bbox
                    poly = Polygon([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                    for point in cluster_points:
                        if poly.contains(Point(point[0], point[1])):
                            cnt += 1
                    if cnt > max_cnt:
                        max_cnt = cnt
                        best_bbox_idx = bbox_idx
                        best_bbox_score = conf

                image_results.append(
                    {
                        "mask": mask,
                        "oriented_bbox": bboxes[best_bbox_idx][:8],
                        "score": best_bbox_score,
                    }
                )

            results.append(image_results)

        return results
