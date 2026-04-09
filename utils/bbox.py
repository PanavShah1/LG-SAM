"""Bounding box utilities for oriented bounding box calculation."""

from typing import Annotated, List, Optional, Tuple, Union

import cv2
import numpy as np
from shapely.geometry import Polygon


def mask_to_bbox(
    mask: np.ndarray, threshold: float = 0.5
) -> List[Tuple[float, float, float, float]]:
    """
    Convert a binary segmentation mask to axis-aligned bounding boxes.

    Args:
        mask: Binary segmentation mask as numpy array (H, W) with values 0 or 1/255
        threshold: Threshold value (0-1) used to decide which pixels belong to a bounding box

    Returns:
        List of bounding boxes, each as (x1, y1, x2, y2) in pixel coordinates
        Returns empty list if no contours found
    """
    # Ensure mask is uint8 and binary (0 or 255)
    if mask.dtype != np.uint8:
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)

    # Ensure binary mask (0 or 255) for cv2.findContours
    mask = (mask > 255 * threshold).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []

    for contour in contours:
        # Skip very small contours
        if len(contour) < 3:
            continue

        # Get axis-aligned bounding box
        x, y, w, h = cv2.boundingRect(contour)
        x2 = x + w
        y2 = y + h

        bboxes.append((float(x), float(y), float(x2), float(y2)))

    return bboxes


def mask_to_oriented_bbox(
    mask: np.ndarray,
    confidence_map: Optional[np.ndarray] = None,
    conf_threshold: float = 0.5,
    min_component_area: int = 10,
    connectivity: int = 8,
) -> List[Tuple[float, float, float, float, float]]:
    """
    Convert a binary segmentation mask to oriented bounding boxes.

    Args:
        mask: Binary segmentation mask as numpy array (H, W)
        confidence_map: Optional confidence map aligned with the mask. When provided,
            clustering operates on this map instead of the raw mask values.
        conf_threshold: Confidence threshold (0-1) used to decide which pixels belong
            to a cluster.
        min_component_area: Minimum pixel area for a connected component to be
            considered valid.
        connectivity: Pixel connectivity (4 or 8) passed to OpenCV when clustering.

    Returns:
        List of oriented bounding boxes, each as (center_x, center_y, width, height, angle_degrees)
        Returns empty list if no contours found
    """
    if mask is None or mask.size == 0:
        return []

    confidence_source = confidence_map if confidence_map is not None else mask

    # Normalize confidence source to [0, 1]
    confidence = confidence_source.astype(np.float32)
    max_val = confidence.max()
    if max_val > 0:
        if max_val > 1.0:
            confidence = confidence / 255.0
    else:
        return []

    confidence = np.clip(confidence, 0.0, 1.0)

    # Threshold to keep only the confident pixels and convert to uint8 mask
    clustered_mask = (confidence >= conf_threshold).astype(np.uint8)
    if clustered_mask.max() == 0:
        return []

    num_components, labels, stats, _ = cv2.connectedComponentsWithStats(
        clustered_mask, connectivity=connectivity
    )

    oriented_bboxes = []

    for component_idx in range(1, num_components):
        area = stats[component_idx, cv2.CC_STAT_AREA]
        if area < min_component_area:
            continue

        component_mask = np.where(labels == component_idx, 255, 0).astype(np.uint8)

        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            continue

        rect = cv2.minAreaRect(contour)
        center, (w, h), angle = rect
        oriented_bboxes.append((center[0], center[1], w, h, angle))

    return oriented_bboxes


def corners_to_obb5(pts: list | np.ndarray) -> List[float]:
    """
    pts: 1x8 array of corner points in order
         (either clockwise or counter-clockwise)
    returns: [cx, cy, w, h, angle] (in degrees)
    """

    pts = np.asarray(pts, dtype=float).reshape(4, 2)

    # 1) center
    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()

    # 2) pick one edge direction (p1 → p2)
    edge1 = pts[1] - pts[0]
    edge2 = pts[2] - pts[1]

    # lengths
    w = np.linalg.norm(edge1)
    h = np.linalg.norm(edge2)

    # 3) angle of the first edge
    theta = np.arctan2(edge1[1], edge1[0])

    # -------------------------------
    # Normalize θ to be in [-π/2, 0)
    # -------------------------------
    # 1) Wrap angle to [-π, π)
    theta = (theta + np.pi) % (2 * np.pi) - np.pi

    # 2) Convert representation so that width is the long edge
    #    (optional depending on your convention)
    if theta < -np.pi / 2:
        # Rotate by +90° and swap w/h
        theta += np.pi / 2
        w, h = h, w
    elif theta >= 0:
        # Rotate by -90° and swap w/h
        theta -= np.pi / 2
        w, h = h, w

    return [cx, cy, w, h, np.rad2deg(theta)]  # type: ignore


def obb5_to_corners(cx, cy, w, h, theta):
    """
    Convert oriented bounding box (cx, cy, w, h, theta) to its 4 corner points.
    theta is in radians and expected in [-pi/2, 0)

    Returns: 4x2 array of corner points in CCW order
    """

    # half sizes
    dw = w / 2.0
    dh = h / 2.0

    # local corners before rotation (centered at 0,0)
    # CCW order: bottom-left, bottom-right, top-right, top-left
    local = np.array([[-dw, -dh], [dw, -dh], [dw, dh], [-dw, dh]])

    # rotation matrix
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    # rotate + translate
    pts = (R @ local.T).T
    pts[:, 0] += cx
    pts[:, 1] += cy

    return pts


def hbb_to_corners(hbb: List[float]) -> List[float]:
    """
    Convert horizontal bounding box to its 4 corner points.
    """
    x1, y1, x2, y2 = hbb
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def M2B_woEPOC(mask, probs, conf_threshold, box_type):
    boxes = []
    num_connect, label_mtrix, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    for i in range(1, num_connect):
        # confidence
        box_mask = np.where(label_mtrix == i, 1, 0).astype(np.uint8)
        conf = np.sum(probs * box_mask) / stats[i][-1]
        if conf < conf_threshold:
            continue
        # bnbox
        if box_type == "hbb":
            x, y, w, h, _ = stats[i]
            boxes.append([x, y, x + w, y + h, conf])
        elif box_type == "obb":
            countour, _ = cv2.findContours(
                box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            rec = cv2.minAreaRect(countour[0])
            box = cv2.boxPoints(rec).astype(int)
            boxes.append(
                [
                    box[0][0],
                    box[0][1],
                    box[1][0],
                    box[1][1],
                    box[2][0],
                    box[2][1],
                    box[3][0],
                    box[3][1],
                    conf,
                ]
            )

    return boxes


def box1_in_box2(box1, box2):
    # cv2 implementation, change to shapely recommended
    assert len(box1) == 5 or len(box1) == 9, (
        f"len(box) should be 5 or 9, but got {len(box1)}"
    )
    assert len(box2) == 5 or len(box2) == 9, (
        f"len(box) should be 5 or 9, but got {len(box2)}"
    )

    if len(box1) == 5:
        # expand to 4 points
        x1, y1, x2, y2 = box1[0], box1[1], box1[2], box1[3]
        box1 = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        x1, y1, x2, y2 = box2[0], box2[1], box2[2], box2[3]
        box2 = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    else:
        box1 = np.array(box1[:-1]).reshape(-1, 2)
        box2 = np.array(box2[:-1]).reshape(-1, 2)

    w = max(box1[:, 0].max(), box2[:, 0].max())
    h = max(box1[:, 1].max(), box2[:, 1].max())

    canvas1 = np.zeros((h, w), dtype=np.uint8)
    canvas2 = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(canvas1, [box1], 1)
    cv2.fillPoly(canvas2, [box2], 1)

    return np.logical_and(canvas1, canvas2).sum() == canvas1.sum()


def M2B(
    mask,
    probs,
    *,
    box_type,
    conf_threshold=0.01,
    remove_inner=True,
) -> List[Annotated[List[float], 9]]:
    """
    Convert mask to bounding box.
    Args:
        :param mask: original mask from model
        :param probs: probability map from model
        :param box_type: 'hbb' or 'obb'
        :param conf_threshold: confidence threshold
        :param remove_inner: whether to remove inner boxes
    Returns:
        :return: boxes: list of bounding boxes
    """

    assert box_type in ["hbb", "obb"], (
        f"box_type should be 'hbb' or 'obb', but got {box_type}"
    )

    boxes = M2B_woEPOC(mask, probs, conf_threshold, box_type)

    # remove small box contained in large box
    if remove_inner:
        indice = []
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i == j:
                    continue
                if box1_in_box2(boxes[i], boxes[j]):
                    indice.append(i)
        boxes = [boxes[i] for i in range(len(boxes)) if i not in indice]
    return boxes


def M2BSingle(
    mask,
    probs,
    *,
    box_type,
    conf_threshold=0.01,
) -> Optional[Annotated[List[float], 9]]:
    """
    Convert mask to bounding box.
    Args:
        :param mask: original mask from model
        :param probs: probability map from model
        :param box_type: 'hbb' or 'obb'
        :param conf_threshold: confidence threshold
        :param remove_inner: whether to remove inner boxes
    Returns:
        :return: boxes: list of bounding boxes
    """

    assert box_type in ["hbb", "obb"], (
        f"box_type should be 'hbb' or 'obb', but got {box_type}"
    )

    boxes = M2B_woEPOC(mask, probs, conf_threshold, box_type)

    if len(boxes) == 0:
        return None

    return max(boxes, key=lambda x: x[8])


def denormalize_obj_corners(
    obj_corners: List[float], image_width: int, image_height: int
) -> List[float]:
    """
    Convert obj_corners format to pixel coordinates.

    Args:
        obj_corners: List of 8 values [x1, y1, x2, y2, x3, y3, x4, y4] in normalized coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        List of 8 values [x1, y1, x2, y2, x3, y3, x4, y4] in pixel coordinates
    """
    # Extract coordinates
    coords = np.array(obj_corners).reshape(4, 2).astype(float)

    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    # Most common case: normalized [0, 1] coordinates
    if (
        np.all(x_coords >= -0.2)
        and np.all(x_coords <= 1.2)
        and np.all(y_coords >= -0.2)
        and np.all(y_coords <= 1.2)
    ):
        pixel_coords = coords.copy()
        pixel_coords[:, 0] *= image_width
        pixel_coords[:, 1] *= image_height
    else:
        # Fallback: assume already in pixel coordinates or use clipping
        pixel_coords = coords.copy()
        pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, image_width)
        pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, image_height)

    # Ensure coordinates are within image bounds
    pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, image_width)
    pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, image_height)

    return pixel_coords.flatten().tolist()


def obj_corners_iou(
    obj_corners1: Union[List[float], np.ndarray],
    obj_corners2: Union[List[float], np.ndarray],
) -> float:
    """
    Calculate IoU between two object corners.

    Args:
        obj_corners1: First object corners
        obj_corners2: Second object corners

    Returns:
        IoU value between 0 and 1
    """
    if isinstance(obj_corners1, list):
        obj_corners1 = np.array(obj_corners1)
    if isinstance(obj_corners2, list):
        obj_corners2 = np.array(obj_corners2)

    poly1 = Polygon(obj_corners1.reshape(4, 2))
    poly2 = Polygon(obj_corners2.reshape(4, 2))

    if not poly1.is_valid or not poly2.is_valid:
        # Try to fix invalid polygons
        poly1 = poly1.buffer(0)
        poly2 = poly2.buffer(0)

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    try:
        intersection = poly1.intersection(poly2)
        union = poly1.union(poly2)

        if union.area == 0:
            return 0.0

        iou = intersection.area / union.area
        return max(0.0, min(1.0, iou))
    except Exception:
        return 0.0


def compute_area_rectangle(
    corners: Union[List[float], np.ndarray]
) -> float:
    """
    Compute area of oriented bounding box given its corner points. 
    
    Args:
        corners: List or array of 8 values [x1, y1, x2, y2, x3, y3, x4, y4]
    
    Returns:
        Area of the oriented bounding box
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = corners
    return 1/2 * abs(x1*y2 + x2*y3 + x3*y4 + x4*y1 - y1*x2 - y2*x3 - y3*x4 - y4*x1)


def greedy_obb_matching(
    obj_corners_list_1: List[List[float]], obj_corners_list_2: List[List[float]]
) -> List[Tuple[int, int, float]]:
    """
    Match predicted object corners to ground truth object corners using greedy IoU matching.

    Args:
        obj_corners_list_1: List of predicted object corners
        obj_corners_list_2: List of ground truth object corners

    Returns:
        List of tuples (obj_corners_1_idx, obj_corners_2_idx, iou) for matched pairs
    """
    if len(obj_corners_list_1) == 0 or len(obj_corners_list_2) == 0:
        return []

    # Calculate IoU matrix
    iou_matrix = np.zeros((len(obj_corners_list_1), len(obj_corners_list_2)))
    for i, obj_corners_1 in enumerate(obj_corners_list_1):
        for j, obj_corners_2 in enumerate(obj_corners_list_2):
            iou_matrix[i, j] = obj_corners_iou(obj_corners_1, obj_corners_2)

    # Greedy matching: match highest IoU pairs first
    matches = []
    used_obj_corners_1 = set()
    used_obj_corners_2 = set()

    # Sort by IoU descending
    sorted_indices = np.dstack(
        np.unravel_index(np.argsort(-iou_matrix.ravel()), iou_matrix.shape)
    )[0]

    for idx in sorted_indices:
        obj_corners_1_idx = int(idx[0])
        obj_corners_2_idx = int(idx[1])
        if (
            obj_corners_1_idx not in used_obj_corners_1
            and obj_corners_2_idx not in used_obj_corners_2
        ):
            iou = float(iou_matrix[obj_corners_1_idx, obj_corners_2_idx])
            if iou > 0:
                matches.append((obj_corners_1_idx, obj_corners_2_idx, iou))
                used_obj_corners_1.add(obj_corners_1_idx)
                used_obj_corners_2.add(obj_corners_2_idx)

    return matches


def compute_area_rectangle(
    corners: Union[List[float], np.ndarray]
) -> float:
    """
    Compute area of oriented bounding box given its corner points. 
    
    Args:
        corners: List or array of 8 values [x1, y1, x2, y2, x3, y3, x4, y4]
    
    Returns:
        Area of the oriented bounding box
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = corners
    return 1/2 * abs(x1*y2 + x2*y3 + x3*y4 + x4*y1 - y1*x2 - y2*x3 - y3*x4 - y4*x1)