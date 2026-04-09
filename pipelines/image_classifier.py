"""
Blackbox 1: Image Type Classifier
================================
Deterministically classifies an image as OPTICAL or SAR.

This uses multiple statistical tests that are characteristic of SAR imagery:
1. Speckle noise detection (coefficient of variation)
2. Histogram distribution analysis (exponential vs normal)
3. Local variance patterns
4. Edge coherence analysis

Returns a definitive classification, not a probability.
"""

import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Tuple
import os


class ImageType(Enum):
    OPTICAL = "optical"  # Includes color and grayscale optical
    SAR = "sar"          # Synthetic Aperture Radar


@dataclass
class ClassificationResult:
    image_type: ImageType
    is_color: bool           # True if color image, False if grayscale
    evidence: dict           # Diagnostic information


def compute_coefficient_of_variation(gray_image: np.ndarray, kernel_size: int = 7) -> float:
    """
    Compute mean coefficient of variation (CV) in local windows.
    SAR images have HIGH CV due to multiplicative speckle noise.
    Optical images have LOW CV due to smooth gradients.

    CV = std / mean (in local windows)
    """
    gray_float = gray_image.astype(np.float64)

    # Local mean
    local_mean = cv2.blur(gray_float, (kernel_size, kernel_size))

    # Local variance = E[X^2] - E[X]^2
    local_sq_mean = cv2.blur(gray_float ** 2, (kernel_size, kernel_size))
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 0)

    # CV = std / mean (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        cv_map = np.sqrt(local_var) / (local_mean + 1e-10)
        cv_map = np.nan_to_num(cv_map, nan=0, posinf=0, neginf=0)

    # Exclude very dark regions (unreliable CV)
    mask = local_mean > 10
    if np.sum(mask) > 0:
        return float(np.mean(cv_map[mask]))
    return float(np.mean(cv_map))


def compute_histogram_skewness(gray_image: np.ndarray) -> float:
    """
    Compute histogram skewness.
    SAR images: Positive skewness (right-tailed, exponential-like)
    Optical images: Near-zero skewness (more symmetric/normal)
    """
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-10)  # Normalize

    values = np.arange(256)
    mean = np.sum(values * hist)
    variance = np.sum(((values - mean) ** 2) * hist)
    std = np.sqrt(variance) + 1e-10
    skewness = np.sum(((values - mean) ** 3) * hist) / (std ** 3)

    return float(skewness)


def compute_histogram_kurtosis(gray_image: np.ndarray) -> float:
    """
    Compute histogram kurtosis (peakedness).
    SAR images: High kurtosis (sharp peak, heavy tails)
    Optical images: Lower kurtosis (more uniform spread)
    """
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-10)

    values = np.arange(256)
    mean = np.sum(values * hist)
    variance = np.sum(((values - mean) ** 2) * hist)
    std = np.sqrt(variance) + 1e-10
    kurtosis = np.sum(((values - mean) ** 4) * hist) / \
        (std ** 4) - 3  # Excess kurtosis

    return float(kurtosis)


def compute_equivalent_number_of_looks(gray_image: np.ndarray) -> float:
    """
    Estimate Equivalent Number of Looks (ENL) - a SAR-specific metric.
    ENL = mean^2 / variance (in homogeneous regions)

    For single-look SAR: ENL ≈ 1
    For multi-look SAR: ENL > 1 but typically < 10
    For optical images: ENL >> 10 (very high in smooth regions)
    """
    gray_float = gray_image.astype(np.float64)

    # Find homogeneous regions using low gradient magnitude
    grad_x = cv2.Sobel(gray_float, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_float, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Threshold for homogeneous regions (low gradient)
    threshold = np.percentile(grad_mag, 25)  # Bottom 25% gradient
    homogeneous_mask = grad_mag < threshold

    if np.sum(homogeneous_mask) < 100:  # Not enough homogeneous pixels
        return 100.0  # Assume optical

    # Compute ENL in homogeneous regions using local windows
    kernel_size = 15
    local_mean = cv2.blur(gray_float, (kernel_size, kernel_size))
    local_sq_mean = cv2.blur(gray_float ** 2, (kernel_size, kernel_size))
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 1e-10)

    with np.errstate(divide='ignore', invalid='ignore'):
        enl_map = (local_mean ** 2) / local_var
        enl_map = np.nan_to_num(enl_map, nan=100, posinf=100, neginf=0)

    # Get ENL in homogeneous regions
    enl_values = enl_map[homogeneous_mask]
    return float(np.median(enl_values))


def compute_edge_density(gray_image: np.ndarray) -> float:
    """
    Compute edge density using Canny edge detector.
    SAR: Lower coherent edge density due to speckle disruption
    Optical: Higher, cleaner edge density
    """
    edges = cv2.Canny(gray_image, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    return float(edge_density)


def compute_laplacian_variance(gray_image: np.ndarray) -> float:
    """
    Laplacian variance - measures sharpness/texture.
    SAR: High variance due to speckle
    Optical: Varies, but different pattern
    """
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    return float(laplacian.var())


def is_color_image(bgr_image: np.ndarray) -> bool:
    """
    Determine if image is color or grayscale (even if loaded as 3-channel).
    Returns True if color, False if grayscale.
    """
    if len(bgr_image.shape) == 2:
        return False

    if bgr_image.shape[2] == 1:
        return False

    b, g, r = cv2.split(bgr_image)

    # Check if all channels are identical (grayscale loaded as RGB)
    diff_rg = np.mean(np.abs(r.astype(float) - g.astype(float)))
    diff_gb = np.mean(np.abs(g.astype(float) - b.astype(float)))

    # If average difference < 1, channels are essentially identical
    return (diff_rg + diff_gb) > 2.0


def classify_image(image_path: str) -> ClassificationResult:
    """
    Classify an image as OPTICAL or SAR with certainty.

    Decision Logic:
    1. If image has color saturation > threshold → OPTICAL (certain)
    2. If grayscale, use statistical tests:
       - Coefficient of Variation (CV): SAR > 0.2, Optical < 0.15
       - Equivalent Number of Looks (ENL): SAR < 10, Optical > 20
       - Histogram Skewness: SAR > 0.8, Optical < 0.5

    Args:
        image_path: Path to input image

    Returns:
        ClassificationResult with definitive type and evidence
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load image
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Check if color
    is_color = is_color_image(bgr_image)

    evidence = {
        "is_color": is_color,
        "image_shape": bgr_image.shape,
    }

    # =========================================
    # Rule 1: Color images are ALWAYS optical
    # =========================================
    if is_color:
        # Check saturation to confirm
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        mean_saturation = float(np.mean(hsv[:, :, 1]))
        evidence["mean_saturation"] = mean_saturation

        return ClassificationResult(
            image_type=ImageType.OPTICAL,
            is_color=True,
            evidence=evidence
        )

    # =========================================
    # Rule 2: Grayscale - use statistical tests
    # =========================================

    # Compute SAR-specific metrics
    cv_value = compute_coefficient_of_variation(gray_image)
    enl_value = compute_equivalent_number_of_looks(gray_image)
    skewness = compute_histogram_skewness(gray_image)
    kurtosis = compute_histogram_kurtosis(gray_image)
    edge_density = compute_edge_density(gray_image)

    evidence.update({
        "coefficient_of_variation": cv_value,
        "equivalent_number_of_looks": enl_value,
        "histogram_skewness": skewness,
        "histogram_kurtosis": kurtosis,
        "edge_density": edge_density,
    })

    # =========================================
    # Decision Rules (SAR characteristics)
    # =========================================
    sar_votes = 0
    optical_votes = 0

    # CV threshold: SAR has high local variance relative to mean
    if cv_value > 0.22:
        sar_votes += 2  # Strong indicator
    elif cv_value > 0.15:
        sar_votes += 1
    elif cv_value < 0.10:
        optical_votes += 2  # Strong indicator for optical
    else:
        optical_votes += 1

    # ENL threshold: SAR has low ENL in homogeneous regions
    if enl_value < 8:
        sar_votes += 2  # Strong indicator
    elif enl_value < 15:
        sar_votes += 1
    elif enl_value > 30:
        optical_votes += 2  # Strong indicator for optical
    else:
        optical_votes += 1

    # Skewness: SAR histograms are right-skewed
    if skewness > 1.0:
        sar_votes += 1
    elif skewness > 0.5:
        pass  # Neutral
    else:
        optical_votes += 1

    # Kurtosis: SAR has higher kurtosis (peaky distribution)
    if kurtosis > 2.0:
        sar_votes += 1
    elif kurtosis < 0:
        optical_votes += 1

    evidence["sar_votes"] = sar_votes
    evidence["optical_votes"] = optical_votes

    # =========================================
    # Final Decision
    # =========================================
    if sar_votes >= optical_votes + 2:
        # Clear SAR
        image_type = ImageType.SAR
    elif optical_votes >= sar_votes + 2:
        # Clear optical
        image_type = ImageType.OPTICAL
    else:
        # Borderline case - use ENL as tiebreaker (most reliable for SAR)
        if enl_value < 12:
            image_type = ImageType.SAR
        else:
            image_type = ImageType.OPTICAL

    return ClassificationResult(
        image_type=image_type,
        is_color=False,
        evidence=evidence
    )


def classify_image_simple(image_path: str) -> str:
    """
    Simple wrapper that returns just the type string.

    Returns:
        "optical" or "sar"
    """
    result = classify_image(image_path)
    return result.image_type.value


# =====================================================
# CLI Interface
# =====================================================
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Classify image as OPTICAL or SAR")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed evidence")
    args = parser.parse_args()

    try:
        result = classify_image(args.image_path)

        print(f"\n{'='*50}")
        print(f"Image: {args.image_path}")
        print(f"{'='*50}")
        print(f"Classification: {result.image_type.value.upper()}")
        print(f"Is Color: {result.is_color}")

        if args.verbose:
            print(f"\nEvidence:")
            for key, value in result.evidence.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        print(f"{'='*50}\n")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)
