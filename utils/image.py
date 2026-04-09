"""Image preprocessing utilities."""

from typing import List, Tuple, Union

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def load_image(image_input: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    """
    Load an image from various input formats.

    Args:
        image_input: Can be a file path (str), PIL Image, or numpy array

    Returns:
        PIL Image in RGB format
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3 and image_input.shape[2] == 3:
            image = Image.fromarray(image_input).convert("RGB")
        else:
            raise ValueError(f"Unsupported numpy array shape: {image_input.shape}")
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    return image


def preprocess_image(
    image: Image.Image, size: int = 800
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Preprocess image for Grounded DINO model.

    Args:
        image: PIL Image in RGB format
        size: Target size for resizing (default 800)

    Returns:
        Tuple of (preprocessed_tensor, original_size)
    """
    original_size = image.size  # (width, height)

    # Resize while maintaining aspect ratio
    w, h = original_size
    scale = size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize image
    image_resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Convert to tensor and normalize
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    tensor = transform(image_resized)

    # Pad to make it square
    pad_w = size - new_w
    pad_h = size - new_h
    tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), value=0)

    return tensor.unsqueeze(0), original_size


def batch_preprocess(
    images: List[Image.Image], size: int = 800
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """
    Preprocess a batch of images.

    Args:
        images: List of PIL Images
        size: Target size for resizing (default 800)

    Returns:
        Tuple of (batched_tensor, list_of_original_sizes)
    """
    tensors = []
    original_sizes = []

    for image in images:
        tensor, orig_size = preprocess_image(image, size)
        tensors.append(tensor)
        original_sizes.append(orig_size)

    # Stack tensors into a batch
    batched_tensor = torch.cat(tensors, dim=0)

    return batched_tensor, original_sizes


def overlay_masks(image: Image.Image, masks: torch.Tensor) -> Image.Image:
    """
    Overlay masks on an image.

    Args:
        image: PIL Image
        masks: Tensor of masks

    Returns:
        PIL Image with masks overlaid
    """

    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)  # type: ignore

    n_masks = masks.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image


def remove_haziness(image: Image.Image) -> Image.Image:
    """
    Remove haziness from an image.

    Args:
        image: PIL Image

    Returns:
        PIL Image with haziness removed
    """

    # Apply contrast-limited adaptive histogram equalization (CLAHE)
    lab = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)  # noqa: E741
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    clahe_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # Optional sharpening
    sharp = cv2.GaussianBlur(clahe_img, (0, 0), 3)
    sharp = cv2.addWeighted(clahe_img, 1.6, sharp, -0.6, 0)

    return Image.fromarray(sharp)
