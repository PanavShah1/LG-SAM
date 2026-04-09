import json
import math
import multiprocessing as mp
import random
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import astuple, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.earthmind import EarthMind
from models.qwen3_vl_8b import Qwen3VL8B
from models.qwen_image_edit import QwenImageEditGenerator
from models.remotesam import RemoteSAM
from models.sam3 import SAM3Segmenter
from pipelines.base import BasePipeline
from pipelines.earthmind import EarthMindPipeline
from pipelines.majority_voting import MajorityVotingPipeline
from pipelines.qwen_sgr import QwenSGRPipeline
from pipelines.remotesam import RemoteSAMPipeline
from pipelines.sgr import SGRPipeline
from pipelines.cgr import CGRPipeline
from pipelines.sam3 import SAM3Pipeline
from utils.bbox import denormalize_obj_corners, greedy_obb_matching


def get_model(device: str):
    # fmt: off
    # Models
    earthmind = EarthMind(device="cuda:0"); earthmind.load()
    # earthmind_ft = EarthMind(load_finetuned=True); earthmind_ft.load()
    # falcon = Falcon(); falcon.load()
    sam3 = SAM3Segmenter(device="cuda:0"); sam3.load()
    remotesam = RemoteSAM(device="cuda:0"); remotesam.load()
    qwen3_vl_8b = Qwen3VL8B(device="cuda:0"); qwen3_vl_8b.load()
    qwen_image_edit = QwenImageEditGenerator(device="cuda:1"); qwen_image_edit.load()

    # Pipelines
    earthmind_pipeline = EarthMindPipeline(); earthmind_pipeline.model = earthmind
    # falcon_pipeline = FalconPipeline(); falcon_pipeline.model = falcon
    sam3_pipeline = SAM3Pipeline(); sam3_pipeline.model = sam3
    remotesam_pipeline = RemoteSAMPipeline(); remotesam_pipeline.model = remotesam
    sgr_pipeline = SGRPipeline(); sgr_pipeline.qwen = qwen3_vl_8b; sgr_pipeline.detector = remotesam; sgr_pipeline.sam3 = sam3
    cgr_pipeline = CGRPipeline(); cgr_pipeline.qwen = qwen3_vl_8b; cgr_pipeline.detector = remotesam; cgr_pipeline.sam3 = sam3
    qwen_sgr_pipeline = QwenSGRPipeline(); qwen_sgr_pipeline.qwen_image_edit = qwen_image_edit; qwen_sgr_pipeline.sam3 = sam3; qwen_sgr_pipeline.remotesam = remotesam

    pipeline = MajorityVotingPipeline(
        pipelines=[
            earthmind_pipeline,
            # falcon_pipeline,
            sam3_pipeline,
            remotesam_pipeline,
            sgr_pipeline,
            cgr_pipeline,
            qwen_sgr_pipeline,
        ],
    )
    pipeline.load_models()
    # fmt: on

    return pipeline


@dataclass
class Result:
    iou: float
    gt_count: int
    pred_count: int
    score: float
    acc_05: float
    acc_07: float


TBatchMetadata = NewType("TBatchMetadata", Tuple[str, List[List[float]], str, int])


def visualize_result(
    image: Image.Image,
    results: list[dict],
    referring_sentence: str,
    gt_obj_corners_list: List[List[float]],
    output_path: str,
    generated_image: Optional[Image.Image] = None,
) -> None:
    """
    Visualize detection result with referring sentence in header.

    Args:
        image: Original image
        results: List of detection result dictionaries
        referring_sentence: The text prompt used for detection
        gt_obj_corners: List of ground truth object corners
        output_path: Path to save the visualization
        generated_image: Generated image (optional)
    """

    vis_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    vis_generated_image = None
    if generated_image is not None:
        # Resize to match original image size to ensure annotations align
        if generated_image.size != image.size:
            generated_image = generated_image.resize(
                image.size, Image.Resampling.BILINEAR
            )
        vis_generated_image = cv2.cvtColor(np.array(generated_image), cv2.COLOR_RGB2BGR)

    # Draw ground truth object corners (blue)
    for gt_obj_corners in gt_obj_corners_list:
        points = np.array(gt_obj_corners).reshape(4, 2).astype(np.int32)
        cv2.polylines(
            vis_image, [points], isClosed=True, color=(255, 0, 0), thickness=2
        )
        if vis_generated_image is not None:
            cv2.polylines(
                vis_generated_image,
                [points],
                isClosed=True,
                color=(255, 0, 0),
                thickness=2,
            )

    for result in results:
        # Draw oriented bbox (green)
        points = np.array(result["oriented_bbox"]).reshape(4, 2).astype(np.int32)
        cv2.polylines(
            vis_image, [points], isClosed=True, color=(0, 255, 0), thickness=2
        )
        if vis_generated_image is not None:
            cv2.polylines(
                vis_generated_image,
                [points],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2,
            )

        # Overlay mask
        mask = result.get("mask", None)
        if mask is not None:
            colored_mask = np.zeros_like(vis_image, dtype=np.uint8)
            colored_mask[mask == 1] = [0, 255, 0]
            vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)
            if vis_generated_image is not None:
                vis_generated_image = cv2.addWeighted(
                    vis_generated_image, 0.7, colored_mask, 0.3, 0
                )

    # Add header with text and legend
    header_height = 100
    header = np.zeros(
        (
            header_height,
            vis_image.shape[1]
            if vis_generated_image is None
            else vis_image.shape[1] * 2,
            3,
        ),
        dtype=np.uint8,
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 1

    # Dynamic font scale for referring sentence
    target_width = vis_image.shape[1] - 20  # 10px padding on each side
    if vis_generated_image is not None:
        target_width = (vis_image.shape[1] * 2) - 20  # 10px padding on each side

    text_size = cv2.getTextSize(referring_sentence, font, font_scale, font_thickness)[0]

    if text_size[0] > target_width:
        font_scale = font_scale * (target_width / text_size[0])

    # 1. Draw referring sentence
    cv2.putText(
        header,
        referring_sentence,
        (10, 40),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
    )

    # 2. Draw Legend
    legend_font_scale = 0.5
    legend_thickness = 1

    # Ground Truth - Blue (255, 0, 0)
    cv2.rectangle(header, (10, 60), (30, 80), (255, 0, 0), -1)
    cv2.putText(
        header,
        "Ground Truth",
        (40, 75),
        font,
        legend_font_scale,
        (255, 255, 255),
        legend_thickness,
    )

    # Prediction - Green (0, 255, 0)
    cv2.rectangle(header, (160, 60), (180, 80), (0, 255, 0), -1)
    cv2.putText(
        header,
        "Prediction",
        (190, 75),
        font,
        legend_font_scale,
        (255, 255, 255),
        legend_thickness,
    )

    # Stack header and image
    if vis_generated_image is not None:
        final_image = np.vstack([header, np.hstack([vis_image, vis_generated_image])])
    else:
        final_image = np.vstack([header, vis_image])

    cv2.imwrite(output_path, final_image)


def process_result(
    image: Image.Image,
    detection_results: List[dict],
    batch_metadata: TBatchMetadata,
    viz_dir: Optional[str] = None,
) -> List[Result]:
    ann_filename, gt_obj_corners, referring_sentence, obj_idx = batch_metadata

    if not detection_results:
        return []

    results: List[Result] = list(
        [
            Result(
                iou=match[2],
                gt_count=len(gt_obj_corners),
                pred_count=len(detection_results),
                score=match[2]
                * math.exp(-2.5 * abs(len(gt_obj_corners) - len(detection_results))),
                acc_05=int(match[2] >= 0.5),
                acc_07=int(match[2] >= 0.7),
            )
            for match in greedy_obb_matching(
                [
                    detection_result["oriented_bbox"]
                    for detection_result in detection_results
                ],
                gt_obj_corners,
            )
        ]
    )  # type: ignore

    if viz_dir:
        output_filename = f"{Path(ann_filename).stem}_{obj_idx}.png"
        output_path = str(Path(viz_dir) / output_filename)

        visualize_result(
            image,
            detection_results,
            referring_sentence,
            gt_obj_corners,
            output_path,
            generated_image=detection_results[0].get("generated_image", None),
        )

    return results


def flush_batch(
    batch_images: List[Image.Image],
    batch_prompts: List[str],
    batch_metadata: List[TBatchMetadata],
    pipeline: BasePipeline,
    results: Dict[str, List[Result]],
    viz_dir: Optional[str] = None,
) -> None:
    if not batch_images:
        return

    try:
        # Run batched inference
        batch_results = pipeline.process_batch(
            batch_images,  # type: ignore
            batch_prompts,
        )

        # Process results
        for i, detection_results in enumerate(batch_results):
            ann_identifier = batch_metadata[i][0]

            if ann_identifier not in results:
                results[ann_identifier] = []

            results[ann_identifier].extend(
                process_result(
                    batch_images[i], detection_results, batch_metadata[i], viz_dir
                )
            )
    except Exception as e:
        print(f"Error in batch processing: {e}\n{traceback.format_exc()}")
    finally:
        # Clear buffers
        batch_images.clear()
        batch_prompts.clear()
        batch_metadata.clear()


def process_annotation_batch(
    annotation_ids: List[Any],
    annotation_loader: Callable[[Any], Dict],
    images_dir: str,
    gpu_id: int,
    batch_size: int = 4,
    viz_dir: Optional[str] = None,
    progress_counter: Optional[Any] = None,
) -> List[Tuple[Any, List[Result]]]:
    """
    Process a batch of annotation objects using batched inference.

    Args:
        annotation_ids: List of annotation IDs
        annotation_loader: Function to load an annotation from an ID
        images_dir: Directory containing images
        gpu_id: GPU ID to use (0, 1, 2, etc.)
        progress_counter: Shared counter for progress tracking
        batch_size: Number of annotations to process in parallel
        viz_dir: Directory to save visualizations (optional)

    Returns:
        List of tuples (annotation_id, list_of_ious)
    """
    images_path = Path(images_dir)

    if viz_dir:
        Path(viz_dir).mkdir(parents=True, exist_ok=True)

    # Set CUDA device for this process
    if torch.cuda.is_available():
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
    else:
        device = "cpu"

    # Initialize pipeline for this GPU (load once for the batch)
    pipeline = get_model(device=device)
    pipeline.load_models()

    results: Dict[str, List[Result]] = {}  # Map filename to list of results

    # Buffers for batch processing
    batch_images = []
    batch_prompts = []
    batch_metadata = []  # (ann_filename, gt_obj_corners, referring_sentence, obj_idx)

    # Process each annotation file
    for ann_id in annotation_ids:
        ann_data = annotation_loader(ann_id)
        ann_id = Path(ann_data["image"]).stem

        image_path = images_path / ann_data["image"]

        if ann_id not in results:
            results[ann_id] = []

        # We need to open image to get size for GT object corners denormalization
        # And we pass the PIL image to the batch to avoid re-opening
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size

        # Flag to track if we processed this annotation
        ann_processed = False

        for obj_idx, obj in enumerate(ann_data.get("objects", [])):
            referring_sentence = obj.get("referring_sentence", "")
            if not referring_sentence:
                continue

            gt_obj_corners = obj.get("obj_corner", None)
            if gt_obj_corners is None:
                continue

            gt_obj_corners = [
                denormalize_obj_corners(gt_obj_corner, image_width, image_height)
                for gt_obj_corner in gt_obj_corners
            ]
            # gt_obj_corners = denormalize_obj_corners(
            #     gt_obj_corners, image_width, image_height
            # )

            # print("gt_obj_corners:", gt_obj_corners)

            # Add to batch
            batch_images.append(image)  # Use loaded PIL image
            batch_prompts.append(referring_sentence)
            # TODO: Add support for multiple ground truth object corners
            batch_metadata.append((ann_id, gt_obj_corners, referring_sentence, obj_idx))
            ann_processed = True

            if len(batch_images) >= batch_size:
                flush_batch(
                    batch_images,
                    batch_prompts,
                    batch_metadata,
                    pipeline,
                    results,
                    viz_dir,
                )

        if ann_processed and progress_counter is not None:
            progress_counter.value += 1

    # Process remaining items
    flush_batch(
        batch_images,
        batch_prompts,
        batch_metadata,
        pipeline,
        results,
        viz_dir,
    )

    return list(results.items())


def process_annotation_objects(
    annotation_objects: List[Any],
    images_dir: str,
    gpu_id: int,
    batch_size: int = 4,
    viz_dir: Optional[str] = None,
    progress_counter: Optional[Any] = None,
) -> List[Tuple[Any, List[Result]]]:
    return process_annotation_batch(
        list(range(len(annotation_objects))),
        lambda x: annotation_objects[x],
        images_dir,
        gpu_id,
        batch_size,
        viz_dir,
        progress_counter,
    )


def get_process_annotation_files(
    annotations_dir: str,
) -> Callable[[Any], List[Tuple[Any, List[Result]]]]:
    def loader(ann_id: Any) -> Dict:
        with open(Path(annotations_dir) / ann_id, "r") as f:  # type: ignore
            return json.load(f)

    def process_batch(
        annotation_files: List[str],
        images_dir: str,
        gpu_id: int,
        batch_size: int = 4,
        viz_dir: Optional[str] = None,
    ) -> List[Tuple[Any, List[Result]]]:
        return process_annotation_batch(
            annotation_files, loader, images_dir, gpu_id, batch_size, viz_dir
        )

    return process_batch  # type: ignore[return-value]


def test_pipeline_on_vrsbench(
    annotations_dir: Optional[str] = "data/vrsbench/annotations",
    annotations_file: Optional[str] = "data/vrsbench/annotations.json",
    images_dir: str = "data/vrsbench/images",
    num_gpus: Optional[int] = None,
    num_workers_per_gpu: int = 2,
    num_images: Optional[int] = None,
    batch_size: int = 4,
    viz_dir: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Test the pipeline on vrsbench data and calculate mean IoU with parallel processing.

    Args:
        annotations_dir: Directory containing annotation JSON files (used if annotations_file is None)
        annotations_file: JSON file containing an array of annotation objects (alternative to annotations_dir)
        images_dir: Directory containing images
        num_gpus: Number of GPUs to use (None = auto-detect)
        num_workers_per_gpu: Number of worker processes per GPU
        batch_size: Batch size for inference
        viz_dir: Directory to save visualizations

    Returns:
        Mean IoU score
    """
    use_annotations_file = annotations_file is not None
    annotation_objects = None
    num_annotations = 0

    if use_annotations_file:
        # Read annotation objects from JSON file
        annotations_file_path = Path(annotations_file)
        if not annotations_file_path.exists():
            raise ValueError(f"Annotations file not found: {annotations_file}")

        try:
            with open(annotations_file_path, "r") as f:
                annotation_objects = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in annotations file: {e}")

        if not isinstance(annotation_objects, list):
            raise ValueError("Annotations file must contain a JSON array")

        if len(annotation_objects) == 0:
            raise ValueError(f"No annotation objects found in {annotations_file}")

        if num_images is not None:
            print(f"Sampling {num_images} random annotation objects")
            annotation_objects = random.sample(annotation_objects, num_images)

        num_annotations = len(annotation_objects)
    else:
        if annotations_dir is None:
            raise ValueError(
                "Either annotations_dir or annotations_file must be provided"
            )

        annotations_path = Path(annotations_dir)
        if not annotations_path.exists():
            raise ValueError(f"Annotations directory not found: {annotations_dir}")

        # Find all annotation files starting with "P"
        annotation_objects = sorted(annotations_path.glob("*.json"))

        if len(annotation_objects) == 0:
            raise ValueError(
                f"No annotation files found starting with 'P' in {annotations_dir}"
            )

        if num_images is not None:
            print(f"Sampling {num_images} random annotation files")
            annotation_objects = random.sample(annotation_objects, num_images)

        num_annotations = len(annotation_objects)

    print(f"Found {num_annotations} annotation objects")

    if viz_dir:
        print(f"Saving visualizations to {viz_dir}")
        Path(viz_dir).mkdir(parents=True, exist_ok=True)

    # Detect available GPUs
    if torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        if num_gpus is None:
            num_gpus = available_gpus
        else:
            num_gpus = min(num_gpus, available_gpus)
        print(f"Using {num_gpus} GPU(s) out of {available_gpus} available")
    else:
        print("No CUDA GPUs available, using CPU")
        num_gpus = 1

    # Create batches based on whether we're using annotation files or objects
    max_workers = num_gpus * num_workers_per_gpu
    tasks = []

    # Group annotation objects by GPU (round-robin)
    objects_per_gpu = [[] for _ in range(num_gpus)]
    for idx, ann_obj in enumerate(annotation_objects):
        gpu_id = idx % num_gpus
        objects_per_gpu[gpu_id].append(ann_obj)

    # Create batches: distribute objects across workers for each GPU
    for gpu_id in range(num_gpus):
        gpu_objects = objects_per_gpu[gpu_id]
        # Split objects for this GPU across workers
        objects_per_worker = len(gpu_objects) // num_workers_per_gpu
        for worker_idx in range(num_workers_per_gpu):
            start_idx = worker_idx * objects_per_worker
            if worker_idx == num_workers_per_gpu - 1:
                # Last worker gets remaining objects
                batch_objects = gpu_objects[start_idx:]
            else:
                batch_objects = gpu_objects[start_idx : start_idx + objects_per_worker]

            if batch_objects:  # Only add non-empty batches
                tasks.append(
                    (
                        batch_objects,
                        images_dir,
                        gpu_id,
                        batch_size,
                        viz_dir,
                    )
                )

    print(
        f"Processing {len(annotation_objects)} annotation objects in {len(tasks)} batches across {num_gpus} GPU(s)"
    )

    # Process in parallel using ProcessPoolExecutor
    all_results = []
    results_dict: Dict[str, List[Result]] = {}

    # Create shared counter for progress tracking using Manager
    # Manager allows sharing objects across processes with 'spawn' method
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    progress_counter = manager.Value("i", 0)  # Shared integer counter

    # Add progress counter to all tasks and determine processing function
    tasks_with_progress = [(*args, progress_counter) for args in tasks]
    proc_func = (
        process_annotation_objects
        if use_annotations_file
        else get_process_annotation_files(annotations_dir)  # type: ignore[arg-type]
    )

    # Use ProcessPoolExecutor with workers equal to number of GPUs * workers per GPU
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        # Submit all batch tasks
        future_to_task = {}
        for task in tasks_with_progress:
            future = executor.submit(proc_func, *task)  # type: ignore[call-arg]
            future_to_task[future] = task

        # Create progress bar that updates based on shared counter
        pbar = tqdm(total=num_annotations, desc="Processing objects", unit="object")
        last_count = 0
        stop_polling = threading.Event()

        # Function to poll and update progress bar
        def update_progress():
            nonlocal last_count
            while not stop_polling.is_set():
                # Manager.Value() proxies handle synchronization internally
                current_count = progress_counter.value
                if current_count > last_count:
                    pbar.update(current_count - last_count)
                    last_count = current_count
                time.sleep(0.1)  # Poll every 100ms

        # Start progress polling thread
        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()

        try:
            # Process completed batches
            for future in as_completed(future_to_task):
                try:
                    batch_results = future.result()
                    for ann_filename, image_results in batch_results:
                        if image_results:
                            results_dict[ann_filename] = (
                                results_dict.get(ann_filename, []) + image_results
                            )
                except Exception as e:
                    task = future_to_task[future]
                    print(
                        f"Error processing batch on GPU {task[2]}: {e}\n{traceback.format_exc()}"
                    )

            # Stop polling and do final update
            stop_polling.set()
            progress_thread.join(timeout=1.0)

            # Final update to ensure progress bar reaches 100%
            final_count = progress_counter.value
            if final_count > last_count:
                pbar.update(final_count - last_count)
        finally:
            stop_polling.set()
            pbar.close()

    # Print summary
    for ann_filename, image_results in sorted(results_dict.items()):
        if not image_results:
            continue

        # Unpack results
        results_arr = np.array([astuple(result) for result in image_results])
        all_results.extend(results_arr)
        mean_iou = np.mean(results_arr[:, 0])
        score = np.mean(results_arr[:, 3])

        if len(image_results) > 0:
            print(
                f"{ann_filename}: {len(image_results)} objects matched, "
                f"mean IoU: {mean_iou:.4f}, score: {score:.4f} (pred: {results_arr[0, 2]}, gt: {results_arr[0, 1]})"
            )
        else:
            print(
                f"{ann_filename}: {len(image_results)} objects matched, "
                f"mean IoU: {mean_iou:.4f}, score: {score:.4f} (no results)"
            )

    if len(all_results) == 0:
        print("Warning: No IoU values computed!")
        return 0.0, 0.0

    all_results_arr = np.array(all_results)
    mean_final_iou = np.mean(all_results_arr[:, 0])
    mean_final_score = np.mean(all_results_arr[:, 3])
    count_difference = all_results_arr[:, 1] - all_results_arr[:, 2]
    acc_05 = np.mean(all_results_arr[:, 4])
    acc_07 = np.mean(all_results_arr[:, 5])

    print(f"\nOverall: {len(all_results)} matched detections")
    print(
        f"Average count difference (pred - gt): {np.mean(count_difference):.4f} ({np.mean(all_results_arr[:, 2]):.4f} - {np.mean(all_results_arr[:, 1]):.4f})"
    )
    print(f"Average absolute count difference: {np.mean(np.abs(count_difference)):.4f}")
    print(f"Variance of count difference (pred - gt): {np.var(count_difference):.4f}")
    print(f"Accuracy @ IoU 0.5: {acc_05:.4f}")
    print(f"Accuracy @ IoU 0.7: {acc_07:.4f}")

    return mean_final_iou.item(), mean_final_score.item()


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This must be done before creating any ProcessPoolExecutor
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set, ignore
        pass

    import argparse

    parser = argparse.ArgumentParser(description="Test pipeline on vrsbench data")

    # Create mutually exclusive group for annotations_dir and annotations_file
    annotations_group = parser.add_mutually_exclusive_group()
    annotations_group.add_argument(
        "--annotations_dir",
        type=str,
        default="data/vrsbench/annotations",
        help="Directory containing annotation JSON files",
    )
    annotations_group.add_argument(
        "--annotations_file",
        type=str,
        help="JSON file containing an array of annotation file paths (mutually exclusive with annotations_dir)",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="data/vrsbench/images",
        help="Directory containing images",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect all available)",
    )
    parser.add_argument(
        "--num_workers_per_gpu",
        type=int,
        default=1,
        help="Number of worker processes per GPU (default: 2)",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Number of images to process (default: all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)",
    )
    parser.add_argument(
        "--viz_dir",
        type=str,
        default=None,
        help="Directory to save visualizations (default: None, no visualization)",
    )

    args = parser.parse_args()

    assert args.num_images is None or args.num_images > 0
    assert args.num_gpus is None or args.num_gpus > 0
    assert args.num_workers_per_gpu is None or args.num_workers_per_gpu > 0
    assert args.batch_size > 0
    # Validate that either annotations_file or annotations_dir is provided (mutually exclusive)
    # Note: argparse's mutually_exclusive_group prevents both from being explicitly provided,
    # but annotations_dir may still have its default value even if annotations_file is provided.
    # The function will use annotations_file if provided and ignore annotations_dir.
    if args.annotations_file is not None:
        assert Path(args.annotations_file).exists(), (
            f"Annotations file not found: {args.annotations_file}"
        )
        # If annotations_file is provided, annotations_dir should not be explicitly set
        # (mutually_exclusive_group handles this, but we ensure annotations_dir is None for clarity)
        args.annotations_dir = None
    else:
        assert (
            args.annotations_dir is not None and Path(args.annotations_dir).exists()
        ), f"Annotations directory not found: {args.annotations_dir}"
    assert args.images_dir is not None and Path(args.images_dir).exists()

    mean_iou, mean_score = test_pipeline_on_vrsbench(
        annotations_dir=args.annotations_dir,
        annotations_file=args.annotations_file,
        images_dir=args.images_dir,
        num_gpus=args.num_gpus,
        num_workers_per_gpu=args.num_workers_per_gpu,
        num_images=args.num_images,
        batch_size=args.batch_size,
        viz_dir=args.viz_dir,
    )

    print(f"\nFinal Mean IoU: {mean_iou:.4f}, score: {mean_score:.4f}")
