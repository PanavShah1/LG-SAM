import cv2
import numpy as np

from pipelines.remotesam import RemoteSAMPipeline
from pipelines.sam3 import SAM3Pipeline
from pipelines.sgr import SGRPipeline
# from pipelines.cgr import CGRPipeline
from pipelines.earthmind import EarthMindPipeline
from pipelines.majority_voting import MajorityVotingPipeline

image_path = "/home/eesel/panav.shah/coding_stuff/input_images/img1.jpeg"
text_prompt = "roads"

pipeline = RemoteSAMPipeline()
pipeline = MajorityVotingPipeline([
    RemoteSAMPipeline(),
    SAM3Pipeline(),
    SGRPipeline(),
    EarthMindPipeline(),
])

results = pipeline.process_image(
    image=image_path,
    text_prompt=text_prompt,
    # box_threshold=0.4,
    # text_threshold=0.3,
)

print(f"\nFound {len(results)} objects:")
for i, result in enumerate(results):
    print(f"\nObject {i + 1}:")
    print(f"  Score: {result['score']:.3f}")
    points = np.array(result["oriented_bbox"]).reshape(4, 2).astype(np.int32)
    print(f"  Oriented bbox: points=({points}), ")
    # print(f"  Mask shape: {result['mask'].shape}")

# Draw the results on the image
img = cv2.imread(image_path)

# Visualize the mask

for i, result in enumerate(results):
    image = cv2.imread(image_path)
    # mask = result["mask"]
    # colored_mask = np.zeros_like(image, dtype=np.uint8)
    # colored_mask[mask == 1] = [0, 255, 0]  # Set green color for masked area

    # Blend the original image and the colored mask
    # alpha (image transparency), beta (mask transparency), gamma (constant added to result)
    # overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

    points = np.array(result["oriented_bbox"]).reshape(4, 2).astype(np.int32)
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imwrite(f"/home/eesel/panav.shah/coding_stuff/output_images/result_{i}.png", image)
