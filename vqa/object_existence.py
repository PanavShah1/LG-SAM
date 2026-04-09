import json
import sys

from PIL import Image
from tqdm import tqdm

from models.earthmind import EarthMind

if len(sys.argv) != 4:
    print("Usage: python temp.py <annotations_file> <output_file> <images_dir>")
    sys.exit(1)

earthmind = EarthMind(load_finetuned=True)
earthmind.load()


with open(sys.argv[1], "r") as f:
    annotations_entries = json.load(f)


def batch_generate(batch_items):
    images = []
    prompts = []
    for item in batch_items:
        qa, image_path = item
        question = qa["question"]
        # prompt = BOOLEAN_ANSWER_PROMPT.format(question=question)
        full_image_path = f"{sys.argv[3]}/{image_path}"
        image = Image.open(full_image_path).convert("RGB")
        images.append(image)
        prompts.append(question)
    answers = [
        earthmind.answer(image, prompt) for image, prompt in zip(images, prompts)
    ]
    return answers


BATCH_SIZE = 32
batch_items = []  # List of (qa, image_path) tuples

# Collect all objects with their image paths
for entry in annotations_entries:
    image_path = entry["image"]
    for qa in entry["qa_pairs"]:
        if qa["type"] == "object existence":
            batch_items.append((qa, image_path))

print(f"Total number of items to process: {len(batch_items)}")

# Process in batches across all entries
for i in tqdm(range(0, len(batch_items), BATCH_SIZE)):
    batch = batch_items[i : i + BATCH_SIZE]
    answers = batch_generate(batch)
    for (qa, image_path), answer in zip(batch, answers):
        if (
            "true" in answer.lower()
            or "false" in answer.lower()
            or "yes" in answer.lower()
            or "no" in answer.lower()
        ):
            qa["our_answer"] = "true" in answer.lower() or "yes" in answer.lower()
        else:
            qa["our_answer"] = False
            print(
                f"Warning: Model generated a non-boolean answer for {qa['question']} (image: {image_path}): {answer}"
            )

with open(sys.argv[2], "w") as f:
    json.dump(annotations_entries, f)

total_correct = 0
total_count = 0
for entry in annotations_entries:
    for qa in entry["qa_pairs"]:
        if qa["type"] == "object existence":
            if qa["answer"].lower() in ["yes", "true"]:
                answer = True
            elif qa["answer"].lower() in ["no", "false"]:
                answer = False
            else:
                answer = None
                print(
                    f"Warning: Invalid answer for {qa['question']} (image: {entry['image']}): {qa['answer']}"
                )

            if answer is not None:
                total_correct += int(answer == qa["our_answer"])
                total_count += 1

print(f"Accuracy: {total_correct / total_count} ({total_correct}/{total_count})")
